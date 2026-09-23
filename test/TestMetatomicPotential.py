import os
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

torch = pytest.importorskip("torch", reason="torch is not installed")
mta = pytest.importorskip("metatomic.torch", reason="metatomic-torch is not installed")
from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa: E402

platform_ints = range(mm.Platform.getNumPlatforms())
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _energy_map(energy: torch.Tensor, key: str = "energy") -> TensorMap:
    block = TensorBlock(
        values=energy,
        samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    return TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])


def _energy_outputs(
    energy: torch.Tensor, outputs: Dict[str, mta.ModelOutput]
) -> Dict[str, TensorMap]:
    # Annotations matter: without them TorchScript treats key as Tensor and
    # rejects str comparisons / startswith.
    result: Dict[str, TensorMap] = {}
    for key in outputs.keys():
        if key == "energy" or (len(key) >= 7 and key[0:7] == "energy/"):
            result[key] = _energy_map(energy, key)
    return result


def _mask_positions(
    positions: torch.Tensor, selected_atoms: Optional[Labels]
) -> torch.Tensor:
    if selected_atoms is None:
        return positions
    indices = selected_atoms.column("atom")
    return positions[indices]


class HarmonicModel(torch.nn.Module):
    def __init__(self, force_constant: float, equilibrium_positions: torch.Tensor):
        super().__init__()
        self.force_constant = force_constant
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def _energy(
        self, systems: List[mta.System], selected_atoms: Optional[Labels] = None
    ) -> torch.Tensor:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            pos = _mask_positions(system.positions, selected_atoms)
            eq = _mask_positions(self.equilibrium_positions, selected_atoms)
            energy[i] += torch.sum(self.force_constant * (pos - eq) ** 2)
        return energy

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return _energy_outputs(self._energy(systems, selected_atoms), outputs)


class HarmonicNLModel(HarmonicModel):
    def __init__(self, force_constant, equilibrium_positions, cutoff=0.4):
        # cutoff is in the model's length unit (nm).
        super().__init__(force_constant, equilibrium_positions)
        self._nl_options = mta.NeighborListOptions(cutoff, False, True)

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl_options)
            energy[i] += neighbors.values.sum() * 0.0
            pos = _mask_positions(system.positions, selected_atoms)
            eq = _mask_positions(self.equilibrium_positions, selected_atoms)
            energy[i] += torch.sum(self.force_constant * (pos - eq) ** 2)
        return _energy_outputs(energy, outputs)


class HarmonicChargeModel(HarmonicModel):
    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {"charge": mta.ModelOutput(unit="e", sample_kind="system")}


class HarmonicSpinModel(HarmonicModel):
    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {
            "spin_multiplicity": mta.ModelOutput(unit="", sample_kind="system")
        }


class PerAtomChargeModel(HarmonicModel):
    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {"charge": mta.ModelOutput(unit="e", sample_kind="atom")}


class HarmonicUQModel(HarmonicModel):
    def __init__(self, force_constant, equilibrium_positions, uncertainty):
        super().__init__(force_constant, equilibrium_positions)
        self.uncertainty = uncertainty

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        result = _energy_outputs(self._energy(systems, selected_atoms), outputs)
        if "energy_uncertainty" not in outputs:
            return result
        n_atoms = systems[0].positions.shape[0]
        if selected_atoms is not None:
            atom_indices = selected_atoms.column("atom")
            n_atoms = len(atom_indices)
            samples = torch.stack(
                [
                    torch.zeros(n_atoms, dtype=torch.int32),
                    atom_indices.to(dtype=torch.int32),
                ],
                dim=1,
            )
        else:
            samples = torch.stack(
                [
                    torch.zeros(n_atoms, dtype=torch.int32),
                    torch.arange(n_atoms, dtype=torch.int32),
                ],
                dim=1,
            )
        values = torch.full(
            (n_atoms, 1), self.uncertainty, dtype=systems[0].positions.dtype
        )
        block = TensorBlock(
            values=values,
            samples=Labels(["system", "atom"], samples),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        result["energy_uncertainty"] = TensorMap(
            keys=Labels("_", torch.tensor([[0]])), blocks=[block]
        )
        return result


def _export_model(path, model, atomic_types, interaction_range=0.0, outputs=None):
    if outputs is None:
        outputs = {"energy": mta.ModelOutput(unit="eV", sample_kind="system")}
    capabilities = mta.ModelCapabilities(
        outputs=outputs,
        atomic_types=sorted(set(atomic_types)),
        interaction_range=interaction_range,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )
    wrapper = mta.AtomisticModel(model.eval(), mta.ModelMetadata(), capabilities)
    wrapper.save(path)
    return path


def _export_harmonic(path, positions_nm, atomic_types, force_constant=1.0):
    model = HarmonicModel(
        force_constant,
        torch.tensor(positions_nm, dtype=torch.float64),
    )
    return _export_model(path, model, atomic_types)


def _direct_energy_forces(model_path, numbers, positions_nm, energy_key="energy"):
    model = mta.load_atomistic_model(model_path)
    dtype = torch.float64
    types = torch.tensor(numbers, dtype=torch.int32)
    pos = torch.tensor(positions_nm, dtype=dtype, requires_grad=True)
    cell = torch.zeros((3, 3), dtype=dtype)
    pbc = torch.tensor([False, False, False])
    system = mta.System(types, pos, cell, pbc)
    options = mta.ModelEvaluationOptions(
        length_unit="nm",
        outputs={energy_key: mta.ModelOutput(unit="kJ/mol", sample_kind="system")},
    )
    energy = model([system], options, False)[energy_key].block().values.sum()
    energy.backward()
    return (
        float(energy.detach()),
        (-pos.grad).detach().numpy(),
    )


@pytest.fixture(scope="module")
def harmonic_toluene():
    pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
    positions = np.asarray(pdb.getPositions(asNumpy=True), dtype=np.float64)
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "harmonic.pt")
        _export_harmonic(path, positions, numbers)
        yield pdb, numbers, positions, path


def _energy(context):
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMetatomicPotential:
    def testCreatePureMLSystem(self, platform_int, harmonic_toluene):
        pdb, numbers, positions, model_path = harmonic_toluene
        displaced = positions + 0.01
        potential = MLPotential("metatomic", modelPath=model_path, device="cpu")
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(displaced * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy_ml = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces_ml = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        energy_ref, forces_ref = _direct_energy_forces(model_path, numbers, displaced)
        assert np.isclose(energy_ref, energy_ml, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(forces_ref, forces_ml, rtol=1e-5, atol=1e-6)

    def testCreateMixedSystem(self, platform_int, harmonic_toluene):
        _, _, positions, _ = harmonic_toluene
        prmtop = app.AmberPrmtopFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7")
        )
        ml_atoms = list(range(15))
        all_numbers = [
            atom.element.atomic_number for atom in prmtop.topology.atoms()
        ]
        # Equilibrium only for the ML subset; selected_atoms masks to those indices.
        # atomic_types must cover the full Topology (including solvent).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-mixed.pt")
            _export_harmonic(path, positions, all_numbers)
            mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            mixed_system = potential.createMixedSystem(
                prmtop.topology,
                mm_system,
                ml_atoms,
                interpolate=False,
                mlLongRange=False,
            )
            interp_system = potential.createMixedSystem(
                prmtop.topology,
                mm_system,
                ml_atoms,
                interpolate=True,
                mlLongRange=False,
            )
            platform = mm.Platform.getPlatform(platform_int)
            mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform)
            mixed_context = mm.Context(
                mixed_system, mm.VerletIntegrator(0.001), platform
            )
            interp_context = mm.Context(
                interp_system, mm.VerletIntegrator(0.001), platform
            )
            mm_context.setPositions(inpcrd.positions)
            mixed_context.setPositions(inpcrd.positions)
            interp_context.setPositions(inpcrd.positions)
            assert np.isclose(
                _energy(mixed_context), _energy(interp_context), rtol=1e-5
            )
            interp_context.setParameter("lambda_interpolate", 0)
            assert np.isclose(_energy(mm_context), _energy(interp_context), rtol=1e-5)
            python_forces = [
                f for f in mixed_system.getForces() if isinstance(f, mm.PythonForce)
            ]
            assert python_forces
            assert python_forces[0].usesPeriodicBoundaryConditions()
            # Empty => applies to all atoms; OpenMM returns a tuple.
            assert len(python_forces[0].getParticles()) == 0

    def testNeighborList(self, platform_int, harmonic_toluene):
        pytest.importorskip("vesin", reason="vesin is not installed")
        pdb, _, positions, _ = harmonic_toluene
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-nl.pt")
            _export_model(
                path,
                HarmonicNLModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
                interaction_range=0.4,
            )
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            system = potential.createSystem(pdb.topology)
            platform = mm.Platform.getPlatform(platform_int)
            context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
            context.setPositions((positions + 0.01) * unit.nanometer)
            energy = _energy(context)
            assert np.isfinite(energy)
            assert energy != 0.0

    def testChargeAndCheckConsistency(self, platform_int, harmonic_toluene):
        pdb, _, positions, _ = harmonic_toluene
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-charge.pt")
            _export_model(
                path,
                HarmonicChargeModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
            )
            potential = MLPotential(
                "metatomic",
                modelPath=path,
                device="cpu",
                checkConsistency=True,
            )
            system = potential.createSystem(
                pdb.topology, charge=0, multiplicity=1
            )
            platform = mm.Platform.getPlatform(platform_int)
            context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
            context.setPositions(positions * unit.nanometer)
            assert np.isclose(_energy(context), 0.0, atol=1e-8)

    def testPerAtomChargeRejected(self, platform_int, harmonic_toluene):
        pdb, _, positions, _ = harmonic_toluene
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-atom-charge.pt")
            _export_model(
                path,
                PerAtomChargeModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
            )
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            with pytest.raises(ValueError, match="sample_kind='atom'"):
                potential.createSystem(pdb.topology)

    def testPartialPbc(self, platform_int, harmonic_toluene):
        pdb, _, positions, model_path = harmonic_toluene
        box = [mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2)]
        box = [v * unit.nanometer for v in box]
        potential = MLPotential("metatomic", modelPath=model_path, device="cpu")
        system = potential.createSystem(pdb.topology, pbc=(True, True, False))
        system.setDefaultPeriodicBoxVectors(*box)
        python_forces = [
            f for f in system.getForces() if isinstance(f, mm.PythonForce)
        ]
        assert python_forces[0].usesPeriodicBoundaryConditions()
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPeriodicBoxVectors(*box)
        context.setPositions(positions * unit.nanometer)
        assert np.isfinite(_energy(context))

    def testAgreesWithASE(self, platform_int, harmonic_toluene):
        pytest.importorskip("ase", reason="ASE is not installed")
        pytest.importorskip("metatomic_ase", reason="metatomic-ase is not installed")
        from metatomic_ase import MetatomicCalculator

        pdb, _, positions, model_path = harmonic_toluene
        displaced = positions + 0.005
        platform = mm.Platform.getPlatform(platform_int)
        native = MLPotential("metatomic", modelPath=model_path, device="cpu")
        native_system = native.createSystem(pdb.topology)
        native_context = mm.Context(
            native_system, mm.VerletIntegrator(0.001), platform
        )
        native_context.setPositions(displaced * unit.nanometer)
        native_energy = _energy(native_context)

        calculator = MetatomicCalculator(
            model_path, device="cpu", do_gradients_with_energy=True
        )
        ase_system = MLPotential("ase").createSystem(
            pdb.topology, calculator=calculator
        )
        ase_context = mm.Context(ase_system, mm.VerletIntegrator(0.001), platform)
        ase_context.setPositions(displaced * unit.nanometer)
        assert np.isclose(native_energy, _energy(ase_context), rtol=1e-5, atol=1e-6)


class TestMetatomicPotentialOptions:
    def testNonConservativeRequiresOutput(self, harmonic_toluene):
        pdb, _, _, model_path = harmonic_toluene
        potential = MLPotential(
            "metatomic",
            modelPath=model_path,
            device="cpu",
            nonConservative=True,
        )
        with pytest.raises(ValueError, match="non_conservative_force"):
            potential.createSystem(pdb.topology)

    def testVariants(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-variant.pt")
            _export_model(
                path,
                HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
                outputs={
                    "energy/pbe": mta.ModelOutput(unit="eV", sample_kind="system")
                },
            )
            missing = MLPotential("metatomic", modelPath=path, device="cpu")
            with pytest.raises(ValueError, match="no default variant"):
                missing.createSystem(pdb.topology)
            potential = MLPotential(
                "metatomic",
                modelPath=path,
                device="cpu",
                variants={"energy": "pbe"},
            )
            system = potential.createSystem(pdb.topology)
            context = mm.Context(system, mm.VerletIntegrator(0.001))
            displaced = positions + 0.01
            context.setPositions(displaced * unit.nanometer)
            energy_ml = _energy(context)
            energy_ref, _ = _direct_energy_forces(
                path, numbers, displaced, energy_key="energy/pbe"
            )
            assert np.isclose(energy_ref, energy_ml, rtol=1e-6, atol=1e-8)

    def testUncertaintyThreshold(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-uq.pt")
            # Model stores uncertainty in eV; engine requests kJ/mol conversion.
            _export_model(
                path,
                HarmonicUQModel(
                    1.0, torch.tensor(positions, dtype=torch.float64), 0.5
                ),
                numbers,
                outputs={
                    "energy": mta.ModelOutput(unit="eV", sample_kind="system"),
                    "energy_uncertainty": mta.ModelOutput(
                        unit="eV", sample_kind="atom"
                    ),
                },
            )
            quiet = MLPotential(
                "metatomic",
                modelPath=path,
                device="cpu",
                uncertaintyThreshold=None,
            )
            system = quiet.createSystem(pdb.topology)
            context = mm.Context(system, mm.VerletIntegrator(0.001))
            context.setPositions(positions * unit.nanometer)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _energy(context)
            assert not any(
                "uncertainties" in str(w.message) for w in caught
            )

            loud = MLPotential("metatomic", modelPath=path, device="cpu")
            system = loud.createSystem(pdb.topology)
            context = mm.Context(system, mm.VerletIntegrator(0.001))
            context.setPositions(positions * unit.nanometer)
            with pytest.warns(UserWarning, match="atomic energy uncertainties"):
                _energy(context)

    def testSpinMultiplicityAlias(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-spin.pt")
            _export_model(
                path,
                HarmonicSpinModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
            )
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            potential.createSystem(pdb.topology, multiplicity=1)
            potential.createSystem(pdb.topology, spinMultiplicity=1)

    def testAtomTypes(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        custom_types = [100 + i for i in range(len(numbers))]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-types.pt")
            _export_harmonic(path, positions, custom_types)
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            with pytest.raises(ValueError, match="atomic type"):
                potential.createSystem(pdb.topology)
            system = potential.createSystem(pdb.topology, atomTypes=custom_types)
            context = mm.Context(system, mm.VerletIntegrator(0.001))
            context.setPositions(positions * unit.nanometer)
            assert np.isclose(_energy(context), 0.0, atol=1e-8)

    def testInvalidPbcLength(self, harmonic_toluene):
        pdb, _, _, model_path = harmonic_toluene
        potential = MLPotential("metatomic", modelPath=model_path, device="cpu")
        with pytest.raises(ValueError, match="length-3"):
            potential.createSystem(pdb.topology, pbc=(True, False))
