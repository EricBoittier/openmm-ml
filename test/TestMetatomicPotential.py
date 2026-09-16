import os
import tempfile
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


def _energy_block(energy):
    block = TensorBlock(
        values=energy,
        samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    return {
        "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
    }


class HarmonicModel(torch.nn.Module):
    def __init__(self, force_constant: float, equilibrium_positions: torch.Tensor):
        super().__init__()
        self.force_constant = force_constant
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            energy[i] += torch.sum(
                self.force_constant
                * (system.positions - self.equilibrium_positions) ** 2
            )
        return _energy_block(energy)


class HarmonicNLModel(HarmonicModel):
    def __init__(self, force_constant, equilibrium_positions, cutoff=4.0):
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
            energy[i] += torch.sum(
                self.force_constant
                * (system.positions - self.equilibrium_positions) ** 2
            )
        return _energy_block(energy)


class HarmonicChargeModel(HarmonicModel):
    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {"charge": mta.ModelOutput(unit="e", sample_kind="system")}


class PerAtomChargeModel(HarmonicModel):
    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {"charge": mta.ModelOutput(unit="e", sample_kind="atom")}


def _export_model(path, model, atomic_types, interaction_range=0.0):
    capabilities = mta.ModelCapabilities(
        outputs={"energy": mta.ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=sorted(set(atomic_types)),
        interaction_range=interaction_range,
        length_unit="Angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    wrapper = mta.AtomisticModel(model.eval(), mta.ModelMetadata(), capabilities)
    wrapper.save(path)
    return path


def _export_harmonic(path, positions_angstrom, atomic_types, force_constant=1.0):
    model = HarmonicModel(
        force_constant,
        torch.tensor(positions_angstrom, dtype=torch.float64),
    )
    return _export_model(path, model, atomic_types)


def _direct_energy_forces(model_path, numbers, positions_angstrom):
    model = mta.load_atomistic_model(model_path)
    dtype = torch.float64
    types = torch.tensor(numbers, dtype=torch.int32)
    pos = torch.tensor(positions_angstrom, dtype=dtype, requires_grad=True)
    cell = torch.zeros((3, 3), dtype=dtype)
    pbc = torch.tensor([False, False, False])
    system = mta.System(types, pos, cell, pbc)
    options = mta.ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={"energy": mta.ModelOutput(unit="eV", sample_kind="system")},
    )
    energy = model([system], options, False)["energy"].block().values.sum()
    energy.backward()
    scale = float(mta.unit_conversion_factor("eV", "kJ/mol"))
    return (
        float(energy.detach()) * scale,
        (-pos.grad * scale * 10.0).detach().numpy(),
    )


@pytest.fixture(scope="module")
def harmonic_toluene():
    pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "harmonic.pt")
        _export_harmonic(path, positions, numbers)
        yield pdb, numbers, positions, path


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMetatomicPotential:
    def testCreatePureMLSystem(self, platform_int, harmonic_toluene):
        pdb, numbers, positions, model_path = harmonic_toluene
        displaced = positions + 0.1
        potential = MLPotential("metatomic", modelPath=model_path, device="cpu")
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(displaced * unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        energy_ml = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces_ml = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        energy_ref, forces_ref = _direct_energy_forces(model_path, numbers, displaced)
        assert np.isclose(energy_ref, energy_ml, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(forces_ref, forces_ml, rtol=1e-5, atol=1e-6)

    def testCreateMixedSystem(self, platform_int, harmonic_toluene):
        pdb, numbers, positions, model_path = harmonic_toluene
        prmtop = app.AmberPrmtopFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7")
        )
        ml_atoms = list(range(15))
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("metatomic", modelPath=model_path, device="cpu")
        mixed_system = potential.createMixedSystem(
            prmtop.topology, mm_system, ml_atoms, interpolate=False
        )
        interp_system = potential.createMixedSystem(
            prmtop.topology, mm_system, ml_atoms, interpolate=True
        )
        platform = mm.Platform.getPlatform(platform_int)
        mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform)
        mixed_context = mm.Context(mixed_system, mm.VerletIntegrator(0.001), platform)
        interp_context = mm.Context(interp_system, mm.VerletIntegrator(0.001), platform)
        mm_context.setPositions(inpcrd.positions)
        mixed_context.setPositions(inpcrd.positions)
        interp_context.setPositions(inpcrd.positions)
        mm_energy = (
            mm_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        mixed_energy = (
            mixed_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        interp_energy1 = (
            interp_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        interp_context.setParameter("lambda_interpolate", 0)
        interp_energy2 = (
            interp_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        assert np.isclose(mixed_energy, interp_energy1, rtol=1e-5)
        assert np.isclose(mm_energy, interp_energy2, rtol=1e-5)
        python_forces = [
            f for f in mixed_system.getForces() if isinstance(f, mm.PythonForce)
        ]
        assert python_forces
        assert python_forces[0].usesPeriodicBoundaryConditions()

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
                interaction_range=4.0,
            )
            potential = MLPotential("metatomic", modelPath=path, device="cpu")
            system = potential.createSystem(pdb.topology)
            platform = mm.Platform.getPlatform(platform_int)
            context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
            context.setPositions((positions + 0.1) * unit.angstrom)
            energy = (
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
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
                pdb.topology, charge=0, info={"spin_multiplicity": 1}
            )
            platform = mm.Platform.getPlatform(platform_int)
            context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
            context.setPositions(positions * unit.angstrom)
            energy = (
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
            assert np.isclose(energy, 0.0, atol=1e-8)

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

    def testAgreesWithASE(self, platform_int, harmonic_toluene):
        pytest.importorskip("ase", reason="ASE is not installed")
        pytest.importorskip("metatomic_ase", reason="metatomic-ase is not installed")
        from metatomic_ase import MetatomicCalculator

        pdb, _, positions, model_path = harmonic_toluene
        displaced = positions + 0.05
        platform = mm.Platform.getPlatform(platform_int)
        native = MLPotential("metatomic", modelPath=model_path, device="cpu")
        native_system = native.createSystem(pdb.topology)
        native_context = mm.Context(
            native_system, mm.VerletIntegrator(0.001), platform
        )
        native_context.setPositions(displaced * unit.angstrom)
        native_energy = (
            native_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )

        calculator = MetatomicCalculator(
            model_path, device="cpu", do_gradients_with_energy=True
        )
        ase_system = MLPotential("ase").createSystem(
            pdb.topology, calculator=calculator
        )
        ase_context = mm.Context(ase_system, mm.VerletIntegrator(0.001), platform)
        ase_context.setPositions(displaced * unit.angstrom)
        ase_energy = (
            ase_context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        assert np.isclose(native_energy, ase_energy, rtol=1e-5, atol=1e-6)
