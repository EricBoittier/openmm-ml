import os
import re
import tempfile
from typing import Dict, List, Optional

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import torch
import metatomic.torch as mta
from metatensor.torch import Labels, TensorBlock, TensorMap

from openmmml import MLPotential

platform_ints = range(mm.Platform.getNumPlatforms())
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Same parameters as the metatomic ASE engine's lj-test comparison.
_LJ_CUTOFF = 5.0  # Angstrom
_LJ_SIGMA = 1.5808
_LJ_EPSILON = 0.1729


def _energy_map(energy: torch.Tensor) -> TensorMap:
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
            result[key] = _energy_map(energy)
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


class RequestedInputModel(HarmonicModel):
    def __init__(self, force_constant, equilibrium_positions, requested):
        super().__init__(force_constant, equilibrium_positions)
        self._requested = requested

    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return self._requested


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


def _potential(path, **kwargs):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("checkConsistency", True)
    return MLPotential("metatomic", modelPath=path, **kwargs)


def _energy(context):
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )


def _forces(context):
    return context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
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


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMetatomicPotential:
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
            potential = _potential(path)
            # Finite interaction_range => getMLLongRange() is False; no need to pass
            # mlLongRange. An explicit value remains allowed as an override.
            mixed_system = potential.createMixedSystem(
                prmtop.topology,
                mm_system,
                ml_atoms,
                interpolate=False,
            )
            interp_system = potential.createMixedSystem(
                prmtop.topology,
                mm_system,
                ml_atoms,
                interpolate=True,
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

    def testSelectedAtoms(self, platform_int):
        # Same mixed system as the other backends. The harmonic energy depends
        # only on the ML subset passed through as selected_atoms.
        prmtop = app.AmberPrmtopFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7")
        )
        ml_atoms = list(range(15))
        positions = np.asarray(
            inpcrd.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        numbers = [
            atom.element.atomic_number for atom in prmtop.topology.atoms()
        ]
        delta = 0.01
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-selected.pt")
            _export_harmonic(path, positions, numbers)
            mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
            mixed = _potential(path).createMixedSystem(
                prmtop.topology, mm_system, ml_atoms, forceGroup=1
            )
            platform = mm.Platform.getPlatform(platform_int)
            context = mm.Context(mixed, mm.VerletIntegrator(0.001), platform)
            context.setPositions((positions + delta) * unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True, groups={1})
            factor = float(mta.unit_conversion_factor("eV", "kJ/mol"))
            energy = len(ml_atoms) * 3 * delta**2 * factor
            forces = np.zeros_like(positions)
            forces[ml_atoms] = -2 * delta * factor
            assert np.isclose(
                energy,
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                rtol=1e-5,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                forces,
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoules_per_mole / unit.nanometer
                ),
                rtol=1e-5,
                atol=1e-6,
            )

    def testExtraInputs(self, platform_int, harmonic_toluene):
        pdb, _, positions, _ = harmonic_toluene
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        equilibrium = torch.tensor(positions, dtype=torch.float64)
        # charge and spin_multiplicity are the extra inputs this backend accepts.
        requested = {
            "charge": mta.ModelOutput(unit="e", sample_kind="system"),
            "spin_multiplicity": mta.ModelOutput(unit="", sample_kind="system"),
        }
        cases = [
            {},
            {"charge": 0, "multiplicity": 1},
            {"charge": 0, "spinMultiplicity": 1},
        ]
        platform = mm.Platform.getPlatform(platform_int)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "extra.pt")
            _export_model(
                path, RequestedInputModel(1.0, equilibrium, requested), numbers
            )
            potential = _potential(path)
            for kwargs in cases:
                system = potential.createSystem(pdb.topology, **kwargs)
                context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
                context.setPositions(positions * unit.nanometer)
                assert np.isclose(_energy(context), 0.0, atol=1e-8)

            path = os.path.join(tmp, "per-atom-charge.pt")
            _export_model(
                path,
                RequestedInputModel(
                    1.0,
                    equilibrium,
                    {"charge": mta.ModelOutput(unit="e", sample_kind="atom")},
                ),
                numbers,
            )
            message = (
                "this model requests extra input 'charge' (sample_kind='atom'), "
                "which is not implemented by MLPotential('metatomic')"
            )
            with pytest.raises(ValueError, match=re.escape(message)):
                _potential(path).createSystem(pdb.topology)

    def testPartialPbc(self, platform_int, harmonic_toluene):
        pdb, _, positions, model_path = harmonic_toluene
        box = [mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2)]
        box = [v * unit.nanometer for v in box]
        system = _potential(model_path).createSystem(
            pdb.topology, pbc=(True, True, False)
        )
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

    def testLennardJones(self, platform_int):
        ase = pytest.importorskip("ase")
        pytest.importorskip("vesin")
        lj = pytest.importorskip("metatomic_lj_test")
        import ase.calculators.lj
        import ase.units

        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        model = lj.lennard_jones_model(
            atomic_type=numbers[0],
            cutoff=_LJ_CUTOFF,
            sigma=_LJ_SIGMA,
            epsilon=_LJ_EPSILON,
            length_unit="Angstrom",
            energy_unit="eV",
            with_extension=False,
        )
        model._capabilities.atomic_types = sorted(set(numbers))
        platform = mm.Platform.getPlatform(platform_int)
        # ASE reports eV and eV/Angstrom. OpenMM reports kJ/mol and kJ/mol/nm.
        to_kj_mol = ase.units.mol / ase.units.kJ
        angstrom_per_nm = ase.units.nm
        positions = pdb.getPositions(asNumpy=True)
        ref = ase.Atoms(
            numbers=numbers,
            positions=positions.value_in_unit(unit.angstrom),
            pbc=False,
        )
        ref.calc = ase.calculators.lj.LennardJones(
            sigma=_LJ_SIGMA,
            epsilon=_LJ_EPSILON,
            rc=_LJ_CUTOFF,
            ro=_LJ_CUTOFF,
            smooth=False,
        )
        energy_ref = ref.get_potential_energy() * to_kj_mol
        forces_ref = ref.get_forces() * to_kj_mol * angstrom_per_nm

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lj.pt")
            model.save(path)

            def evaluate(**potential_kwargs):
                system = _potential(path, **potential_kwargs).createSystem(
                    pdb.topology, removeCMMotion=False
                )
                context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
                context.setPositions(positions)
                return context

            context = evaluate(uncertaintyThreshold=None)
            assert np.isclose(energy_ref, _energy(context), rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(
                forces_ref, _forces(context), rtol=1e-5, atol=1e-6
            )

            context = evaluate(
                variants={"energy": "doubled"}, uncertaintyThreshold=None
            )
            assert np.isclose(
                2.0 * energy_ref, _energy(context), rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                2.0 * forces_ref, _forces(context), rtol=1e-5, atol=1e-6
            )

            message = (
                "Some of the atomic energy uncertainties are larger than the "
                "threshold of 10.0 kJ/mol."
            )
            with pytest.warns(UserWarning, match=re.escape(message)):
                _energy(evaluate())


class TestMetatomicPotentialOptions:
    def testNonConservativeRequiresOutput(self, harmonic_toluene):
        pdb, _, _, model_path = harmonic_toluene
        potential = _potential(model_path, nonConservative=True)
        message = "output 'non_conservative_force' not found in outputs"
        with pytest.raises(ValueError, match=re.escape(message)):
            potential.createSystem(pdb.topology)

    def testAtomTypes(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        custom_types = [100 + i for i in range(len(numbers))]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "harmonic-types.pt")
            _export_harmonic(path, positions, custom_types)
            potential = _potential(path)
            message = "this model does not support atomic type 6"
            with pytest.raises(ValueError, match=re.escape(message)):
                potential.createSystem(pdb.topology)
            system = potential.createSystem(pdb.topology, atomTypes=custom_types)
            context = mm.Context(system, mm.VerletIntegrator(0.001))
            context.setPositions(positions * unit.nanometer)
            assert np.isclose(_energy(context), 0.0, atol=1e-8)

    def testInvalidPbcLength(self, harmonic_toluene):
        pdb, _, _, model_path = harmonic_toluene
        potential = _potential(model_path)
        message = "pbc must be a length-3 sequence of booleans"
        with pytest.raises(ValueError, match=re.escape(message)):
            potential.createSystem(pdb.topology, pbc=(True, False))

    def testGetMLLongRangeFromInteractionRange(self, harmonic_toluene):
        from openmmml.models.metatomicpotential import MetatomicPotentialImpl

        _, numbers, positions, _ = harmonic_toluene
        with tempfile.TemporaryDirectory() as tmp:
            short_path = os.path.join(tmp, "short.pt")
            long_path = os.path.join(tmp, "long.pt")
            _export_model(
                short_path,
                HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
                interaction_range=0.5,
            )
            _export_model(
                long_path,
                HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                numbers,
                interaction_range=float("inf"),
            )
            short = MetatomicPotentialImpl(
                "metatomic", short_path, "cpu", None, True
            )
            long = MetatomicPotentialImpl(
                "metatomic", long_path, "cpu", None, True
            )
            assert short.getMLLongRange() is False
            assert long.getMLLongRange() is True

    def testMLLongRangeOverride(self, harmonic_toluene):
        from openmmml.models.metatomicpotential import MetatomicPotentialImpl

        _, numbers, positions, _ = harmonic_toluene
        prmtop = app.AmberPrmtopFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7")
        )
        ml_atoms = list(range(15))
        all_numbers = [
            atom.element.atomic_number for atom in prmtop.topology.atoms()
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # Model reports long-range; override to short-range for embedding.
            path = os.path.join(tmp, "long-override.pt")
            _export_model(
                path,
                HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
                all_numbers,
                interaction_range=float("inf"),
            )
            mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
            potential = _potential(path)
            impl = MetatomicPotentialImpl(
                "metatomic", path, "cpu", None, True
            )
            assert impl.getMLLongRange() is True
            mixed = potential.createMixedSystem(
                prmtop.topology,
                mm_system,
                ml_atoms,
                mlLongRange=False,
            )
            assert mixed is not None
