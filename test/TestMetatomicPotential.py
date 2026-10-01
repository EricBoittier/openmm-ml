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


class SpinAsEnergy(torch.nn.Module):
    """Energy equals the system's spin_multiplicity, in eV."""

    def requested_inputs(self) -> Dict[str, mta.ModelOutput]:
        return {
            "spin_multiplicity": mta.ModelOutput(unit="", sample_kind="system"),
        }

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            spin = system.get_data("spin_multiplicity").block().values
            # Touch positions so autograd can build forces. The energy is the spin.
            energy[i] += spin.reshape(()) + system.positions.sum() * 0
        return _energy_outputs(energy, outputs)


class WholeBoxEnergy(torch.nn.Module):
    """Energy is the sum of every coordinate the model is given.

    ``selected_atoms`` is ignored. A real model uses the atoms in its system,
    including any MM atoms left inside the cutoff.
    """

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            energy[i] += system.positions.sum()
        return _energy_outputs(energy, outputs)


class NeighborPairEnergy(torch.nn.Module):
    """Sum of squared neighbor vectors. ``selected_atoms`` does not drop pairs."""

    def __init__(self, cutoff):
        super().__init__()
        self._nl = mta.NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        return [self._nl]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device
        energy = torch.zeros((len(systems), 1), dtype=dtype, device=device)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl)
            disp = neighbors.values.reshape(-1, 3)
            energy[i] += disp.pow(2).sum()
        return _energy_outputs(energy, outputs)


class NeighborPairForce(torch.nn.Module):
    """Non-conservative pair forces. Output samples follow ``selected_atoms``,
    but the values are computed from every neighbor in the system.
    """

    def __init__(self, cutoff):
        super().__init__()
        self._nl = mta.NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        return [self._nl]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device
        energy = torch.zeros((len(systems), 1), dtype=dtype, device=device)
        all_forces = []
        for system_i, system in enumerate(systems):
            forces = torch.zeros((len(system), 3), dtype=dtype, device=device)
            neighbors = system.get_neighbor_list(self._nl)
            first = neighbors.samples.column("first_atom").to(torch.long)
            second = neighbors.samples.column("second_atom").to(torch.long)
            disp = neighbors.values.reshape(-1, 3)
            forces.index_add_(0, first, disp)
            forces.index_add_(0, second, -disp)
            if selected_atoms is not None:
                mask = selected_atoms.column("system") == system_i
                idx = selected_atoms.column("atom")[mask].to(torch.long)
                forces = forces[idx]
            all_forces.append(forces)

        result = _energy_outputs(energy, outputs)
        if "non_conservative_force" not in outputs:
            return result
        nc = torch.cat(all_forces).reshape(-1, 3, 1)
        if selected_atoms is None:
            rows = []
            for s, system in enumerate(systems):
                n_atoms = len(system)
                row = torch.zeros((n_atoms, 2), dtype=torch.int32, device=device)
                row[:, 0] = s
                row[:, 1] = torch.arange(n_atoms, device=device)
                rows.append(row)
            samples = Labels(["system", "atom"], torch.cat(rows))
        else:
            samples = selected_atoms
        result["non_conservative_force"] = TensorMap(
            keys=Labels("_", torch.tensor([[0]], device=device)),
            blocks=[
                TensorBlock(
                    values=nc,
                    samples=samples,
                    components=[
                        Labels(
                            ["xyz"],
                            torch.arange(3, device=device).reshape(-1, 1),
                        )
                    ],
                    properties=Labels(
                        ["non_conservative_force"],
                        torch.tensor([[0]], device=device),
                    ),
                )
            ],
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
        # Equilibrium only for the ML subset. The backend shows the model that
        # subset alone. atomic_types must cover the full Topology (including solvent).
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
            # The ML subset is an isolated molecule, so this force is not periodic.
            assert not python_forces[0].usesPeriodicBoundaryConditions()
            # Empty => applies to all atoms; OpenMM returns a tuple.
            assert len(python_forces[0].getParticles()) == 0

    def testSelectedAtoms(self, platform_int):
        # Same mixed system as the other backends. The harmonic reference covers
        # only the ML atoms: that is the system the model is given.
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
            _export_harmonic(path, positions[ml_atoms], numbers)
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
            {"charge": 0, "spin_multiplicity": 1},
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
    def testSpinMultiplicity(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        factor = float(mta.unit_conversion_factor("eV", "kJ/mol"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spin.pt")
            _export_model(path, SpinAsEnergy(), numbers)
            potential = _potential(path)
            context = mm.Context(
                potential.createSystem(pdb.topology),
                mm.VerletIntegrator(0.001),
            )
            context.setPositions(positions * unit.nanometer)
            assert np.isclose(_energy(context), factor, rtol=1e-5, atol=1e-8)
            context = mm.Context(
                potential.createSystem(pdb.topology, spin_multiplicity=3),
                mm.VerletIntegrator(0.001),
            )
            context.setPositions(positions * unit.nanometer)
            assert np.isclose(_energy(context), 3 * factor, rtol=1e-5, atol=1e-8)

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


def _bare_system(n_atoms, box=None):
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for _ in range(n_atoms):
        topology.addAtom("C", app.element.carbon, residue)
    system = mm.System()
    for _ in range(n_atoms):
        system.addParticle(1.0)
    if box is not None:
        topology.setPeriodicBoxVectors(box)
        system.setDefaultPeriodicBoxVectors(*box)
    return topology, system


def _direct_model(path, positions, types, selected, box, non_conservative=False):
    """Run ``path`` the way the backend used to.

    Every atom is in the metatomic system. ``selected`` only filters outputs,
    so atoms outside that subset still contribute as environment.
    """
    from openmmml.models.metatomicpotential import _NL_SKIN

    model = mta.load_atomistic_model(path)
    dtype = getattr(torch, model.capabilities().dtype)
    pos = torch.tensor(positions, dtype=dtype, requires_grad=not non_conservative)
    types_t = torch.tensor(types, dtype=torch.int32)
    if box is None:
        cell = torch.zeros((3, 3), dtype=dtype)
        pbc = torch.zeros(3, dtype=torch.bool)
    else:
        cell = torch.tensor(box, dtype=dtype)
        pbc = torch.ones(3, dtype=torch.bool)
    selected_atoms = None
    if selected is not None:
        selected_atoms = Labels(
            ["system", "atom"],
            torch.tensor([[0, i] for i in selected], dtype=torch.int32),
        )
    metatomic_system = mta.System(types_t, pos, cell, pbc)
    requested = model.requested_neighbor_lists()
    # Keep the calculators alive: copy=False stores a pointer into them.
    neighbor_lists = []
    if requested:
        import vesin.metatomic

        for options in requested:
            neighbors = vesin.metatomic.NeighborList(
                options=options,
                length_unit="nm",
                check_consistency=True,
                skin=_NL_SKIN,
            )
            neighbors.add_neighbor_list(systems=[metatomic_system], copy=False)
            neighbor_lists.append(neighbors)
    outputs = {"energy": mta.ModelOutput(unit="kJ/mol", sample_kind="system")}
    if non_conservative:
        outputs["non_conservative_force"] = mta.ModelOutput(
            unit="kJ/mol/nm", sample_kind="atom"
        )
    options = mta.ModelEvaluationOptions(
        length_unit="nm",
        outputs=outputs,
        selected_atoms=selected_atoms,
    )
    result = model([metatomic_system], options, check_consistency=True)
    energy = float(result["energy"].block().values.sum().detach())
    if non_conservative:
        block = result["non_conservative_force"].block()
        nc = block.values.detach().reshape(-1, 3)
        nc = nc - nc.mean(dim=0, keepdim=True)
        atoms = block.samples.column("atom").detach().cpu().numpy()
        forces = np.zeros((len(positions), 3), dtype=np.float64)
        forces[atoms] = nc.cpu().numpy()
        return energy, forces
    result["energy"].block().values.sum().backward()
    forces = (-metatomic_system.positions.grad).detach().cpu().numpy()
    return energy, forces


def _openmm_subset(path, positions, ml_atoms, platform_int, box=None, **potential_kwargs):
    from openmmml.models.metatomicpotential import MetatomicPotentialImpl

    topology, system = _bare_system(len(positions), box)
    impl = MetatomicPotentialImpl(
        "metatomic", path, "cpu", None, True, **potential_kwargs
    )
    impl.addForces(topology, system, ml_atoms, 0)
    platform = mm.Platform.getPlatform(platform_int)
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    context.setPositions(positions * unit.nanometer)
    python_forces = [f for f in system.getForces() if isinstance(f, mm.PythonForce)]
    uses_pbc = python_forces[0].usesPeriodicBoundaryConditions()
    return _energy(context), _forces(context), uses_pbc


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMetatomicMixedRegion:
    def testWholeBoxEnergyIncludesMmAtoms(self, platform_int):
        # Two ML atoms and one MM atom. The model sums every coordinate it is
        # given and does not look at selected_atoms.
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.2, 0.0]],
            dtype=np.float64,
        )
        ml_atoms = [0, 1]
        mm_atom = 2
        types = [6, 6, 6]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "whole-box.pt")
            _export_model(path, WholeBoxEnergy(), types)
            old_energy, old_forces = _direct_model(
                path, positions, types, ml_atoms, box=None
            )
            isolated_energy, isolated_forces = _direct_model(
                path,
                positions[ml_atoms],
                [types[i] for i in ml_atoms],
                selected=None,
                box=None,
            )
            new_energy, new_forces, _ = _openmm_subset(
                path, positions, ml_atoms, platform_int
            )

        assert not np.isclose(old_energy, isolated_energy)
        assert np.linalg.norm(old_forces[mm_atom]) > 1.0
        np.testing.assert_allclose(new_energy, isolated_energy, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(new_forces[ml_atoms], isolated_forces, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(new_forces[mm_atom], 0.0, atol=1e-8)

        moved = positions.copy()
        moved[mm_atom, 0] += 0.3
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "whole-box.pt")
            _export_model(path, WholeBoxEnergy(), types)
            moved_old, _ = _direct_model(path, moved, types, ml_atoms, box=None)
            moved_new, moved_forces, _ = _openmm_subset(
                path, moved, ml_atoms, platform_int
            )
        assert not np.isclose(moved_old, old_energy)
        np.testing.assert_allclose(moved_new, new_energy, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(moved_forces[mm_atom], 0.0, atol=1e-8)

    def testNeighborInsideCutoff(self, platform_int):
        # MM atom sits inside the model cutoff of both ML atoms.
        cutoff = 0.5
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.35, 0.0, 0.0]],
            dtype=np.float64,
        )
        ml_atoms = [0, 1]
        types = [6, 6, 6]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.pt")
            _export_model(
                path,
                NeighborPairEnergy(cutoff),
                types,
                interaction_range=cutoff,
            )
            old_energy, old_forces = _direct_model(
                path, positions, types, ml_atoms, box=None
            )
            isolated_energy, isolated_forces = _direct_model(
                path,
                positions[ml_atoms],
                [types[i] for i in ml_atoms],
                selected=None,
                box=None,
            )
            new_energy, new_forces, _ = _openmm_subset(
                path, positions, ml_atoms, platform_int
            )

        assert old_energy > isolated_energy
        assert np.linalg.norm(old_forces[2]) > 1.0
        np.testing.assert_allclose(new_energy, isolated_energy, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            new_forces[ml_atoms], isolated_forces, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(new_forces[2], 0.0, atol=1e-8)

    def testPeriodicImageOfMmAtom(self, platform_int):
        # The MM atom is outside the cutoff in the box, and inside it across
        # the periodic boundary. The old path counts that image; the subset
        # path is not periodic and does not contain the MM atom.
        cutoff = 0.3
        positions = np.array([[0.05, 0.5, 0.5], [0.90, 0.5, 0.5]], dtype=np.float64)
        ml_atoms = [0]
        types = [6, 6]
        box_vectors = [mm.Vec3(1, 0, 0), mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 1)]
        box = [v * unit.nanometer for v in box_vectors]
        cell = np.eye(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "periodic.pt")
            _export_model(
                path,
                NeighborPairEnergy(cutoff),
                types,
                interaction_range=cutoff,
            )
            old_energy, old_forces = _direct_model(
                path, positions, types, ml_atoms, box=cell
            )
            isolated_energy, isolated_forces = _direct_model(
                path,
                positions[ml_atoms],
                [types[i] for i in ml_atoms],
                selected=None,
                box=None,
            )
            new_energy, new_forces, uses_pbc = _openmm_subset(
                path, positions, ml_atoms, platform_int, box=box
            )

        assert old_energy > 0.0
        assert np.linalg.norm(old_forces[1]) > 1.0
        assert isolated_energy == 0.0
        np.testing.assert_allclose(new_energy, 0.0, atol=1e-8)
        np.testing.assert_allclose(new_forces, 0.0, atol=1e-8)
        np.testing.assert_allclose(isolated_forces, 0.0, atol=1e-8)
        assert not uses_pbc

    def testNonConservativeForcesStayOnTheRegion(self, platform_int):
        cutoff = 0.5
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.35, 0.0, 0.0]],
            dtype=np.float64,
        )
        ml_atoms = [0, 1]
        types = [6, 6, 6]
        outputs = {
            "energy": mta.ModelOutput(unit="eV", sample_kind="system"),
            "non_conservative_force": mta.ModelOutput(
                unit="eV/nm", sample_kind="atom"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nc.pt")
            _export_model(
                path,
                NeighborPairForce(cutoff),
                types,
                interaction_range=cutoff,
                outputs=outputs,
            )
            old_energy, old_forces = _direct_model(
                path, positions, types, ml_atoms, box=None, non_conservative=True
            )
            isolated_energy, isolated_forces = _direct_model(
                path,
                positions[ml_atoms],
                [types[i] for i in ml_atoms],
                selected=None,
                box=None,
                non_conservative=True,
            )
            new_energy, new_forces, _ = _openmm_subset(
                path, positions, ml_atoms, platform_int, nonConservative=True
            )

        assert not np.allclose(old_forces[ml_atoms], isolated_forces, atol=1e-6)
        np.testing.assert_allclose(new_energy, isolated_energy, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            new_forces[ml_atoms], isolated_forces, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(new_forces[2], 0.0, atol=1e-8)
        assert np.linalg.norm(new_forces[ml_atoms]) > 1.0
