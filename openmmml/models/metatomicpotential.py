"""
metatomicpotential.py: Implements potentials from metatomic models.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Eric D. Boittier

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import warnings
from typing import Iterable, Optional

import numpy as np
import openmm
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

_DTYPES = {"float32", "float64"}
_INPUT_DEFAULTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}
_VALID_NC = (True, False, "forces")
# 0.1 eV in kJ/mol; OpenMM native energy unit.
_DEFAULT_UNCERTAINTY_THRESHOLD = 9.648533212
# Neighbor-list skin in nm (2 Å). Set to -1 once vesin supports auto skin.
_NL_SKIN = 0.2


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MetatomicPotentialImpl objects."""

    def createImpl(
        self,
        name: str,
        modelPath: str,
        device=None,
        extensionsDirectory=None,
        checkConsistency: bool = False,
        nonConservative=False,
        variants=None,
        uncertaintyThreshold=_DEFAULT_UNCERTAINTY_THRESHOLD,
        **args,
    ) -> MLPotentialImpl:
        return MetatomicPotentialImpl(
            name,
            modelPath,
            device,
            extensionsDirectory,
            checkConsistency,
            nonConservative,
            variants,
            uncertaintyThreshold,
        )


class MetatomicPotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl evaluates a metatomic model.

    Load a TorchScript model produced by :func:`metatomic.torch.load_atomistic_model`
    (typically a ``.pt`` file) and install a single :class:`openmm.PythonForce`.
    Neighbor lists use ``vesin.metatomic`` (CPU and CUDA). Conservative forces are
    ``-dE/dx`` via autograd. The energy always depends on the cell when the
    system is periodic; OpenMM MonteCarlo barostats can run NPT by
    finite-differencing that energy. :class:`openmm.PythonForce` returns only
    energy and forces, so an explicit virial cannot be passed to the integrator.

    >>> potential = MLPotential(
    ...     "metatomic",
    ...     modelPath="model.pt",
    ...     device="cuda",
    ...     extensionsDirectory="./extensions",
    ...     checkConsistency=False,
    ...     nonConservative=False,
    ...     variants={"energy": "pbe"},
    ...     uncertaintyThreshold=9.65,
    ... )
    >>> system = potential.createSystem(topology)

    Optional ``createSystem()`` / ``createMixedSystem()`` arguments:

    - ``charge``: total charge (default 0), used if the model requests it
    - ``multiplicity``: spin multiplicity (default 1); ``spinMultiplicity`` is
      accepted as an alias
    - ``atomTypes``: integer type for each Topology atom; defaults to element
      atomic numbers when omitted
    - ``pbc``: length-3 sequence of booleans; default is all-on or all-off from
      the topology and System

    For periodic mixed ML/MM systems, :meth:`getMLLongRange` is inferred from the
    model's ``interaction_range`` (``True`` when infinite). Pass ``mlLongRange``
    to :meth:`~openmmml.MLPotential.createMixedSystem` to override that choice.
    """

    def __init__(
        self,
        name,
        modelPath,
        device,
        extensionsDirectory,
        checkConsistency,
        nonConservative=False,
        variants=None,
        uncertaintyThreshold=_DEFAULT_UNCERTAINTY_THRESHOLD,
    ):
        if nonConservative not in _VALID_NC:
            raise ValueError(
                f"nonConservative must be one of {list(_VALID_NC)}, "
                f"got {nonConservative!r}"
            )
        self.name = name
        self.modelPath = modelPath
        self.device = device
        self.extensionsDirectory = extensionsDirectory
        self.checkConsistency = checkConsistency
        self.nonConservative = nonConservative
        self.variants = variants
        self.uncertaintyThreshold = uncertaintyThreshold
        self._ml_long_range = None

    def getMLLongRange(self) -> bool:
        """Return whether the model includes all-image ML-ML interactions.

        Uses ``capabilities.interaction_range``: infinite means long-range
        (``True``), any finite value means short-range (``False``). Callers can
        still pass ``mlLongRange`` to ``createMixedSystem()`` to override.
        """
        if self._ml_long_range is None:
            try:
                from metatomic.torch import load_atomistic_model
            except ImportError as e:
                raise ImportError(
                    "Failed to import metatomic. Install it with "
                    "'pip install metatomic-torch'."
                ) from e
            model = load_atomistic_model(
                self.modelPath, extensions_directory=self.extensionsDirectory
            )
            self._ml_long_range = math.isinf(model.capabilities().interaction_range)
        return self._ml_long_range

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        **args,
    ):
        try:
            import torch
            from metatensor.torch import Labels
            from metatomic.torch import (
                ModelEvaluationOptions,
                ModelOutput,
                load_atomistic_model,
                pick_device,
                pick_output,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import metatomic. Install it with "
                "'pip install metatomic-torch'."
            ) from e

        topology_atoms = list(topology.atoms())
        types = _resolve_atom_types(topology_atoms, args.get("atomTypes"))

        model = load_atomistic_model(
            self.modelPath, extensions_directory=self.extensionsDirectory
        )
        capabilities = model.capabilities()
        allowed = set(capabilities.atomic_types)
        for atom_type in types:
            if atom_type not in allowed:
                raise ValueError(
                    f"this model does not support atomic type {atom_type}"
                )
        desired = self.device
        if desired is not None and not isinstance(desired, str):
            desired = str(desired)
        device = torch.device(pick_device(capabilities.supported_devices, desired))
        dtype_name = capabilities.dtype
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported model dtype '{dtype_name}'.")
        dtype = getattr(torch, dtype_name)
        model = model.to(device=device)
        types = torch.tensor(types, dtype=torch.int32, device=device)

        energy_key, nc_forces_key, uq_key = _resolve_output_keys(
            capabilities.outputs,
            self.variants,
            self.nonConservative,
            self.uncertaintyThreshold,
            pick_output,
        )

        extras = _extra_inputs(
            model.requested_inputs(use_new_names=True), args, dtype, device
        )

        neighbor_lists = []
        requested_nl = model.requested_neighbor_lists()
        if requested_nl:
            try:
                import vesin.metatomic
            except ImportError as e:
                raise ImportError(
                    "Failed to import vesin. Install it with 'pip install vesin'."
                ) from e
            neighbor_lists = [
                vesin.metatomic.NeighborList(
                    options=options,
                    length_unit="nm",
                    check_consistency=self.checkConsistency,
                    skin=_NL_SKIN,
                )
                for options in requested_nl
            ]

        selected_atoms = None
        selected_atom_indices = None
        if atoms is not None:
            selected_atom_indices = list(atoms)
            selected_atoms = Labels(
                ["system", "atom"],
                torch.tensor(
                    [[0, i] for i in selected_atom_indices], dtype=torch.int32
                ),
            )

        pbc = _resolve_pbc(args, topology, system, device)
        outputs = {
            energy_key: ModelOutput(unit="kJ/mol", sample_kind="system"),
        }
        if nc_forces_key is not None:
            outputs[nc_forces_key] = ModelOutput(
                unit="kJ/mol/nm", sample_kind="atom"
            )
        if uq_key is not None:
            outputs[uq_key] = ModelOutput(unit="kJ/mol", sample_kind="atom")
        options = ModelEvaluationOptions(
            length_unit="nm",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )

        compute = _ComputeMetatomic(
            model=model,
            model_path=self.modelPath,
            extensions_directory=self.extensionsDirectory,
            types=types,
            extras=extras,
            neighbor_lists=neighbor_lists,
            nl_options=requested_nl,
            options=options,
            energy_key=energy_key,
            nc_forces_key=nc_forces_key,
            uq_key=uq_key,
            uncertainty_threshold=self.uncertaintyThreshold,
            check_consistency=self.checkConsistency,
            pbc=pbc,
            dtype=dtype,
            selected_atom_indices=selected_atom_indices,
        )
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(bool(pbc.any()))
        system.addForce(force)


def _resolve_atom_types(topology_atoms, atom_types):
    n_atoms = len(topology_atoms)
    if atom_types is not None:
        atom_types = list(atom_types)
        if len(atom_types) != n_atoms:
            raise ValueError(
                f"atomTypes must have length {n_atoms} (one entry per Topology "
                f"atom), got {len(atom_types)}"
            )
        return [int(t) for t in atom_types]
    if any(atom.element is None for atom in topology_atoms):
        raise ValueError(
            "All atoms in the Topology must have elements defined, or pass "
            "atomTypes with an integer type for every atom."
        )
    return [atom.element.atomic_number for atom in topology_atoms]


def _resolve_output_keys(
    outputs, variants, non_conservative, uncertainty_threshold, pick_output
):
    variants = dict(variants or {})
    default_variant = variants.get("energy")
    resolved = {
        key: variants.get(key, default_variant)
        for key in [
            "energy",
            "energy_uncertainty",
            "non_conservative_force",
        ]
    }
    if "non_conservative_forces" in variants:
        warnings.warn(
            "variant name 'non_conservative_forces' is deprecated, please use "
            "'non_conservative_force' instead",
            stacklevel=3,
        )
        if "non_conservative_force" in variants:
            raise ValueError(
                "you can not specify both 'non_conservative_force' and "
                "'non_conservative_forces' in `variants`"
            )
        resolved["non_conservative_force"] = variants["non_conservative_forces"]

    energy_key = pick_output("energy", outputs, resolved["energy"])

    has_energy_uq = any("energy_uncertainty" in key for key in outputs.keys())
    uq_key = (
        pick_output(
            "energy_uncertainty", outputs, resolved["energy_uncertainty"]
        )
        if has_energy_uq and uncertainty_threshold is not None
        else None
    )

    nc_forces = non_conservative in (True, "forces")
    nc_forces_key = (
        pick_output(
            "non_conservative_force",
            outputs,
            resolved["non_conservative_force"],
        )
        if nc_forces
        else None
    )
    return energy_key, nc_forces_key, uq_key


def _resolve_pbc(args, topology, system, device):
    import torch

    periodic = (
        topology.getPeriodicBoxVectors() is not None
        or system.usesPeriodicBoundaryConditions()
    )
    user_pbc = args.get("pbc")
    if user_pbc is None:
        flags = [periodic, periodic, periodic]
    else:
        flags = [bool(x) for x in user_pbc]
        if len(flags) != 3:
            raise ValueError("pbc must be a length-3 sequence of booleans")
    return torch.tensor(flags, dtype=torch.bool, device=device)


def _unsupported_input(name, sample_kind=None):
    kind = f" (sample_kind={sample_kind!r})" if sample_kind is not None else ""
    return ValueError(
        f"this model requests extra input '{name}'{kind}, which is not "
        "implemented by MLPotential('metatomic')"
    )


def _spin_value(args, default):
    for name in ("multiplicity", "spinMultiplicity"):
        if name in args:
            return args[name]
    return default


def _extra_inputs(requested, args, dtype, device):
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    extras = {}
    for name, option in requested.items():
        sample_kind = option.sample_kind
        if name not in _INPUT_DEFAULTS or sample_kind != "system":
            raise _unsupported_input(name, sample_kind)
        default, input_unit = _INPUT_DEFAULTS[name]
        value = (
            _spin_value(args, default)
            if name == "spin_multiplicity"
            else args.get(name, default)
        )
        block = TensorBlock(
            values=torch.tensor([[float(value)]], dtype=dtype),
            samples=Labels(["system"], torch.zeros((1, 1), dtype=torch.int32)),
            components=[],
            properties=Labels([name], torch.tensor([[0]])),
        )
        tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])
        tensor.set_info("unit", input_unit)
        extras[name] = tensor.to(dtype=dtype, device=device)
    return extras


class _ComputeMetatomic:
    def __init__(
        self,
        model,
        model_path,
        extensions_directory,
        types,
        extras,
        neighbor_lists,
        nl_options,
        options,
        energy_key,
        nc_forces_key,
        uq_key,
        uncertainty_threshold,
        check_consistency,
        pbc,
        dtype,
        selected_atom_indices,
    ):
        self.model = model
        self.model_path = model_path
        self.extensions_directory = extensions_directory
        self.types = types
        self.extras = extras
        self.neighbor_lists = neighbor_lists
        self.nl_options = nl_options
        self.options = options
        self.energy_key = energy_key
        self.nc_forces_key = nc_forces_key
        self.uq_key = uq_key
        self.uncertainty_threshold = uncertainty_threshold
        self.check_consistency = check_consistency
        self.pbc = pbc
        self.dtype = dtype
        self.selected_atom_indices = selected_atom_indices

    def __call__(self, state):
        import torch
        from metatomic.torch import System

        positions = np.asarray(state.getPositions(asNumpy=True), dtype=np.float64)
        device = self.types.device
        pos = torch.tensor(positions, dtype=self.dtype, device=device)
        if bool(self.pbc.any()):
            cell = torch.tensor(
                np.asarray(state.getPeriodicBoxVectors(asNumpy=True), dtype=np.float64),
                dtype=self.dtype,
                device=device,
            )
            cell = cell * self.pbc.to(dtype=self.dtype).unsqueeze(1)
        else:
            cell = torch.zeros((3, 3), dtype=self.dtype, device=device)

        do_force_grad = self.nc_forces_key is None
        if do_force_grad:
            pos.requires_grad_(True)

        system = System(self.types, pos, cell, self.pbc)
        for name, tensor in self.extras.items():
            system.add_data(name, tensor)
        if self.neighbor_lists:
            if system.device.type not in ("cpu", "cuda"):
                system = system.to(device="cpu")
            for neighbors in self.neighbor_lists:
                neighbors.add_neighbor_list(systems=[system], copy=False)
            if system.device != device:
                system = system.to(device=device)

        outputs = self.model([system], self.options, self.check_consistency)
        energy = outputs[self.energy_key].block().values.sum()
        if self.uq_key is not None:
            uncertainty = outputs[self.uq_key].block().values.detach().cpu().numpy()
            threshold = self.uncertainty_threshold
            if np.any(uncertainty > threshold):
                warnings.warn(
                    "Some of the atomic energy uncertainties are larger than the "
                    f"threshold of {threshold} kJ/mol. The prediction is above the "
                    f"threshold for atoms {np.where(uncertainty > threshold)[0]}.",
                    stacklevel=2,
                )
        if do_force_grad:
            energy.backward()
        if self.nc_forces_key is not None:
            nc_forces = (
                outputs[self.nc_forces_key].block().values.detach().reshape(-1, 3)
            )
            nc_forces = nc_forces - nc_forces.mean(dim=0, keepdim=True)
            nc_forces = nc_forces.cpu().numpy()
            forces = np.zeros((len(self.types), 3), dtype=np.float64)
            if self.selected_atom_indices is None:
                forces[:] = nc_forces
            else:
                forces[self.selected_atom_indices] = nc_forces
        else:
            grad = system.positions.grad
            if grad is None:
                raise RuntimeError(
                    "model energy does not depend on positions; cannot compute forces"
                )
            forces = (-grad).detach().cpu().numpy()
        return float(energy.detach()), forces

    def __getstate__(self):
        return {
            "model_path": self.model_path,
            "extensions_directory": self.extensions_directory,
            "types": self.types.detach().cpu(),
            "extras": {k: v.to("cpu") for k, v in self.extras.items()},
            "nl_options": self.nl_options,
            "options": self.options,
            "energy_key": self.energy_key,
            "nc_forces_key": self.nc_forces_key,
            "uq_key": self.uq_key,
            "uncertainty_threshold": self.uncertainty_threshold,
            "check_consistency": self.check_consistency,
            "pbc": self.pbc.detach().cpu(),
            "dtype_name": str(self.dtype).removeprefix("torch."),
            "device": str(self.types.device),
            "selected_atom_indices": self.selected_atom_indices,
        }

    def __setstate__(self, state):
        import torch
        from metatomic.torch import load_atomistic_model

        device = torch.device(state["device"])
        self.model_path = state["model_path"]
        self.extensions_directory = state["extensions_directory"]
        self.model = load_atomistic_model(
            self.model_path, extensions_directory=self.extensions_directory
        ).to(device=device)
        self.types = state["types"].to(device=device)
        self.extras = {k: v.to(device=device) for k, v in state["extras"].items()}
        self.nl_options = state["nl_options"]
        self.options = state["options"]
        self.energy_key = state["energy_key"]
        self.nc_forces_key = state["nc_forces_key"]
        self.uq_key = state["uq_key"]
        self.uncertainty_threshold = state["uncertainty_threshold"]
        self.check_consistency = state["check_consistency"]
        self.pbc = state["pbc"].to(device=device)
        self.dtype = getattr(torch, state["dtype_name"])
        self.selected_atom_indices = state["selected_atom_indices"]
        self.neighbor_lists = []
        if self.nl_options:
            import vesin.metatomic

            self.neighbor_lists = [
                vesin.metatomic.NeighborList(
                    options=options,
                    length_unit="nm",
                    check_consistency=self.check_consistency,
                    skin=_NL_SKIN,
                )
                for options in self.nl_options
            ]
