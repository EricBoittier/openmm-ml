"""
metatomicpotential.py: Implements potentials from exported metatomic models.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Eric D. Boittier

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

from functools import partial
from typing import Iterable, Optional

import numpy as np
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

_DTYPES = {"float32", "float64"}
_INPUT_DEFAULTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MetatomicPotentialImpl objects."""

    def createImpl(
        self,
        name: str,
        modelPath: str,
        device=None,
        extensionsDirectory=None,
        checkConsistency: bool = False,
        **args,
    ) -> MLPotentialImpl:
        return MetatomicPotentialImpl(
            name, modelPath, device, extensionsDirectory, checkConsistency
        )


class MetatomicPotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl evaluates an exported metatomic model.

    Load a TorchScript model produced by :func:`metatomic.torch.load_atomistic_model`
    (typically a ``.pt`` file) and install a single :class:`openmm.PythonForce`.
    Neighbor lists use ``vesin.metatomic``; forces are ``-dE/dx`` via autograd.
    This backend returns energy and forces only (NVT/NVE). It does not provide
    a virial, so NPT is not supported. CUDA neighbor lists through nvalchemi
    are available on the ASE path, not here.

    >>> potential = MLPotential(
    ...     "metatomic",
    ...     modelPath="exported-model.pt",
    ...     device="cuda",
    ...     extensionsDirectory="./extensions",
    ...     checkConsistency=False,
    ... )
    >>> system = potential.createSystem(topology)

    Optional ``createSystem()`` / ``createMixedSystem()`` arguments:

    - ``charge``: total charge (default 0), used if the model requests it
    - ``multiplicity`` / ``spin_multiplicity``: spin multiplicity (default 1)
    - ``info``: dict with the same keys, matching the ASE calculator convention

    Models that request any other extra input should be run through
    ``MLPotential("ase")`` and :class:`metatomic_ase.MetatomicCalculator` instead.
    """

    def __init__(self, name, modelPath, device, extensionsDirectory, checkConsistency):
        self.name = name
        self.modelPath = modelPath
        self.device = device
        self.extensionsDirectory = extensionsDirectory
        self.checkConsistency = checkConsistency

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
            from metatomic.torch import (
                ModelEvaluationOptions,
                ModelOutput,
                load_atomistic_model,
                pick_device,
                pick_output,
                unit_conversion_factor,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import metatomic. Install it with "
                "'pip install metatomic-torch'."
            ) from e

        if any(atom.element is None for atom in topology.atoms()):
            raise ValueError("All atoms in the Topology must have elements defined.")

        model = load_atomistic_model(
            self.modelPath, extensions_directory=self.extensionsDirectory
        )
        capabilities = model.capabilities()
        desired = self.device
        if desired is not None and not isinstance(desired, str):
            desired = str(desired)
        device = torch.device(pick_device(capabilities.supported_devices, desired))
        dtype_name = capabilities.dtype
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported model dtype '{dtype_name}'.")
        dtype = getattr(torch, dtype_name)
        model = model.to(device=device)
        energy_key = pick_output("energy", capabilities.outputs, None)

        includedAtoms = list(topology.atoms())
        if atoms is None:
            indices = None
        else:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms)
        numbers = [atom.element.atomic_number for atom in includedAtoms]
        unsupported = set(numbers) - set(capabilities.atomic_types)
        if unsupported:
            raise ValueError(
                f"this model does not support atomic types {sorted(unsupported)}"
            )
        types = torch.tensor(numbers, dtype=torch.int32, device=device)

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
                    length_unit="angstrom",
                    check_consistency=self.checkConsistency,
                    skin=2.0,
                )
                for options in requested_nl
            ]

        periodic = (
            topology.getPeriodicBoxVectors() is not None
            or system.usesPeriodicBoundaryConditions()
        )
        pbc = torch.tensor([periodic] * 3, dtype=torch.bool, device=device)
        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={energy_key: ModelOutput(unit="eV", sample_kind="system")},
        )
        energy_scale = float(unit_conversion_factor("eV", "kJ/mol"))

        compute = partial(
            _computeMetatomic,
            model=model,
            types=types,
            extras=extras,
            neighbor_lists=neighbor_lists,
            options=options,
            energy_key=energy_key,
            check_consistency=self.checkConsistency,
            indices=indices,
            periodic=periodic,
            pbc=pbc,
            dtype=dtype,
            energy_scale=energy_scale,
        )
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _unsupported_input(name, sample_kind=None):
    kind = f" (sample_kind={sample_kind!r})" if sample_kind is not None else ""
    return ValueError(
        f"this model requests extra input '{name}'{kind}, which is not "
        "implemented by MLPotential('metatomic'). Use MLPotential('ase') "
        "with metatomic_ase.MetatomicCalculator instead."
    )


def _extra_inputs(requested, args, dtype, device):
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    info = dict(args.get("info") or {})
    extras = {}
    for name, option in requested.items():
        sample_kind = option.sample_kind
        if name not in _INPUT_DEFAULTS or sample_kind != "system":
            raise _unsupported_input(name, sample_kind)
        default, input_unit = _INPUT_DEFAULTS[name]
        if name == "spin_multiplicity":
            value = args.get(
                "spin_multiplicity",
                args.get(
                    "multiplicity",
                    info.get("spin_multiplicity", info.get("spin", default)),
                ),
            )
        else:
            value = args.get(name, info.get(name, default))
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


def _computeMetatomic(
    state,
    model,
    types,
    extras,
    neighbor_lists,
    options,
    energy_key,
    check_consistency,
    indices,
    periodic,
    pbc,
    dtype,
    energy_scale,
):
    import torch
    from metatomic.torch import System

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    pos = torch.tensor(positions, dtype=dtype, device=types.device, requires_grad=True)
    if periodic:
        cell = torch.tensor(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=dtype,
            device=types.device,
        )
    else:
        cell = torch.zeros((3, 3), dtype=dtype, device=types.device)
    system = System(types, pos, cell, pbc)
    for name, tensor in extras.items():
        system.add_data(name, tensor)
    if neighbor_lists:
        if system.device.type not in ("cpu", "cuda"):
            system = system.to(device="cpu")
        for neighbors in neighbor_lists:
            neighbors.add_neighbor_list(systems=[system], copy=False)
        if system.device != types.device:
            system = system.to(device=types.device)
    energy = (
        model([system], options, check_consistency)[energy_key].block().values.sum()
    )
    energy.backward()
    grad = system.positions.grad
    if grad is None:
        raise RuntimeError(
            "model energy does not depend on positions; cannot compute forces"
        )
    forces = (-grad * energy_scale * 10.0).detach().cpu().numpy()
    if indices is not None:
        scattered = np.zeros((numAtoms, 3), dtype=forces.dtype)
        scattered[indices] = forces
        forces = scattered
    return float(energy.detach()) * energy_scale, forces
