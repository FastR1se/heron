"""What a robot package may import from the platform (ADR-0021).

A vendor's package that reaches into the validator or the command line works on
the day it is written and proves nothing: the claim that a robot joins through
the extension points is only true if it joins through nothing else. A test holds
every package in ``examples/plugins`` to this list, the way the CI job without
``examples/`` holds the core to invariant 1.

Modules are named relative to the package root, so the list survives a rename
(section 17). Adding one widens a promise made to every robot package, and is a
decision rather than an edit.
"""

from __future__ import annotations

from typing import Final

PUBLIC_MODULES: Final = frozenset(
    {
        # The data a robot is described by and the words it is described in.
        "core.capabilities",
        "core.contract",
        "core.enums",
        "core.environment",
        "core.mjcf",
        "core.models",
        "core.types",
        "core.units",
        "core.urdf",
        # The five extension points.
        "core.protocols",
        # What a twin in a Gazebo world contributes, and the values it returns.
        "generators.common",
        "generators.twin",
        # What a simulation backend and a twin's processes are started with.
        "sim.base",
        "sim.process",
        # What a twin that is a MuJoCo world of its own contributes (ADR-0030).
        "sim.mujoco_twin",
    }
)
