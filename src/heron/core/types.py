"""Runtime values exchanged between a skill, an adapter and the world.

Deliberately built from plain tuples rather than arrays: the core installs
without vendor dependencies (section 10), and pulling a numerical stack into it
for six floats would make that claim false. Training works with dense arrays,
but training lives behind a file boundary (invariant 2) and never shares these
types.

Ordering is not documented per field: every vector here follows
``contract.joint_order``, and that is the only ordering the platform has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ActionType


@dataclass(frozen=True, slots=True)
class Pose:
    """Position in metres and orientation as a unit quaternion (w, x, y, z)."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RobotState:
    """One reading of a robot, in the units its contract declares."""

    timestamp: float
    """Seconds, monotonic within a run."""

    joint_position: tuple[float, ...] = ()
    joint_velocity: tuple[float, ...] = ()
    joint_torque: tuple[float, ...] = ()
    base: Pose | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific readings. The core never interprets them."""


@dataclass(frozen=True, slots=True)
class Command:
    """One command to a robot, tagged with what kind of quantity it carries."""

    type: ActionType
    values: tuple[float, ...] = ()
    pose: Pose | None = None
    """Set instead of ``values`` for ``CARTESIAN_POSE``."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Identity of a run, attached to everything the store records."""

    scenario: str
    seed: int | None = None
    started_at: float = 0.0
