"""Closed vocabularies shared by data schemas and runtime types.

They live apart from the schemas so that protocols can refer to them without
importing pydantic models, and so that every list of allowed values sits in one
place. Each of these is a registry in the sense of ADR-0001: adding a member is
a decision, not a tweak.
"""

from __future__ import annotations

from enum import StrEnum


class AngleUnit(StrEnum):
    """Unit of every angular quantity in a contract."""

    RAD = "rad"
    DEG = "deg"


class ControlMode(StrEnum):
    """How a robot accepts commands (ARCHITECTURE.md, section 3).

    Orthogonal to capabilities and decisive for whether a skill is executable at
    all: a policy running a 50 Hz loop cannot drive a ``command`` robot.
    """

    STREAMING = "streaming"
    """Accepts a command stream at a fixed rate (ros2_control, DDS SDK)."""

    COMMAND = "command"
    """Accepts "perform this motion" and reports completion (REST API)."""


class AccessLevel(StrEnum):
    """Depth of control for robots with a built-in controller."""

    HIGH = "high"
    """The built-in controller executes high-level commands."""

    PARTIAL = "partial"
    """The built-in controller holds balance, the user drives part of the body."""

    LOW = "low"
    """Full control, built-in controller off. Mutually exclusive with HIGH."""


class ObservationType(StrEnum):
    """Terms a policy may observe.

    Proprioception only. Images are absent by decision, not by omission:
    ADR-0005 forbids cameras in training, because Isaac Sim already takes 3-4
    GiB of the 8 GiB VRAM budget.
    """

    JOINT_POS = "joint_pos"
    JOINT_VEL = "joint_vel"
    JOINT_TORQUE = "joint_torque"
    BASE_LIN_VEL = "base_lin_vel"
    BASE_ANG_VEL = "base_ang_vel"
    BASE_ORIENTATION = "base_orientation"
    PREVIOUS_ACTION = "previous_action"


class ActionType(StrEnum):
    """What a policy or skill emits."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    BASE_VELOCITY = "base_velocity"
    CARTESIAN_POSE = "cartesian_pose"
    GRIPPER_WIDTH = "gripper_width"


class Status(StrEnum):
    """Result of one skill tick."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class Role(StrEnum):
    """Role of a robot inside a training scenario (ARCHITECTURE.md, section 7)."""

    LEARNER = "learner"
    """Receives actions from the RL algorithm."""

    FROZEN = "frozen"
    """Runs an already trained policy."""

    SCRIPTED = "scripted"
    """Simple logic inside the simulator."""

    REPLAY = "replay"
    """Replays a recorded trajectory from a past run (ADR-0003)."""

    STATIC = "static"
    """An obstacle."""


class Severity(StrEnum):
    """Severity of a validator finding."""

    ERROR = "error"
    WARNING = "warning"
