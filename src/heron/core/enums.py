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


class MotionPhase(StrEnum):
    """Where an accepted motion of a ``command`` robot is (ADR-0049). Closed.

    "Done" alone does not prove arrival: a buffer emptied by a stop in the middle
    of a path looks like one emptied by the end of it. So a skill reads the phase
    for the end of a motion and the pose for arrival, and needs both.
    """

    IDLE = "idle"
    """No accepted motion is unfinished."""

    MOVING = "moving"
    """The accepted motion is being executed."""

    HALTED = "halted"
    """The accepted motion is neither executing nor finished: a pause, a protective
    stop, a safety status."""


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
    JOINT_POS_REL = "joint_pos_rel"
    """``q - default_pose``: the pose measured from the contract's reference (ADR-0042)."""

    JOINT_VEL = "joint_vel"
    JOINT_TORQUE = "joint_torque"
    BASE_LIN_VEL = "base_lin_vel"
    BASE_ANG_VEL = "base_ang_vel"
    BASE_ORIENTATION = "base_orientation"
    PREVIOUS_ACTION = "previous_action"
    """The policy's output of the previous tick after ``clip``, before ``scale``."""

    TARGET_JOINT_POS = "target_joint_pos"
    """The goal as an absolute joint pose, ordered like ``joint_order`` (ADR-0042, ADR-0043)."""


class ActionOffset(StrEnum):
    """What a policy's scaled output is added to (ADR-0041).

    A closed vocabulary rather than a flag: an action relative to the measured pose
    would be a third member, not a change of the field's type.
    """

    ZERO = "zero"
    DEFAULT_POSE = "default_pose"
    """The contract's ``default_pose``, the same reference the observation uses."""


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


class TwinKind(StrEnum):
    """What the other side of a link in simulation is (ADR-0021).

    Declared by the twin, never inferred: whether a vendor's program steps the
    physics itself decides whether anything else may share its scene (ADR-0002).
    """

    OWNS_PHYSICS = "owns_physics"
    """A whole simulator of its own. Its scene holds exactly one robot."""

    BRIDGE = "bridge"
    """Code running inside the world of a backend. Bridges share a scene."""


class Severity(StrEnum):
    """Severity of a validator finding."""

    ERROR = "error"
    WARNING = "warning"
