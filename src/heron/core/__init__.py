"""Core: data models, protocols, capability registry, contract identity.

Nothing here knows a robot name, a vendor SDK or a simulator (invariant 1).
The check is mechanical: delete ``examples/robots`` and the core still builds
and passes its tests.
"""

from .branding import APP_NAME, APP_TITLE, DOCKER_NAMESPACE, ENV_PREFIX, env_var
from .capabilities import CORE_CAPABILITIES, Capability, CapabilityError
from .contract import ActionSpec, Contract, FieldDiff, ObservationTerm, compatible, diff
from .enums import (
    AccessLevel,
    ActionType,
    AngleUnit,
    ControlMode,
    ObservationType,
    Role,
    Severity,
    Status,
)
from .models import (
    AdapterSpec,
    Limits,
    PolicyManifest,
    RobotProfile,
    Scenario,
    ScenarioRobot,
    SkillSpec,
    Task,
    World,
)
from .protocols import ControlAdapter, SimBackend, Skill, SkillContext, TrainBackend
from .requires import Requirements
from .types import Command, Pose, RobotState

__all__ = [
    "APP_NAME",
    "APP_TITLE",
    "CORE_CAPABILITIES",
    "DOCKER_NAMESPACE",
    "ENV_PREFIX",
    "AccessLevel",
    "ActionSpec",
    "ActionType",
    "AdapterSpec",
    "AngleUnit",
    "Capability",
    "CapabilityError",
    "Command",
    "Contract",
    "ControlAdapter",
    "ControlMode",
    "FieldDiff",
    "Limits",
    "ObservationTerm",
    "ObservationType",
    "PolicyManifest",
    "Pose",
    "Requirements",
    "RobotProfile",
    "RobotState",
    "Role",
    "Scenario",
    "ScenarioRobot",
    "Severity",
    "SimBackend",
    "Skill",
    "SkillContext",
    "SkillSpec",
    "Status",
    "Task",
    "TrainBackend",
    "World",
    "compatible",
    "diff",
    "env_var",
]
