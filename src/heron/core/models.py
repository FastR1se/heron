"""Data schemas: robot profile, scenario, skill declaration, policy manifest.

A profile is data, not code (invariant 5): adding a robot must not touch the
core. Every schema is strict -- unknown fields are rejected rather than ignored,
because a typo that is silently dropped is exactly the class of error the
platform exists to catch.

Schemas carry ``schema_version`` and never the platform name (section 17): a
rename must not have to rewrite user data.
"""

from __future__ import annotations

import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .capabilities import Capability
from .contract import Contract
from .enums import AccessLevel, ControlMode, Role
from .requires import Requirements

NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
SIM_TARGET_PREFIX: Final = "sim."
REAL_TARGET: Final = "real"


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError(f"{value!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
    return value


def is_sim_target(target: str) -> bool:
    """True for an adapter key naming a simulation backend (``sim.gazebo``)."""
    return target.startswith(SIM_TARGET_PREFIX)


def sim_backend_of(target: str) -> str | None:
    """Backend named by a ``sim.*`` adapter key, or ``None`` for ``real``."""
    return target[len(SIM_TARGET_PREFIX) :] if is_sim_target(target) else None


def sim_target(backend: str) -> str:
    """Adapter key a scenario with this world backend requires (ADR-0002)."""
    return f"{SIM_TARGET_PREFIX}{backend}"


class ModelDescription(BaseModel):
    """Where the kinematic description lives. Generators read it, core does not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    urdf: str
    mesh_dir: str | None = None


class JointRange(BaseModel):
    """Per-joint bounds, ordered like ``contract.joint_order``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: list[float] = Field(min_length=1)
    max: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> JointRange:
        if len(self.min) != len(self.max):
            raise ValueError(f"min ({len(self.min)}) и max ({len(self.max)}) разной длины")
        wrong = [i for i, (lo, hi) in enumerate(zip(self.min, self.max, strict=True)) if lo > hi]
        if wrong:
            raise ValueError(f"min > max в суставах с индексами {wrong}")
        return self


class JointMax(BaseModel):
    """Per-joint upper bound on a signed quantity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max: list[float] = Field(min_length=1)

    @field_validator("max")
    @classmethod
    def _positive(cls, value: list[float]) -> list[float]:
        wrong = [i for i, v in enumerate(value) if v <= 0]
        if wrong:
            raise ValueError(f"предел должен быть положительным, нарушено в индексах {wrong}")
        return value


class Limits(BaseModel):
    """Safety envelope. Outside the contract hash: limits do not change meaning
    of the data, they bound it (ADR-0004)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    joint_position: JointRange | None = None
    joint_velocity: JointMax | None = None
    joint_torque: JointMax | None = None
    cartesian_velocity_max: float | None = Field(default=None, gt=0)
    gripper_width_max: float | None = Field(default=None, gt=0)


class AdapterSpec(BaseModel):
    """A plugin and its configuration.

    Invariant 3 lives here: the same image runs in simulation and on hardware,
    and only this block differs between them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin: str
    """Entry point name, or ``module:Class`` for a plugin not yet installed."""

    params: dict[str, Any] = Field(default_factory=dict)


class RobotProfile(BaseModel):
    """Full declarative description of a robot. The single source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str
    description: ModelDescription
    contract: Contract
    capabilities: frozenset[Capability] = Field(min_length=1)
    control_mode: ControlMode
    access_levels: tuple[AccessLevel, ...] = ()
    limits: Limits = Field(default_factory=Limits)
    adapters: dict[str, AdapterSpec] = Field(min_length=1)
    skills: tuple[str, ...] = ()
    image: str | None = None
    """Bare image name (``robot-arm:latest``). The namespace is prepended by
    ``branding.image_name``; it is not part of the data (section 17)."""

    _validate_name = field_validator("name")(_check_name)

    @field_validator("adapters")
    @classmethod
    def _adapter_keys(cls, value: dict[str, AdapterSpec]) -> dict[str, AdapterSpec]:
        for target in value:
            if target == REAL_TARGET:
                continue
            backend = sim_backend_of(target)
            if backend is None or not NAME_RE.match(backend):
                raise ValueError(
                    f"{target!r}: ключ адаптера — {REAL_TARGET!r} или "
                    f"{SIM_TARGET_PREFIX}<бэкенд>, например {sim_target('gazebo')!r}"
                )
        return value

    @model_validator(mode="after")
    def _access_levels_consistent(self) -> RobotProfile:
        levels = set(self.access_levels)
        if AccessLevel.LOW in levels and AccessLevel.HIGH in levels:
            raise ValueError(
                "уровни доступа low и high взаимоисключающи: низкоуровневое управление "
                "означает отключённый встроенный контроллер (раздел 3)"
            )
        if len(levels) != len(self.access_levels):
            raise ValueError("повтор в access_levels")
        return self

    @property
    def sim_backends(self) -> frozenset[str]:
        """Simulation backends this robot has an adapter for."""
        return frozenset(
            backend for target in self.adapters if (backend := sim_backend_of(target)) is not None
        )

    @property
    def has_real_adapter(self) -> bool:
        return REAL_TARGET in self.adapters


class SkillSpec(BaseModel):
    """What a skill demands of a robot. Code and policy declare it the same way
    (invariant 4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str
    required_capabilities: frozenset[Capability] = Field(min_length=1)
    required_control_mode: ControlMode = ControlMode.STREAMING
    requires: Requirements = Field(default_factory=Requirements)

    _validate_name = field_validator("name")(_check_name)


class World(BaseModel):
    """The single physics authority of a scenario (ADR-0002)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    file: str | None = None

    _validate_backend = field_validator("backend")(_check_name)


class ScenarioRobot(BaseModel):
    """One participant of a scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    profile: str
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    role: Role = Role.STATIC
    policy: str | None = None
    """Artifact id of the frozen policy. Required for role ``frozen``."""

    trajectory: str | None = None
    """Artifact id of the recorded trajectory. Required for role ``replay``."""

    _validate_id = field_validator("id")(_check_name)

    @model_validator(mode="after")
    def _role_payload(self) -> ScenarioRobot:
        if self.role is Role.FROZEN and self.policy is None:
            raise ValueError(f"{self.id}: роль frozen требует поля policy")
        if self.role is Role.REPLAY and self.trajectory is None:
            raise ValueError(f"{self.id}: роль replay требует поля trajectory (ADR-0003)")
        if self.role is not Role.FROZEN and self.policy is not None:
            raise ValueError(f"{self.id}: поле policy имеет смысл только при роли frozen")
        if self.role is not Role.REPLAY and self.trajectory is not None:
            raise ValueError(f"{self.id}: поле trajectory имеет смысл только при роли replay")
        return self


class Task(BaseModel):
    """What the scenario asks for. Everything besides ``skill`` is the goal,
    passed to the skill untouched -- the core does not interpret it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill: str
    goal: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_goal(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "skill" not in data:
            return data
        goal = dict(data.get("goal", {}))
        goal.update({k: v for k, v in data.items() if k not in {"skill", "goal"}})
        return {"skill": data["skill"], "goal": goal}


class Scenario(BaseModel):
    """Description of a robotic system: world, participants, task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str
    world: World
    robots: tuple[ScenarioRobot, ...] = Field(min_length=1)
    task: Task | None = None

    _validate_name = field_validator("name")(_check_name)

    @model_validator(mode="after")
    def _unique_ids(self) -> Scenario:
        ids = [robot.id for robot in self.robots]
        duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
        if duplicates:
            raise ValueError(f"повтор идентификаторов роботов: {', '.join(duplicates)}")
        return self

    @property
    def learners(self) -> tuple[ScenarioRobot, ...]:
        return tuple(robot for robot in self.robots if robot.role is Role.LEARNER)


class PolicyManifest(BaseModel):
    """JSON companion of an exported ONNX policy (invariant 2, ADR-0004).

    The contract travels with the artifact, and the identifier it was created
    under travels next to the contract, so that a tampered file fails the check
    instead of passing it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str
    contract: Contract
    contract_id: str
    capabilities: frozenset[Capability] = Field(min_length=1)
    control_mode: ControlMode = ControlMode.STREAMING

    _validate_name = field_validator("name")(_check_name)

    @property
    def self_consistent(self) -> bool:
        """Whether the recorded identifier matches the carried contract."""
        return self.contract_id == self.contract.contract_id
