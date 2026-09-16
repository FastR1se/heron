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
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from .capabilities import Capability
from .contract import CONTRACT_ID_LENGTH, Contract, canonical_dumps, digest
from .enums import AccessLevel, ControlMode, Role
from .requires import Requirements

NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
SIM_TARGET_PREFIX: Final = "sim."
REAL_TARGET: Final = "real"

POLICY_REF_SEPARATOR: Final = "@"
POLICY_REF_RE: Final = re.compile(
    rf"^([a-z][a-z0-9_]*){POLICY_REF_SEPARATOR}([0-9a-f]{{{CONTRACT_ID_LENGTH}}})$"
)
"""``<name>@<artifact_id>``, the identifier in full (ADR-0046)."""

SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError(f"{value!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
    return value


@dataclass(frozen=True, slots=True)
class PolicyRef:
    """A policy artifact named the one way a profile and a scenario name it (ADR-0046)."""

    name: str
    artifact_id: str

    @classmethod
    def parse(cls, text: str) -> PolicyRef:
        match = POLICY_REF_RE.match(text) if isinstance(text, str) else None
        if match is None:
            raise ValueError(
                f"{text!r}: ссылка на политику — <имя>{POLICY_REF_SEPARATOR}<идентификатор>, "
                f"идентификатор полностью, {CONTRACT_ID_LENGTH} шестнадцатеричных символов"
            )
        return cls(name=match.group(1), artifact_id=match.group(2))

    def __str__(self) -> str:
        return f"{self.name}{POLICY_REF_SEPARATOR}{self.artifact_id}"


def _check_policy_ref(value: str | None) -> str | None:
    if value is not None:
        PolicyRef.parse(value)
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
    mjcf: str | None = None
    """The same robot in MJCF, for a MuJoCo world (ADR-0031). An input like the
    URDF, compared with the profile by the ``mjcf.*`` rules; relative to the
    profile. A MuJoCo scene has to load this file."""


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


class SkillBinding(BaseModel):
    """One entry of a profile's ``skills``: a name, and what implements it (ADR-0046).

    Written as the name alone for a skill of the library, or as a mapping of one key
    for a trained policy under that name::

        skills: [hold, {move_joints: arm6_reach@3f2a9c1d0b7e4a55}]

    A mission names the skill; the profile says how it is made. Swapping one for the
    other is this line (invariant 4).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    policy: str | None = None
    """``<name>@<artifact_id>`` of the artifact that implements the skill, if a policy does."""

    _validate_name = field_validator("name")(_check_name)
    _validate_policy = field_validator("policy")(_check_policy_ref)

    @model_validator(mode="before")
    @classmethod
    def _from_yaml(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if isinstance(data, dict) and set(data) - {"name", "policy"}:
            if len(data) != 1:
                raise ValueError(
                    f"{data!r}: навык — имя или отображение из одного ключа "
                    f"«имя навыка: ссылка на политику»"
                )
            ((name, policy),) = data.items()
            return {"name": name, "policy": policy}
        return data

    @model_serializer
    def _to_yaml(self) -> str | dict[str, str]:
        return self.name if self.policy is None else {self.name: self.policy}

    @property
    def ref(self) -> PolicyRef | None:
        return PolicyRef.parse(self.policy) if self.policy is not None else None

    def __str__(self) -> str:
        return self.name if self.policy is None else f"{self.name} -> {self.policy}"


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
    skills: tuple[SkillBinding, ...] = ()
    """What this robot is to be asked for, and how each is made (ADR-0046)."""

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

    @field_validator("skills")
    @classmethod
    def _unique_skills(cls, value: tuple[SkillBinding, ...]) -> tuple[SkillBinding, ...]:
        names = [item.name for item in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"навык объявлен дважды: {', '.join(duplicates)}. Одно имя — одна реализация"
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

    def skill(self, name: str) -> SkillBinding | None:
        """How this robot implements a skill, or ``None`` if it does not declare it."""
        return next((item for item in self.skills if item.name == name), None)


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
    """``<name>@<artifact_id>`` of the frozen policy, as a profile names one (ADR-0046).
    Required for role ``frozen``."""

    trajectory: str | None = None
    """Artifact id of the recorded trajectory. Required for role ``replay``."""

    _validate_id = field_validator("id")(_check_name)
    _validate_policy = field_validator("policy")(_check_policy_ref)

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


class Completion(BaseModel):
    """When a trained skill has reached its goal, as its training task defined it (ADR-0043).

    ``SUCCESS`` once ``max |q - target| <= tolerance`` has held for ``hold_s`` of world
    time; ``FAILURE`` when that has not happened within ``horizon_s``, the length of a
    training episode -- past it the policy was never trained.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tolerance: float = Field(gt=0)
    """In the contract's ``angle_unit``."""

    hold_s: float = Field(ge=0)
    horizon_s: float = Field(gt=0)


class OnnxSignature(BaseModel):
    """What the exported graph is, recorded next to it (ADR-0046)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    """Of ``policy.onnx``, so that the artifact identifier covers the network too."""

    input: str
    output: str
    observation_size: int = Field(gt=0)
    action_size: int = Field(gt=0)
    opset: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not SHA256_RE.match(value):
            raise ValueError(f"{value!r}: sha256 — 64 шестнадцатеричных символа в нижнем регистре")
        return value


class PolicyManifest(BaseModel):
    """JSON companion of an exported ONNX policy (invariant 2, ADR-0004, ADR-0046).

    The contract travels with the artifact, and the identifier it was created
    under travels next to the contract, so that a tampered file fails the check
    instead of passing it. The artifact itself is named by this document's
    content: its canonical form hashed, which covers the network through
    ``onnx.sha256``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str
    contract: Contract
    contract_id: str
    capabilities: frozenset[Capability] = Field(min_length=1)
    control_mode: ControlMode = ControlMode.STREAMING
    requires: Requirements = Field(default_factory=Requirements)
    completion: Completion
    onnx: OnnxSignature

    _validate_name = field_validator("name")(_check_name)

    @property
    def self_consistent(self) -> bool:
        """Whether the recorded identifier matches the carried contract."""
        return self.contract_id == self.contract.contract_id

    def canonical_json(self) -> str:
        """The pinned form of ADR-0006, applied to the whole manifest.

        The contract enters as its normative form plus ``revision``, so that an
        unstated optional field is absent here as it is from ``contract_id``; the
        capabilities, a set, enter sorted.
        """
        data = self.model_dump(mode="json")
        data["contract"] = {**self.contract.normative(), "revision": self.contract.revision}
        data["capabilities"] = sorted(self.capabilities)
        return canonical_dumps(data)

    @property
    def artifact_id(self) -> str:
        """``sha256(canonical form)[:16]`` -- the name of the artifact (ADR-0046)."""
        return digest(self.canonical_json())

    def ref(self) -> PolicyRef:
        return PolicyRef(name=self.name, artifact_id=self.artifact_id)

    def skill_spec(self, name: str) -> SkillSpec:
        """The declaration of the skill this policy implements, under the name it is bound to."""
        return SkillSpec(
            name=name,
            required_capabilities=self.capabilities,
            required_control_mode=self.control_mode,
            requires=self.requires,
        )
