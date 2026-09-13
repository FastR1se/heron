"""The contract: how data is represented, and how that representation is identified.

ARCHITECTURE.md section 6 calls the contract the heart of the platform. It
travels with a policy on export and is checked on every use, because its silent
divergence produces the classic failure -- a robot going mad without a single
line in the logs.

ADR-0004 gives the contract an identity of its own, derived from content rather
than from the profile version::

    contract_id = sha256(canonical JSON of the normative fields)[:16]

Every field of :class:`Contract` is normative except ``revision``. That is the
whole rule, and it is why the class holds nothing else: a field that does not
change the meaning of the data does not belong here, it belongs elsewhere in the
profile. ``revision`` is a declaration of intent -- content changed while
``revision`` did not is an error, so a representation change cannot be made
without saying so.

The canonical form is public behaviour of the core, not an implementation
detail: an identifier that shifts with a serializer upgrade would invalidate
every stored policy. It is therefore pinned here and covered by a golden test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import ActionType, AngleUnit, ObservationType

CONTRACT_ID_LENGTH: Final = 16
"""Hex characters kept from the digest (ADR-0004)."""

NORMATIVE_FIELDS: Final = (
    "joint_order",
    "angle_unit",
    "observation",
    "action",
    "command_rate_hz",
    "state_rate_hz",
)
"""Fields that enter the hash. ``revision`` is deliberately absent."""

GAIN_FIELDS: Final = ("kp", "kd")
"""Fields of the action that appear in the canonical form only when stated.

Normative when present -- a policy trained against one stiffness is not the same
policy against another -- and absent from the form when absent, so that every
identifier made before gains existed stays what it was (ADR-0019)."""


class ObservationTerm(BaseModel):
    """One entry of the observation vector. Order is normative."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ObservationType
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _finite_nonzero(cls, value: float) -> float:
        if value == 0.0:
            raise ValueError("scale = 0 обнуляет наблюдение; уберите терм вместо этого")
        return value


class ActionSpec(BaseModel):
    """What the policy emits and how it is bounded.

    A position action may carry the gains that close it into a torque, in the
    form a joint drive takes them: ``tau = kp * (q* - q) + kd * (dq* - dq)``.
    Without them "joint position" means whatever the world that executes it
    decides -- a stiff servo in one simulator, a spring on the hardware -- and
    that difference is invisible until a robot behaves differently in each
    (ADR-0019).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActionType
    clip: tuple[float, float] | None = None
    kp: list[float] | None = None
    """Stiffness per joint in N*m/rad (N/m for a prismatic joint), ordered like
    ``joint_order``."""

    kd: list[float] | None = None
    """Damping per joint in N*m*s/rad (N*s/m), ordered like ``joint_order``."""

    @field_validator("clip")
    @classmethod
    def _ordered(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] >= value[1]:
            raise ValueError(f"clip {value}: нижняя граница должна быть меньше верхней")
        return value

    @model_validator(mode="after")
    def _gains_consistent(self) -> ActionSpec:
        if (self.kp is None) != (self.kd is None):
            raise ValueError(
                "kp и kd задаются вместе: PD-закон без одного из коэффициентов — "
                "это другой закон, а не тот же с умолчанием"
            )
        if self.kp is None or self.kd is None:
            return self
        if self.type is not ActionType.JOINT_POSITION:
            raise ValueError(
                f"kp и kd замыкают цель по положению в момент; у действия {self.type} "
                f"цели по положению нет"
            )
        if any(value <= 0 for value in self.kp):
            raise ValueError("kp должен быть положительным: нулевая жёсткость не держит цель")
        if any(value < 0 for value in self.kd):
            raise ValueError("kd не может быть отрицательным: это раскачка, а не демпфирование")
        return self

    @property
    def has_gains(self) -> bool:
        """Whether the action is closed into a torque by gains it states."""
        return self.kp is not None


class Contract(BaseModel):
    """Agreement on data representation between worlds.

    Frozen on purpose: an identifier computed from a mutable object would be a
    promise the class cannot keep.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=1)
    """Declared intent. Outside the hash, checked against it (ADR-0004)."""

    joint_order: list[str] = Field(min_length=1)
    angle_unit: AngleUnit
    state_rate_hz: float = Field(gt=0)
    command_rate_hz: float = Field(gt=0)
    observation: list[ObservationTerm] = Field(min_length=1)
    action: ActionSpec

    @field_validator("joint_order")
    @classmethod
    def _unique_joints(cls, value: list[str]) -> list[str]:
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"joint_order содержит повторы: {', '.join(duplicates)}. "
                f"Порядок суставов — нормативное поле, повтор делает его неоднозначным"
            )
        return value

    @model_validator(mode="after")
    def _gains_fit_the_joints(self) -> Contract:
        for name in GAIN_FIELDS:
            values = getattr(self.action, name)
            if values is not None and len(values) != len(self.joint_order):
                raise ValueError(
                    f"action.{name}: {len(values)} значений при {len(self.joint_order)} "
                    f"суставах; коэффициенты упорядочены как joint_order"
                )
        if self.action.has_gains and self.angle_unit is not AngleUnit.RAD:
            raise ValueError(
                "kp и kd заданы в единицах привода (Н·м/рад), а контракт — в градусах: "
                "один и тот же коэффициент значил бы разное. Контракт с коэффициентами "
                "объявляет angle_unit: rad"
            )
        return self

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    def normative(self) -> dict[str, Any]:
        """The fields that define identity, as plain data."""
        dumped = self.model_dump(mode="json", include=set(NORMATIVE_FIELDS))
        action = dumped["action"]
        for name in GAIN_FIELDS:
            if action.get(name) is None:
                action.pop(name, None)
        return {name: dumped[name] for name in NORMATIVE_FIELDS}

    def canonical_json(self) -> str:
        """Canonical serialization of the normative fields.

        Pinned form: keys sorted, no whitespace, ASCII-escaped, floats in
        Python's shortest round-trip representation, unstated gains omitted.
        Changing any of these changes every identifier in existence.
        """
        return json.dumps(
            self.normative(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def contract_id(self) -> str:
        """Identity of the representation. Computed, never written by hand."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return digest[:CONTRACT_ID_LENGTH]


@dataclass(frozen=True, slots=True)
class FieldDiff:
    """One normative field that differs between two contracts."""

    field: str
    left: Any
    right: Any

    def describe(self, left_label: str = "слева", right_label: str = "справа") -> str:
        return f"{self.field}: {left_label} {self.left!r}, {right_label} {self.right!r}"


def diff(left: Contract, right: Contract) -> list[FieldDiff]:
    """Normative fields where two contracts disagree.

    ADR-0004 requires this: on a mismatch the validator prints the differing
    fields, not the fact that two hashes differ. A human needs to know whether
    to retrain the policy or fix the profile.
    """
    a, b = left.normative(), right.normative()
    return [
        FieldDiff(field=name, left=a[name], right=b[name])
        for name in NORMATIVE_FIELDS
        if a[name] != b[name]
    ]


def compatible(left: Contract, right: Contract) -> bool:
    """Compatibility is equality of identifiers, nothing weaker (ADR-0004)."""
    return left.contract_id == right.contract_id
