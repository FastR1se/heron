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

from .enums import ActionOffset, ActionType, AngleUnit, ObservationType

CONTRACT_ID_LENGTH: Final = 16
"""Hex characters kept from the digest (ADR-0004)."""

NORMATIVE_FIELDS: Final = (
    "joint_order",
    "angle_unit",
    "default_pose",
    "observation",
    "action",
    "command_rate_hz",
    "state_rate_hz",
)
"""Fields that enter the hash. ``revision`` is deliberately absent."""

GAIN_FIELDS: Final = ("kp", "kd")
"""The drive's coefficients, per joint, stated together or not at all (ADR-0019)."""

PER_JOINT_ACTION_FIELDS: Final = (*GAIN_FIELDS, "scale")
"""Lists of the action ordered like ``joint_order``."""

UNSTATED_ACTION_FIELDS: Final = PER_JOINT_ACTION_FIELDS
"""Fields of the action that appear in the canonical form only when stated.

Normative when present -- a policy trained against one stiffness, or one scale, is
not the same policy against another -- and absent from the form when absent, so
that every identifier made before the field existed stays what it was. ADR-0019
made the rule for gains; ADR-0041 made it general: ``default_pose`` when absent and
``offset`` at ``zero`` are left out the same way."""

JOINT_ACTIONS: Final = frozenset(
    {ActionType.JOINT_POSITION, ActionType.JOINT_VELOCITY, ActionType.JOINT_TORQUE}
)
"""Actions with one value per joint of ``joint_order``."""

JOINT_TERMS: Final = frozenset(
    {
        ObservationType.JOINT_POS,
        ObservationType.JOINT_POS_REL,
        ObservationType.JOINT_VEL,
        ObservationType.JOINT_TORQUE,
        ObservationType.TARGET_JOINT_POS,
    }
)
"""Observation terms with one value per joint of ``joint_order`` (ADR-0042)."""

BASE_OBSERVATION_WIDTHS: Final[dict[ObservationType, int]] = {
    ObservationType.BASE_LIN_VEL: 3,
    ObservationType.BASE_ANG_VEL: 3,
    ObservationType.BASE_ORIENTATION: 4,
}
"""Widths of terms that are not per joint. Orientation is ``w, x, y, z``, as ``Pose``."""


class ObservationTerm(BaseModel):
    """One entry of the observation vector. Order is normative.

    Its value is the quantity times ``scale``, and nothing else: noise is a
    randomization of training, a clip of the term is not used, and normalization
    lives inside the exported graph (ADR-0042).
    """

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
    """What the policy emits, how it is bounded, and what it means in joint units.

    A position action may carry the gains that close it into a torque, in the
    form a joint drive takes them: ``tau = kp * (q* - q) + kd * (dq* - dq)``.
    Without them "joint position" means whatever the world that executes it
    decides -- a stiff servo in one simulator, a spring on the hardware -- and
    that difference is invisible until a robot behaves differently in each
    (ADR-0019).

    A policy's raw output ``a`` becomes a command by one mapping in every world
    (ADR-0041)::

        u  = clip(a, clip)                     over the raw output, unitless
        q* = u * scale + origin                origin = default_pose or 0

    For a skill written as code there is no raw output, and the three fields mean
    nothing to it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActionType
    clip: tuple[float, float] | None = None
    """Bounds of the raw output of a policy, before ``scale``. Unitless."""

    scale: list[float] | None = None
    """Per joint, ordered like ``joint_order``. Unstated means one."""

    offset: ActionOffset = ActionOffset.ZERO
    """What the scaled output is added to."""

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

    @model_validator(mode="after")
    def _mapping_consistent(self) -> ActionSpec:
        if self.scale is not None:
            if self.type not in JOINT_ACTIONS:
                raise ValueError(
                    f"scale — по суставу, а у действия {self.type} значений по суставам нет"
                )
            zero = [index for index, value in enumerate(self.scale) if value == 0.0]
            if zero:
                raise ValueError(
                    f"scale = 0 в суставах с индексами {zero}: такой сустав политике не подвластен"
                )
        if self.offset is ActionOffset.DEFAULT_POSE and self.type is not ActionType.JOINT_POSITION:
            raise ValueError(
                f"offset: default_pose — это поза, а действие {self.type} позой не является"
            )
        return self

    @property
    def has_gains(self) -> bool:
        """Whether the action is closed into a torque by gains it states."""
        return self.kp is not None

    @property
    def maps_output(self) -> bool:
        """Whether a raw output is transformed before it becomes a command (ADR-0041)."""
        return self.scale is not None or self.offset is not ActionOffset.ZERO


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
    default_pose: list[float] | None = None
    """The reference pose, per joint, in ``angle_unit``. One quantity for the action
    offset and for ``joint_pos_rel`` (ADR-0041). Normative when stated."""

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
        for name in PER_JOINT_ACTION_FIELDS:
            values = getattr(self.action, name)
            if values is not None and len(values) != len(self.joint_order):
                raise ValueError(
                    f"action.{name}: {len(values)} значений при {len(self.joint_order)} "
                    f"суставах; значения упорядочены как joint_order"
                )
        if self.action.has_gains and self.angle_unit is not AngleUnit.RAD:
            raise ValueError(
                "kp и kd заданы в единицах привода (Н·м/рад), а контракт — в градусах: "
                "один и тот же коэффициент значил бы разное. Контракт с коэффициентами "
                "объявляет angle_unit: rad"
            )
        return self

    @model_validator(mode="after")
    def _reference_pose_consistent(self) -> Contract:
        if self.default_pose is not None and len(self.default_pose) != len(self.joint_order):
            raise ValueError(
                f"default_pose: {len(self.default_pose)} значений при "
                f"{len(self.joint_order)} суставах; поза упорядочена как joint_order"
            )
        if self.default_pose is None:
            if self.action.offset is ActionOffset.DEFAULT_POSE:
                raise ValueError(
                    "action.offset: default_pose, а default_pose в контракте не задана"
                )
            if any(term.type is ObservationType.JOINT_POS_REL for term in self.observation):
                raise ValueError(
                    "терм joint_pos_rel отсчитывается от default_pose, а она не задана"
                )
        return self

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    @property
    def action_size(self) -> int | None:
        """Width of the action vector, when it is a vector per joint (ADR-0042)."""
        return self.dof if self.action.type in JOINT_ACTIONS else None

    def term_size(self, term: ObservationType) -> int | None:
        """Width of one observation term, derived rather than declared."""
        if term in JOINT_TERMS:
            return self.dof
        if term is ObservationType.PREVIOUS_ACTION:
            return self.action_size
        return BASE_OBSERVATION_WIDTHS[term]

    @property
    def observation_size(self) -> int | None:
        """Width of the observation vector; ``None`` when a term has no known width."""
        sizes = [self.term_size(term.type) for term in self.observation]
        return None if any(size is None for size in sizes) else sum(s for s in sizes if s)

    def origin(self) -> tuple[float, ...]:
        """What a scaled output is added to (ADR-0041)."""
        if self.action.offset is ActionOffset.DEFAULT_POSE and self.default_pose is not None:
            return tuple(self.default_pose)
        return (0.0,) * self.dof

    def action_scale(self) -> tuple[float, ...]:
        """Scale per joint; unstated is one."""
        return tuple(self.action.scale) if self.action.scale is not None else (1.0,) * self.dof

    def normative(self) -> dict[str, Any]:
        """The fields that define identity, as plain data.

        An optional field left unstated is left out, not written as a default:
        that is what keeps an identifier made before the field existed.
        """
        dumped = self.model_dump(mode="json", include=set(NORMATIVE_FIELDS))
        action = dumped["action"]
        for name in UNSTATED_ACTION_FIELDS:
            if action.get(name) is None:
                action.pop(name, None)
        if action.get("offset") == ActionOffset.ZERO.value:
            action.pop("offset")
        if dumped.get("default_pose") is None:
            dumped.pop("default_pose", None)
        return {name: dumped[name] for name in NORMATIVE_FIELDS if name in dumped}

    def canonical_json(self) -> str:
        """Canonical serialization of the normative fields.

        Pinned form: keys sorted, no whitespace, ASCII-escaped, floats in
        Python's shortest round-trip representation, unstated optional fields
        omitted. Changing any of these changes every identifier in existence.
        """
        return canonical_dumps(self.normative())

    @property
    def contract_id(self) -> str:
        """Identity of the representation. Computed, never written by hand."""
        return digest(self.canonical_json())


def canonical_dumps(data: Any) -> str:
    """The pinned JSON form (ADR-0006), for anything identified by its content."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def digest(text: str) -> str:
    """``sha256(text)[:16]``: how the platform names content (ADR-0004, ADR-0046)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONTRACT_ID_LENGTH]


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
    to retrain the policy or fix the profile. A field one side leaves unstated
    shows as ``None`` on that side.
    """
    a, b = left.normative(), right.normative()
    return [
        FieldDiff(field=name, left=a.get(name), right=b.get(name))
        for name in NORMATIVE_FIELDS
        if a.get(name) != b.get(name)
    ]


def compatible(left: Contract, right: Contract) -> bool:
    """Compatibility is equality of identifiers, nothing weaker (ADR-0004)."""
    return left.contract_id == right.contract_id
