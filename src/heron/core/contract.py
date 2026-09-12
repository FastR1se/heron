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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    """What the policy emits and how it is bounded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActionType
    clip: tuple[float, float] | None = None

    @field_validator("clip")
    @classmethod
    def _ordered(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] >= value[1]:
            raise ValueError(f"clip {value}: нижняя граница должна быть меньше верхней")
        return value


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

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    def normative(self) -> dict[str, Any]:
        """The fields that define identity, as plain data."""
        dumped = self.model_dump(mode="json", include=set(NORMATIVE_FIELDS))
        return {name: dumped[name] for name in NORMATIVE_FIELDS}

    def canonical_json(self) -> str:
        """Canonical serialization of the normative fields.

        Pinned form: keys sorted, no whitespace, ASCII-escaped, floats in
        Python's shortest round-trip representation. Changing any of these
        changes every identifier in existence.
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
