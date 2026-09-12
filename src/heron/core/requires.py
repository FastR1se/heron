"""Quantitative skill requirements: the ``requires`` block of ADR-0001.

A capability says *what kind* of control a robot offers; it never says *how
much*. The amounts live in the profile's ``contract`` and ``limits``, and a
skill states its thresholds against them here::

    requires: {dof: ">=6", command_rate_hz: ">=50", gripper_width_m: ">=0.06"}

Two restrictions are deliberate and must not be relaxed:

* The language is three comparisons -- ``>=``, ``<=``, ``==``. ADR-0001 forbids
  growing it into expressions: the moment logic appears here, it is code inside
  data and invariant 5 is gone.
* The quantity names are a closed registry, and each one resolves from exactly
  one place in the profile. That is the same rule as everywhere else -- a
  quantity is described once, where it is true.

A quantity the profile does not carry (gripper width on a robot without a
gripper) fails the requirement. It is not skipped: "unknown" and "satisfied"
must not look alike to a validator whose whole purpose is catching silent
mismatch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .models import RobotProfile

_CLAUSE_RE: Final = re.compile(r"^\s*(>=|<=|==)\s*(-?\d+(?:\.\d+)?)\s*$")

OPERATORS: Final[dict[str, Callable[[float, float], bool]]] = {
    ">=": lambda actual, wanted: actual >= wanted,
    "<=": lambda actual, wanted: actual <= wanted,
    "==": lambda actual, wanted: actual == wanted,
}


class RequirementError(ValueError):
    """A requirement clause is malformed or names an unknown quantity."""


def _dof(profile: RobotProfile) -> float | None:
    return float(len(profile.contract.joint_order))


def _command_rate(profile: RobotProfile) -> float | None:
    return profile.contract.command_rate_hz


def _state_rate(profile: RobotProfile) -> float | None:
    return profile.contract.state_rate_hz


def _cartesian_velocity_max(profile: RobotProfile) -> float | None:
    return profile.limits.cartesian_velocity_max


def _gripper_width(profile: RobotProfile) -> float | None:
    return profile.limits.gripper_width_max


QUANTITIES: Final[dict[str, Callable[[RobotProfile], float | None]]] = {
    "dof": _dof,
    "command_rate_hz": _command_rate,
    "state_rate_hz": _state_rate,
    "cartesian_velocity_max": _cartesian_velocity_max,
    "gripper_width_m": _gripper_width,
}
"""Closed registry of comparable quantities, each with a single source."""

SOURCES: Final[dict[str, str]] = {
    "dof": "contract.joint_order (длина)",
    "command_rate_hz": "contract.command_rate_hz",
    "state_rate_hz": "contract.state_rate_hz",
    "cartesian_velocity_max": "limits.cartesian_velocity_max",
    "gripper_width_m": "limits.gripper_width_max",
}
"""Where each quantity is read from, for use in validator messages."""


@dataclass(frozen=True, slots=True)
class Clause:
    """One parsed comparison, e.g. ``>=6``."""

    operator: str
    value: float

    def holds_for(self, actual: float) -> bool:
        return OPERATORS[self.operator](actual, self.value)

    def __str__(self) -> str:
        return f"{self.operator}{self.value:g}"


def parse_clause(text: str) -> Clause:
    """Parse ``">=6"`` into a clause, rejecting anything richer."""
    if not isinstance(text, str):
        raise RequirementError(f"требование должно быть строкой вида \">=6\", получено {text!r}")
    match = _CLAUSE_RE.match(text)
    if match is None:
        raise RequirementError(
            f"{text!r}: требование — сравнение с числом, оператор один из "
            f"{', '.join(OPERATORS)}. Выражения запрещены (ADR-0001)"
        )
    return Clause(operator=match.group(1), value=float(match.group(2)))


class Requirements(dict[str, Clause]):
    """Quantity name -> clause, with both sides validated on construction."""

    __slots__ = ()

    def __init__(self, raw: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for name, text in (raw or {}).items():
            if name not in QUANTITIES:
                raise RequirementError(
                    f"{name!r}: неизвестная величина. Реестр закрыт, доступны: "
                    f"{', '.join(sorted(QUANTITIES))} (ADR-0001)"
                )
            self[name] = text if isinstance(text, Clause) else parse_clause(text)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda req: {name: str(clause) for name, clause in req.items()},
            ),
        )


@dataclass(frozen=True, slots=True)
class Unmet:
    """A requirement the profile does not satisfy."""

    quantity: str
    clause: Clause
    actual: float | None
    """``None`` when the profile does not carry the quantity at all."""

    @property
    def source(self) -> str:
        return SOURCES[self.quantity]

    def describe(self) -> str:
        if self.actual is None:
            return (
                f"{self.quantity} {self.clause}: профиль величину не объявляет "
                f"(источник — {self.source})"
            )
        return f"{self.quantity} {self.clause}: в профиле {self.actual:g} ({self.source})"


def check(requirements: Requirements, profile: RobotProfile) -> list[Unmet]:
    """Every clause the profile fails, in registry order."""
    unmet: list[Unmet] = []
    for name, clause in requirements.items():
        actual = QUANTITIES[name](profile)
        if actual is None or not clause.holds_for(actual):
            unmet.append(Unmet(quantity=name, clause=clause, actual=actual))
    return unmet
