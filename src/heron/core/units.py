"""Angle units.

The contract declares the unit every angular quantity in the platform is
expressed in (``rad`` or ``deg``), and external description formats declare
nothing -- URDF fixes radians by specification. Converting between the two is
therefore not a formatting detail but the exact place where a unit mismatch
either gets caught or silently passes through, so it happens in one function
that both the validator and the generators call.

Nothing here rounds or clamps. A comparison that needs a tolerance states its
own; a conversion that quietly loses precision would be a second source of
disagreement about a quantity, which is what the platform exists to prevent.
"""

from __future__ import annotations

import math

from .enums import AngleUnit


def to_radians(value: float, unit: AngleUnit) -> float:
    """Express a contract-unit angle in radians, the unit URDF fixes."""
    return math.radians(value) if unit is AngleUnit.DEG else value


def from_radians(value: float, unit: AngleUnit) -> float:
    """Express a radian angle in the unit the contract declares."""
    return math.degrees(value) if unit is AngleUnit.DEG else value


def format_angle(value: float, unit: AngleUnit) -> str:
    """An angle with its unit, for a message a human reads."""
    return f"{value:g} {unit}"
