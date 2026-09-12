"""Capability keys: the only language in which a skill talks about a robot.

ADR-0001. A capability is a discrete identity and never carries parameters:
degrees of freedom, units and ranges already live in ``contract`` and ``limits``,
and duplicating them here would create a second source of truth for exactly the
quantities the platform exists to keep aligned (invariants 5 and 6).

Three rules are enforced in this module, not in the validator, so that a
malformed key fails where it is first read instead of silently producing an
empty intersection later:

* form -- exactly two segments, ``<domain>.<aspect>``, lowercase;
* membership -- the core registry is closed, an unknown key is an error;
* extension -- third-party keys live in a domain prefixed ``x_``.

Matching is set equality on keys, with no inference: a controller that accepts
position commands while being torque-driven declares both keys itself. The
knowledge stays where it is true.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Final

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

SEGMENT = r"[a-z][a-z0-9_]*"
_KEY_RE: Final = re.compile(rf"^{SEGMENT}\.{SEGMENT}$")

EXTENSION_PREFIX: Final = "x_"
"""Domain prefix reserved for capabilities the core registry does not know."""

CORE_CAPABILITIES: Final = frozenset(
    {
        "base.velocity",  # linear and angular velocity command
        "base.pose",      # move to a target pose
        "arm.joint_position",
        "arm.joint_torque",
        "arm.cartesian",  # pose of the end effector
        "gripper.binary",  # open / close
        "gripper.width",
        "whole_body.torque",  # low-level control of the whole body
    }
)
"""Closed registry. Removing a key is a breaking change to a public API."""


class CapabilityError(ValueError):
    """A capability key is malformed or outside the closed registry."""


class Capability(str):
    """A validated capability key.

    Subclasses ``str`` so profiles, sets and YAML round-trips stay ordinary
    strings; construction is what enforces ADR-0001.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> Capability:
        if isinstance(value, Capability):
            return value
        if not isinstance(value, str):
            raise CapabilityError(
                f"возможность должна быть строкой, получено {type(value).__name__}"
            )
        if not _KEY_RE.match(value):
            raise CapabilityError(
                f"{value!r}: возможность — ровно два сегмента <домен>.<аспект> "
                f"в нижнем регистре, например arm.joint_position (ADR-0001)"
            )
        domain = value.split(".", 1)[0]
        if not domain.startswith(EXTENSION_PREFIX) and value not in CORE_CAPABILITIES:
            raise CapabilityError(
                f"{value!r}: ключа нет в реестре ядра. Реестр закрыт; "
                f"возможность стороннего робота объявляется в домене {EXTENSION_PREFIX}*, "
                f"например {EXTENSION_PREFIX}vendor.dance_mode (ADR-0001). "
                f"Известные ключи: {', '.join(sorted(CORE_CAPABILITIES))}"
            )
        return super().__new__(cls, value)

    @property
    def domain(self) -> str:
        return self.split(".", 1)[0]

    @property
    def aspect(self) -> str:
        return self.split(".", 1)[1]

    @property
    def is_extension(self) -> bool:
        """True for a key outside the core registry, declared in an ``x_`` domain."""
        return self.domain.startswith(EXTENSION_PREFIX)

    def __repr__(self) -> str:
        return f"Capability({str.__repr__(self)})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


def parse_set(values: object) -> frozenset[Capability]:
    """Validate an iterable of raw keys into a capability set."""
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise CapabilityError("ожидался список возможностей")
    return frozenset(Capability(v) for v in values)


def missing(
    required: frozenset[Capability], available: frozenset[Capability]
) -> frozenset[Capability]:
    """Keys a skill requires that a robot does not declare.

    Plain set difference. No key implies another one (ADR-0001).
    """
    return required - available
