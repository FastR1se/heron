"""What one side of a link needs from the machine it runs on (ADR-0025).

A robot package is two sides of a link that run in different processes and often
on different machines: the adapter wherever a robot is driven from, the twin
wherever a world is started. Each side states what its process has to find --
and, since ADR-0022, what it must not find: a CycloneDDS 0.11 participant brings
down a 0.10.2 robot by being discovered.

A requirement is a value, declared next to the code that needs it, so that it
can be derived from what that code uses instead of written down a second time.
Checking it belongs to the validator; this module only says what can be asked.

Four kinds, and the registry is closed like the one of ADR-0008: each kind is a
question with one honest way to answer it on a machine. Adding a kind is a
decision, not an edit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from packaging.specifiers import InvalidSpecifier, SpecifierSet


class RequirementKind(StrEnum):
    """What a requirement names, and so where a machine is asked about it."""

    PYTHON = "python"
    """A top-level module importable by the process; the version is that of the
    distribution providing it."""

    ROS2 = "ros2"
    """A package in the ament index of the sourced environment; the version is the
    one its ``package.xml`` declares."""

    EXECUTABLE = "executable"
    """A program found on ``PATH``."""

    VARIABLE = "variable"
    """An environment variable set to a non-empty value."""


VERSIONED: Final = frozenset({RequirementKind.PYTHON, RequirementKind.ROS2})
"""Kinds with a version a machine can read. A program on ``PATH`` has no uniform
way to say which version it is, and a guess from its output would be a check
that passes on the machine it was written on."""

_NAMES: Final = {
    RequirementKind.PYTHON: re.compile(r"[A-Za-z_][A-Za-z0-9_]*"),
    RequirementKind.ROS2: re.compile(r"[a-z][a-z0-9_]*"),
    RequirementKind.EXECUTABLE: re.compile(r"[A-Za-z0-9_.+-]+"),
    RequirementKind.VARIABLE: re.compile(r"[A-Z_][A-Z0-9_]*"),
}
"""Names per kind. A Python name is top-level only: asking for a submodule
imports its parents, and a check must not run the code it checks."""


class RequirementError(ValueError):
    """A requirement that cannot be asked of any machine."""


@dataclass(frozen=True, slots=True)
class Requirement:
    """One thing a side of a link needs, or must not have, in its environment."""

    kind: RequirementKind
    name: str
    hint: str
    """What a human does about it: how to install what is needed, how to get rid
    of what is forbidden. Addressed to a human, so Russian."""

    version: str | None = None
    """PEP 440 specifiers, ``==0.10.2`` or ``>=0.11``. For a need, the version
    found must satisfy them; for a prohibition, only such versions are forbidden."""

    forbidden: bool = False
    because: str | None = None
    """Why. Mandatory for a prohibition: a refusal nobody can explain gets removed."""

    def __post_init__(self) -> None:
        kind = RequirementKind(self.kind)
        if not _NAMES[kind].fullmatch(self.name):
            raise RequirementError(f"{kind}: имя {self.name!r} не годится для этого вида")
        if not self.hint.strip():
            raise RequirementError(f"{kind} {self.name}: нет подсказки, что с этим делать")
        if self.version is not None:
            if kind not in VERSIONED:
                raise RequirementError(
                    f"{kind} {self.name}: версию этого вида машина не сообщает; "
                    f"условие на версию есть только у {', '.join(sorted(VERSIONED))}"
                )
            if not self.version.strip():
                raise RequirementError(f"{kind} {self.name}: пустое условие на версию")
            try:
                SpecifierSet(self.version)
            except InvalidSpecifier as exc:
                raise RequirementError(
                    f"{kind} {self.name}: условие {self.version!r} не по PEP 440"
                ) from exc
        if self.forbidden and not (self.because or "").strip():
            raise RequirementError(f"{kind} {self.name}: запрет без причины")

    @property
    def specifiers(self) -> SpecifierSet | None:
        return SpecifierSet(self.version) if self.version is not None else None

    def admits(self, version: str) -> bool:
        """Whether a version falls under the condition. No condition admits any."""
        specifiers = self.specifiers
        return specifiers is None or specifiers.contains(version, prereleases=True)

    def describe(self) -> str:
        """``python cyclonedds ==0.10.2``: the requirement as a human reads it."""
        return " ".join(part for part in (str(self.kind), self.name, self.version) if part)


def needed(requirements: Iterable[object], kind: RequirementKind) -> frozenset[str]:
    """Names of what is needed, of one kind. Prohibitions and non-requirements are not."""
    return frozenset(
        item.name
        for item in requirements
        if isinstance(item, Requirement) and item.kind is kind and not item.forbidden
    )


def needs(
    kind: RequirementKind,
    name: str,
    *,
    hint: str,
    version: str | None = None,
    because: str | None = None,
) -> Requirement:
    """Present, and in a version the condition admits."""
    return Requirement(kind=kind, name=name, hint=hint, version=version, because=because)


def forbids(
    kind: RequirementKind,
    name: str,
    *,
    because: str,
    hint: str,
    version: str | None = None,
) -> Requirement:
    """Absent, or at least not in a version the condition names."""
    return Requirement(
        kind=kind, name=name, hint=hint, version=version, forbidden=True, because=because
    )
