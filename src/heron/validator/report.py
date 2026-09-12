"""Findings and reports.

A finding is a structured object, not a formatted line. Three reasons, all of
them load-bearing:

* Experiment 4 of section 13 measures the share of deliberately introduced
  contract errors the validator catches before a run. Counting by ``code`` is
  possible; counting by grepping prose is not.
* The same finding has to reach a human in the terminal and a machine in CI.
* Codes are a promise: a rule's identifier outlives its wording.

Messages address a human and are therefore Russian; codes, identifiers and this
docstring are English, as CLAUDE.md requires.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..core.enums import Severity


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing the validator has to say."""

    code: str
    """Stable identifier, ``<область>.<правило>``. Wording may change, this does not."""

    severity: Severity
    message: str
    where: str
    """What the finding is about: ``profile:<name>.capabilities`` and the like."""

    hint: str | None = None
    """What a human is expected to do about it."""

    adr: str | None = None
    """Decision the rule enforces, when there is one."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Machine-readable specifics: differing fields, actual values, names."""

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.where}: {self.message}"]
        if self.hint:
            parts.append(f"    {self.hint}")
        if self.adr:
            parts.append(f"    решение: {self.adr}")
        return "\n".join(parts)


class Report:
    """Everything the validator found in one run."""

    def __init__(self, findings: Iterable[Finding] = ()) -> None:
        self.findings: list[Finding] = list(findings)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: Iterable[Finding]) -> None:
        self.findings.extend(findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """A run may start only when nothing is an error (invariant 6)."""
        return not self.errors

    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    def merged(self, other: Report) -> Report:
        return Report([*self.findings, *other.findings])


def error(code: str, where: str, message: str, **kwargs: Any) -> Finding:
    return Finding(code=code, severity=Severity.ERROR, where=where, message=message, **kwargs)


def warning(code: str, where: str, message: str, **kwargs: Any) -> Finding:
    return Finding(code=code, severity=Severity.WARNING, where=where, message=message, **kwargs)
