"""A mission: what a system of robots is to do, as data (ADR-P8A).

A scenario says what the system is made of; a mission says what it does. The two
are separate documents on purpose: a mission names no robot by default, only what
each of its tasks needs, so the same mission is checked against a complex in
Gazebo and against the same composition in a world of numbers.

The schema is strict like every other one. A task's goal is its own mapping rather
than keys mixed in among the task's fields, as the scenario's single ``task``
allows: with five service keys next to it, a misspelt one would otherwise become a
goal field nobody reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.capabilities import Capability
from ..core.loading import load_model
from ..core.models import NAME_RE
from ..core.requires import Requirements


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError(f"{value!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
    return value


def _check_optional_name(value: str | None) -> str | None:
    return value if value is None else _check_name(value)


class MissionTask(BaseModel):
    """One piece of work: a skill, its goal, and what the goal needs of a robot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    skill: str
    """A skill's name. How it is made -- code or policy -- is the profile's line (invariant 4)."""

    goal: dict[str, Any] = Field(default_factory=dict)
    """Passed to the skill untouched; the core does not read it (section 7)."""

    required_capabilities: frozenset[Capability] = frozenset()
    """What the task needs of a robot beyond what its skill declares (ADR-P8A)."""

    requires: Requirements = Field(default_factory=Requirements)
    """Quantities the task needs beyond the skill's, in the closed registry (ADR-0008)."""

    after: tuple[str, ...] | None = None
    """Tasks written above that must succeed first. Unset: the previous task;
    ``[]``: none (ADR-P8C)."""

    robot: str | None = None
    """Narrows the candidates to one robot of the scenario, still checked (ADR-P8B)."""

    _validate_id = field_validator("id", "skill")(_check_name)
    _validate_robot = field_validator("robot")(_check_optional_name)

    @field_validator("after")
    @classmethod
    def _names(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        for item in value:
            _check_name(item)
        if len(set(value)) != len(value):
            raise ValueError(f"after: повтор имён в {list(value)}")
        return value


class Mission(BaseModel):
    """What a system does: tasks in the order they are written (ADR-P8A, ADR-P8C)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str
    tasks: tuple[MissionTask, ...] = Field(min_length=1)

    _validate_name = field_validator("name")(_check_name)

    @model_validator(mode="after")
    def _unique_ids(self) -> Mission:
        ids = [task.id for task in self.tasks]
        duplicates = sorted({tid for tid in ids if ids.count(tid) > 1})
        if duplicates:
            raise ValueError(f"повтор идентификаторов задач: {', '.join(duplicates)}")
        return self

    def waits_for(self, index: int) -> tuple[str, ...]:
        """The tasks the one at ``index`` waits for, the default resolved (ADR-P8C)."""
        task = self.tasks[index]
        if task.after is not None:
            return task.after
        return (self.tasks[index - 1].id,) if index > 0 else ()


def load_mission(path: Path) -> Mission:
    """Load and schema-check a mission document."""
    return load_model(Path(path), Mission)
