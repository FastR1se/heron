"""The plan of a mission run: a file both sides read (ADR-P8D).

Everything the coordinator decided before the run -- which robot does which task,
through which link, in which world environment, where to report -- is written
here before any robot process starts. A robot process reads its part of it and
nothing else, so a robot started by hand in another terminal, or later in a
container, needs exactly one argument besides its own name.

The plan is named by its content (``plan_id``, the canonical form of ADR-0006
hashed): a robot introduces itself with it, so a process from another run cannot
join this one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.contract import canonical_dumps, digest
from ..core.loading import LoadError, SchemaProblem
from ..core.models import RobotProfile, Scenario, sim_target
from .mission import Mission

PLAN_FILE: Final = "plan.json"
EVENTS_FILE: Final = "events.jsonl"
RESULT_FILE: Final = "result.json"
LOOPBACK: Final = "127.0.0.1"


class Address(BaseModel):
    """Where the coordinator listens for robots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str
    port: int = Field(gt=0, lt=65536)


class PlannedRobot(BaseModel):
    """A robot that has tasks, and the link it is driven through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    profile: str
    target: str
    plugin: str
    contract_id: str
    """What the profile was when the plan was made; the robot refuses a changed one."""


class PlannedTask(BaseModel):
    """A task with its robot decided and its wait resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    skill: str
    goal: dict[str, Any] = Field(default_factory=dict)
    robot: str
    after: tuple[str, ...] = ()


class MissionPlan(BaseModel):
    """What a mission run is, before it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    mission: str
    scenario: str
    scenario_path: str
    """Absolute, so that a robot process started elsewhere finds the same file."""

    registry: tuple[str, ...]
    store: str | None = None
    backend: str
    target: str
    environment: dict[str, str] = Field(default_factory=dict)
    """The world's, derived once (ADR-0039); every robot process applies it."""

    record: bool = False
    address: Address | None = None
    robots: tuple[PlannedRobot, ...]
    tasks: tuple[PlannedTask, ...]
    started_utc: str

    @property
    def plan_id(self) -> str:
        return digest(canonical_dumps(self.model_dump(mode="json")))

    def robot(self, robot_id: str) -> PlannedRobot | None:
        return next((item for item in self.robots if item.id == robot_id), None)

    def tasks_of(self, robot_id: str) -> tuple[PlannedTask, ...]:
        return tuple(task for task in self.tasks if task.robot == robot_id)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / PLAN_FILE
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, directory: Path) -> MissionPlan:
        path = Path(directory) / PLAN_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LoadError(path, [SchemaProblem("<файл>", f"план не читается: {exc}")]) from exc
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            problems = [
                SchemaProblem(".".join(str(p) for p in item["loc"]) or "<корень>", item["msg"])
                for item in exc.errors()
            ]
            raise LoadError(path, problems) from exc


def make_plan(
    mission: Mission,
    scenario: Scenario,
    assignment: Mapping[str, str],
    profiles: Mapping[str, RobotProfile],
    *,
    scenario_path: Path,
    registry_roots: tuple[Path, ...],
    store: Path | None,
    environment: Mapping[str, str],
    record: bool,
) -> MissionPlan:
    """The plan of a mission whose every task is assigned (``check_mission`` passed).

    Only robots that have a task get a process; a driven robot the mission does not
    need is not connected at all.
    """
    target = sim_target(scenario.world.backend)
    tasks = tuple(
        PlannedTask(
            id=task.id,
            skill=task.skill,
            goal=dict(task.goal),
            robot=assignment[task.id],
            after=mission.waits_for(index),
        )
        for index, task in enumerate(mission.tasks)
    )
    used = {task.robot for task in tasks}
    robots = tuple(
        PlannedRobot(
            id=robot.id,
            profile=robot.profile,
            target=target,
            plugin=profiles[robot.id].adapters[target].plugin,
            contract_id=profiles[robot.id].contract.contract_id,
        )
        for robot in scenario.robots
        if robot.id in used
    )
    return MissionPlan(
        mission=mission.name,
        scenario=scenario.name,
        scenario_path=str(Path(scenario_path).resolve()),
        registry=tuple(str(Path(root).resolve()) for root in registry_roots),
        store=str(Path(store).resolve()) if store is not None else None,
        backend=scenario.world.backend,
        target=target,
        environment=dict(environment),
        record=record,
        robots=robots,
        tasks=tasks,
        started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
