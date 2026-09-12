"""Running one skill against one adapter.

This is the smallest thing that deserves the name "executor": read state, tick
the skill, let it send commands, stop on a terminal status. The real executor
(phase 3) adds task sequencing and a watchdog; the loop shape stays this one.

The compatibility check happens before the first tick, not during it. Invariant
6: a skill the robot cannot run is a validator error, not a failure discovered
halfway through a motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.capabilities import Capability
from ..core.enums import ControlMode, Status
from ..core.models import RobotProfile, SkillSpec
from ..core.protocols import ControlAdapter, Skill
from ..core.types import Command, RobotState
from ..validator.report import Report, error
from ..validator.rules import check_skill_against_profile


class IncompatibleSkillError(RuntimeError):
    """A skill was started on a robot that cannot run it."""

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__("\n".join(str(f) for f in report.errors))


@dataclass
class AdapterContext:
    """The context a running skill sees: its robot, a clock and a way to act."""

    profile_: RobotProfile
    adapter: ControlAdapter
    log_lines: list[str] = field(default_factory=list)
    current_time: float = 0.0
    """Clock of the world, advanced by the runner from the state it read."""

    @property
    def profile(self) -> RobotProfile:
        return self.profile_

    def send(self, cmd: Command) -> None:
        self.adapter.send_command(cmd)

    def now(self) -> float:
        return self.current_time

    def log(self, message: str, **fields: Any) -> None:
        suffix = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        self.log_lines.append(f"{message} {suffix}".strip())


def _spec_of(skill: Skill) -> SkillSpec | None:
    spec = getattr(skill, "spec", None)
    return spec if isinstance(spec, SkillSpec) else None


def check_runnable(skill: Skill, profile: RobotProfile, adapter: ControlAdapter) -> Report:
    """Whether this skill may run on this robot through this link.

    Two comparisons, not one: the profile states what the robot is, the adapter
    states what the link to it can currently do, and they are allowed to differ
    -- a simulated twin may offer less than the real machine.
    """
    spec = _spec_of(skill)
    if spec is None:
        return Report(
            [
                error(
                    "skill.undeclared",
                    "skill:<без объявления>",
                    "у навыка нет spec: SkillSpec",
                    hint="без объявления совместимость не проверяема (инвариант 6)",
                )
            ]
        )

    report = check_skill_against_profile(spec, profile)
    where = f"skill:{spec.name} -> adapter"

    link: frozenset[Capability] = adapter.capabilities()
    absent = spec.required_capabilities - link
    if absent:
        report.add(
            error(
                "adapter.capabilities_missing",
                where,
                f"связь не предоставляет: {', '.join(sorted(absent))}",
                hint="адаптер объявляет возможности сам; профиль их не навязывает",
                adr="ADR-0001",
                detail={"missing": sorted(absent)},
            )
        )

    if (
        spec.required_control_mode is ControlMode.STREAMING
        and adapter.control_mode() is ControlMode.COMMAND
    ):
        report.add(
            error(
                "adapter.control_mode_incompatible",
                where,
                "навык требует потокового управления, связь принимает команды целиком",
                adr="раздел 3",
            )
        )
    return report


def run_skill(
    skill: Skill,
    adapter: ControlAdapter,
    profile: RobotProfile,
    goal: dict[str, Any] | None = None,
    *,
    max_ticks: int = 10_000,
) -> tuple[Status, list[RobotState]]:
    """Run a skill to a terminal status, returning it and the states seen."""
    report = check_runnable(skill, profile, adapter)
    if not report.ok:
        raise IncompatibleSkillError(report)

    ctx = AdapterContext(profile_=profile, adapter=adapter)
    skill.start(dict(goal or {}), ctx)

    seen: list[RobotState] = []
    for _ in range(max_ticks):
        state = adapter.read_state()
        seen.append(state)
        ctx.current_time = state.timestamp
        status = skill.tick(state)
        if status is not Status.RUNNING:
            return status, seen

    skill.cancel()
    return Status.FAILURE, seen
