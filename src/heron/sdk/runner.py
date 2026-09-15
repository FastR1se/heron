"""What a running skill sees, and whether it may run at all.

The loop itself lives in ``executor`` -- this module holds the two things the
loop needs before it starts: the context handed to a skill, and the check that
says whether this skill may drive this robot through this link.

The check is separate from the loop on purpose. Invariant 6: a skill the robot
cannot run is a validator error, and an error the validator can state before a
run must not be something a run discovers halfway through a motion. Keeping it
callable on its own is what lets ``skill check`` answer the same question
without touching a robot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.capabilities import Capability
from ..core.enums import ControlMode
from ..core.models import RobotProfile, SkillSpec
from ..core.protocols import ControlAdapter, Skill
from ..core.types import Command
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
    sent: list[Command] = field(default_factory=list)
    """Everything the skill emitted, in order.

    Kept because a run is worth nothing if what was commanded during it cannot
    be looked at afterwards, and because the adapter is not obliged to remember:
    ``DummyAdapter`` does, a link to a real robot does not."""

    current_time: float = 0.0
    """Clock of the world, advanced by the executor from the state it read."""

    on_sent: Callable[[Command], None] | None = None
    """Told of every command the adapter accepted, after it did (ADR-0038)."""

    @property
    def profile(self) -> RobotProfile:
        return self.profile_

    def send(self, cmd: Command) -> None:
        self.adapter.send_command(cmd)
        self.sent.append(cmd)
        if self.on_sent is not None:
            self.on_sent(cmd)

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
