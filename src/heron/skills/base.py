"""Base class for skills written as code.

Implementing the ``Skill`` protocol by hand each time would spread the same four
lines of bookkeeping over every skill, so this class holds them. It is a
convenience, not a requirement: the protocol is what the platform checks, and a
skill that satisfies it without inheriting anything is equally valid.

What this class deliberately does not do is interpret the goal. A goal is passed
to the skill untouched (``Task`` in the scenario schema): the core does not know
what ``target: random_in_box`` means, and inventing a vocabulary for it here
would put mission knowledge into the platform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.capabilities import Capability
from ..core.enums import ControlMode, Status
from ..core.models import RobotProfile, SkillSpec
from ..core.protocols import SkillContext
from ..core.requires import Requirements
from ..core.types import RobotState


def goal_target(skill: str, goal: Mapping[str, Any], profile: RobotProfile) -> tuple[float, ...]:
    """The goal's ``target``, checked against the robot before a single command is sent.

    A target of the wrong width or outside the envelope is a mistake in the mission,
    and invariant 6 says a mistake that can be seen before a run must not be
    discovered halfway through a motion. The adapter would refuse the command too --
    but by then the robot is already moving, and it stops wherever the refusal
    caught it. One check for every skill whose goal is a joint pose, coded or
    trained (ADR-0043).
    """
    contract = profile.contract
    raw = goal.get("target")
    if raw is None:
        raise ValueError(
            f"{skill}: в цели нет поля target — "
            f"вектор позиций суставов в порядке contract.joint_order"
        )
    target = floats(raw)
    if len(target) != contract.dof:
        raise ValueError(
            f"{skill}: target на {len(target)} суставов при "
            f"{contract.dof} в contract.joint_order "
            f"({', '.join(contract.joint_order)})"
        )

    bounds = profile.limits.joint_position
    if bounds is not None:
        outside = [
            contract.joint_order[i]
            for i, value in enumerate(target)
            if not (bounds.min[i] <= value <= bounds.max[i])
        ]
        if outside:
            raise ValueError(
                f"{skill}: target вне лимитов профиля в суставах "
                f"{', '.join(outside)}; единицы цели — {contract.angle_unit}, "
                f"как в контракте"
            )
    return target


def floats(value: object) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"target: ожидался список чисел, получено {type(value).__name__}")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target: в списке не число ({exc})") from None


def admissible(
    profile: RobotProfile, pose: Sequence[float]
) -> tuple[tuple[float, ...], list[str]]:
    """The pose moved onto the profile's position envelope, and which joints moved (ADR-0032).

    A measured pose can lie outside the envelope a command must lie inside: a
    joint resting against a soft stop in a simulator, an encoder reading past a
    soft limit after a stop. A skill that commands the pose it measured then
    commands something the adapter refuses, and never starts. Each joint is taken
    to the nearest bound; a joint inside the envelope is not touched.
    """
    values = tuple(float(value) for value in pose)
    bounds = profile.limits.joint_position
    if bounds is None or len(values) != len(bounds.min):
        return values, []
    projected: list[float] = []
    moved: list[str] = []
    for name, value, low, high in zip(
        profile.contract.joint_order, values, bounds.min, bounds.max, strict=True
    ):
        nearest = min(max(value, low), high)
        if nearest != value:
            moved.append(f"{name} {value:.6g} -> {nearest:.6g}")
        projected.append(nearest)
    return tuple(projected), moved


class BaseSkill:
    """A coded skill with its declaration attached."""

    spec: SkillSpec

    def __init__(self) -> None:
        self.ctx: SkillContext | None = None
        self.goal: dict[str, Any] = {}
        self._cancelled = False

    # -- declaration, read by the validator ---------------------------------

    @property
    def required_capabilities(self) -> frozenset[Capability]:
        return self.spec.required_capabilities

    @property
    def required_control_mode(self) -> ControlMode:
        return self.spec.required_control_mode

    @property
    def requires(self) -> Requirements:
        return self.spec.requires

    # -- lifecycle ----------------------------------------------------------

    def start(self, goal: dict[str, Any], ctx: SkillContext) -> None:
        self.ctx = ctx
        self.goal = dict(goal)
        self._cancelled = False
        self.on_start()

    def tick(self, state: RobotState) -> Status:
        if self._cancelled:
            return Status.FAILURE
        return self.on_tick(state)

    def cancel(self) -> None:
        self._cancelled = True
        self.on_cancel()

    # -- to be implemented by a concrete skill ------------------------------

    def on_start(self) -> None:
        """Called once after the goal and context are in place."""

    def on_tick(self, state: RobotState) -> Status:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_cancel(self) -> None:
        """Called on cancellation, before the next tick reports failure."""

    # -- helpers ------------------------------------------------------------

    @property
    def context(self) -> SkillContext:
        if self.ctx is None:
            raise RuntimeError("навык не запущен: start() не вызывался")
        return self.ctx

    def admissible_start(self, pose: Sequence[float]) -> tuple[float, ...]:
        """Where a motion measured from ``pose`` may start, saying so when it is not ``pose``."""
        projected, moved = admissible(self.context.profile, pose)
        if moved:
            self.context.log(
                "измеренная поза вне рамки limits, движение начинается с ближайшей допустимой",
                skill=self.spec.name,
                joints="; ".join(moved),
            )
        return projected
