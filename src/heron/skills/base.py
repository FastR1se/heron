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

from typing import Any

from ..core.capabilities import Capability
from ..core.enums import ControlMode, Status
from ..core.models import SkillSpec
from ..core.protocols import SkillContext
from ..core.requires import Requirements
from ..core.types import RobotState


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
