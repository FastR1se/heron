"""``hold`` -- keep the joint positions the robot had when the skill started.

The smallest skill that is still a real one: it declares capabilities, holds
state between ticks, emits commands and terminates. It exists so that the chain
profile -> validator -> adapter -> skill can be exercised end to end before any
simulator is involved, which is what phase 1 has to demonstrate.
"""

from __future__ import annotations

from ..core.capabilities import Capability
from ..core.enums import ActionType, ControlMode, Status
from ..core.models import SkillSpec
from ..core.requires import Requirements
from ..core.types import Command, RobotState
from .base import BaseSkill

DEFAULT_DURATION_S = 1.0


class HoldSkill(BaseSkill):
    """Hold the starting pose for ``duration`` seconds.

    Goal fields: ``duration`` (seconds, default 1.0).
    """

    spec = SkillSpec(
        name="hold",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.STREAMING,
        requires=Requirements({"dof": ">=1"}),
    )

    def __init__(self) -> None:
        super().__init__()
        self.target: tuple[float, ...] = ()
        self.started_at: float | None = None

    def on_start(self) -> None:
        self.target = ()
        self.started_at = None

    def on_tick(self, state: RobotState) -> Status:
        if not state.joint_position:
            return Status.FAILURE

        if self.started_at is None:
            # First tick defines what "hold" means for this run: the pose the
            # robot is in, moved inside the envelope if it rests past a limit
            # (ADR-0032).
            self.target = self.admissible_start(state.joint_position)
            self.started_at = state.timestamp

        self.context.send(Command(type=ActionType.JOINT_POSITION, values=self.target))

        duration = float(self.goal.get("duration", DEFAULT_DURATION_S))
        if state.timestamp - self.started_at >= duration:
            return Status.SUCCESS
        return Status.RUNNING
