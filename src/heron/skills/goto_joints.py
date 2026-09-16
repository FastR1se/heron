"""``goto_joints`` -- one motion to a joint pose, for a robot that plans it itself.

The command class of section 3 (ADR-0049). ``move_joints`` ramps because a group
position controller does whatever it is told at once; a controller of this class
takes a goal, plans the motion within its own bounds and reports when it is over.
Ramping here would queue a point per tick. So the skill sends one command and
from then on only reads.

"Over" is not "arrived". A buffer emptied by a stop halfway looks like one emptied
by the end of the path, so the end of the motion is read from the phase the link
reports and arrival from the pose, and success needs both. The tolerance has no
default: the same number in degrees and in radians differs 57 times, and a default
would be wrong for one of the two.

Speed and acceleration are not goal fields: the adapter takes them from the
profile's envelope, times the run's speed scale (ADR-0052). Nor is the start pose
projected onto the envelope (ADR-0032 does not apply): the controller plans the
path, and only the goal has to lie inside the envelope.
"""

from __future__ import annotations

from ..core.capabilities import Capability
from ..core.enums import ActionType, ControlMode, MotionPhase, Status
from ..core.models import SkillSpec
from ..core.requires import Requirements
from ..core.types import Command, RobotState
from .base import BaseSkill, goal_target


class GotoJointsSkill(BaseSkill):
    """Command ``target`` once, then succeed when the motion is over within ``tolerance``.

    Goal fields:

    ``target``     joint positions, ordered like ``contract.joint_order``, in the
                   contract's angle unit. Required.
    ``tolerance``  largest deviation of any joint from ``target`` at the end of the
                   motion, in the contract's angle unit. Required, positive.
    """

    spec = SkillSpec(
        name="goto_joints",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.COMMAND,
        requires=Requirements({"dof": ">=1"}),
    )

    def __init__(self) -> None:
        super().__init__()
        self.target: tuple[float, ...] = ()
        self.tolerance = 0.0
        self.sent = False

    def on_start(self) -> None:
        profile = self.context.profile
        self.target = goal_target(self.spec.name, self.goal, profile)
        raw = self.goal.get("tolerance")
        if raw is None:
            raise ValueError(
                f"{self.spec.name}: в цели нет поля tolerance — допуск прибытия в единицах "
                f"контракта ({profile.contract.angle_unit}); умолчания нет (ADR-0049)"
            )
        try:
            self.tolerance = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{self.spec.name}: tolerance — число, получено {raw!r}") from None
        if not self.tolerance > 0:
            raise ValueError(f"{self.spec.name}: tolerance {self.tolerance:g} — нужен > 0")
        self.sent = False

    def on_tick(self, state: RobotState) -> Status:
        if len(state.joint_position) != len(self.target):
            return Status.FAILURE
        if state.motion is None:
            return self._fail("связь не сообщает фазу движения; навык командного класса без неё "
                              "не отличит конец движения от его отсутствия")

        if not self.sent:
            if state.motion is not MotionPhase.IDLE:
                return self._fail("робот уже выполняет движение", motion=state.motion)
            self.context.send(Command(type=ActionType.JOINT_POSITION, values=self.target))
            self.sent = True
            return Status.RUNNING

        if state.motion is MotionPhase.MOVING:
            return Status.RUNNING
        if state.motion is MotionPhase.HALTED:
            return self._fail("движение остановлено и не завершено", motion=state.motion)

        deviations = [
            (name, abs(q - goal))
            for name, q, goal in zip(
                self.context.profile.contract.joint_order,
                state.joint_position,
                self.target,
                strict=True,
            )
        ]
        outside = [f"{name} {dev:.6g}" for name, dev in deviations if dev > self.tolerance]
        if outside:
            return self._fail(
                "движение закончено вне допуска",
                tolerance=f"{self.tolerance:g}",
                joints="; ".join(outside),
            )
        self.context.log(
            "движение завершено",
            skill=self.spec.name,
            deviation=f"{max(dev for _, dev in deviations):.6g}",
        )
        return Status.SUCCESS

    def _fail(self, message: str, **fields: object) -> Status:
        self.context.log(message, skill=self.spec.name, **fields)
        return Status.FAILURE
