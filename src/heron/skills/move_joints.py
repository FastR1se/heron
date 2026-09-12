"""``move_joints`` -- drive the joints to a declared pose along a timed ramp.

The first skill that moves anything, and the smallest one that has to answer a
question ``hold`` could avoid: what to send between here and there.

A group position controller forwards whatever it receives straight into the
joint's command interface. Sending the target once therefore asks the robot to
be somewhere else immediately, and the simulator obliges the only way it can --
by accelerating the joint as hard as the model allows. The motion that comes out
is not the motion anyone asked for, it depends on gains rather than on the goal,
and it will not survive a move to hardware. So the ramp is the skill's job, and
it is done here rather than hidden in the adapter: the adapter's business is one
command, and interpolation is a decision about the motion.

Ramping in joint space and not in Cartesian space is likewise deliberate. A
straight line for the tool would need inverse kinematics, which is a solver this
platform does not have and would have to invent per robot -- the very thing
ADR-0009 refuses. The goal is stated in the units of the contract and travels in
them.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.capabilities import Capability
from ..core.enums import ActionType, ControlMode, Status
from ..core.models import SkillSpec
from ..core.requires import Requirements
from ..core.types import Command, RobotState
from .base import BaseSkill

DEFAULT_DURATION_S = 2.0
MIN_DURATION_S = 1e-3
"""Below this a ramp is a step, and a step is what this skill exists to avoid."""


class MoveJointsSkill(BaseSkill):
    """Move to ``target`` over ``duration`` seconds, then report success.

    Goal fields:

    ``target``    joint positions, ordered like ``contract.joint_order``, in the
                  contract's angle unit. Required.
    ``duration``  seconds the motion takes (default 2.0).
    """

    spec = SkillSpec(
        name="move_joints",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.STREAMING,
        requires=Requirements({"dof": ">=1"}),
    )

    def __init__(self) -> None:
        super().__init__()
        self.target: tuple[float, ...] = ()
        self.duration = DEFAULT_DURATION_S
        self.start_pose: tuple[float, ...] = ()
        self.started_at: float | None = None

    def on_start(self) -> None:
        """Check the goal against the robot before a single command is sent.

        A target of the wrong width or outside the envelope is a mistake in the
        mission, and invariant 6 says a mistake that can be seen before a run
        must not be discovered halfway through a motion. The adapter would
        refuse the command too -- but by then the robot is already moving, and
        it stops wherever the refusal caught it.
        """
        profile = self.context.profile
        contract = profile.contract

        target = self.goal.get("target")
        if target is None:
            raise ValueError(
                f"{self.spec.name}: в цели нет поля target — "
                f"вектор позиций суставов в порядке contract.joint_order"
            )
        self.target = _floats(target)
        if len(self.target) != contract.dof:
            raise ValueError(
                f"{self.spec.name}: target на {len(self.target)} суставов при "
                f"{contract.dof} в contract.joint_order "
                f"({', '.join(contract.joint_order)})"
            )

        bounds = profile.limits.joint_position
        if bounds is not None:
            outside = [
                contract.joint_order[i]
                for i, value in enumerate(self.target)
                if not (bounds.min[i] <= value <= bounds.max[i])
            ]
            if outside:
                raise ValueError(
                    f"{self.spec.name}: target вне лимитов профиля в суставах "
                    f"{', '.join(outside)}; единицы цели — {contract.angle_unit}, "
                    f"как в контракте"
                )

        self.duration = float(self.goal.get("duration", DEFAULT_DURATION_S))
        if self.duration < MIN_DURATION_S:
            raise ValueError(
                f"{self.spec.name}: duration {self.duration:g} с — это не разгон, "
                f"а ступень; минимум {MIN_DURATION_S:g} с"
            )
        self.start_pose = ()
        self.started_at = None

    def on_tick(self, state: RobotState) -> Status:
        if len(state.joint_position) != len(self.target):
            return Status.FAILURE

        if self.started_at is None:
            # The first reading defines where "from" is. Taken from the robot
            # and not from the previous command, so that a skill starting after
            # something else moved the arm ramps from where the arm actually is.
            self.start_pose = state.joint_position
            self.started_at = state.timestamp

        elapsed = state.timestamp - self.started_at
        fraction = 1.0 if elapsed >= self.duration else max(elapsed / self.duration, 0.0)
        values = tuple(
            begin + (end - begin) * fraction
            for begin, end in zip(self.start_pose, self.target, strict=True)
        )
        self.context.send(Command(type=ActionType.JOINT_POSITION, values=values))

        if fraction >= 1.0:
            self.context.log(
                "движение завершено",
                skill=self.spec.name,
                duration=f"{elapsed:.3f}",
            )
            return Status.SUCCESS
        return Status.RUNNING


def _floats(value: object) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"target: ожидался список чисел, получено {type(value).__name__}")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target: в списке не число ({exc})") from None
