"""Reach a joint pose with the example arm: the training task of phase 6.

Chosen for the pipeline and not for the difficulty of learning it, and that is
said openly (ADR-0040, item 4): what phase 6 has to prove is that a policy
trained in one world crosses into another as files and runs there through the
same executor as a coded skill. A task whose goal is a joint pose is the one for
which "did it reach the goal" has an answer that does not depend on how well it
was trained, and the one whose goal is the goal of ``move_joints`` -- so that
swapping the coded skill for the trained one is a line in a profile and not an
edit to a scenario (ADR-0043).

What this file says, and what it deliberately cannot:

* it says what is rewarded, when an episode ends, how a reset scatters the robot,
  and the box a goal is drawn from;
* it cannot say what the policy emits, what it observes, at what rate, or through
  what drive. Those come from the robot's contract, and the protocol has no member
  through which they could be changed (ADR-0044). That is what makes the claim
  "the same representation in both worlds" checkable rather than hopeful.

Isaac Lab is imported inside the methods, never at module level: the platform
reads ``backend``, ``spec`` and ``completion`` on a machine that has no simulator
-- a dry run, a validator, a test on Windows -- and an import at the top would
make every one of those impossible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from heron.core.capabilities import Capability
from heron.core.enums import ControlMode
from heron.core.models import Completion, SkillSpec
from heron.train.isaaclab_task import BACKEND, JointBox, TaskContext

GOAL_SPREAD = 0.5
"""How far from the reference pose a goal may be drawn, per joint, in radians.

Inside the set of targets the action can command (``origin +- clip * scale``,
which for this arm is the reference pose +- 1.0), with margin: a goal the mapped
action cannot reach is one the policy is punished for missing however well it
learns. The platform checks the containment (``train.goal_unreachable``) rather
than trusting this comment."""

RESET_SPREAD = 0.9
"""How far a reset scatters the joints around the reference pose.

Wide enough on purpose: the world this policy will be executed in starts the arm
at zeros, and the reference pose is a working pose, so the pose of the first tick
of a run has to be inside what training saw. The largest gap between the two is
the ``joint2`` element of the reference pose, 0.8 rad."""


class Arm6ReachTask:
    """Bring the joints to a commanded pose and stay there."""

    backend = BACKEND

    spec = SkillSpec(
        name="arm6_reach",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.STREAMING,
        # Written as data, as a profile writes it: the closed registry of
        # quantities parses it (ADR-0008).
        requires={"dof": ">=6"},
    )

    # The tolerance is in radians, the contract's ``angle_unit``. The same number
    # scores success here and ends the skill in a world of execution: one quantity
    # in two worlds, which is the whole reason it is declared by the task and not
    # by the mission (ADR-0043). The horizon is the length of an episode -- past it
    # the policy was never trained -- and matches the four seconds the coded skill
    # is given in the scenario it replaces.
    completion = Completion(tolerance=0.05, hold_s=0.2, horizon_s=4.0)

    # -- what the goal is ---------------------------------------------------

    def goal_range(self, ctx: TaskContext) -> JointBox:
        """A box around the contract's reference pose, in ``joint_order``."""
        pose = ctx.contract.default_pose or [0.0] * ctx.contract.dof
        return JointBox(
            min=[value - GOAL_SPREAD for value in pose],
            max=[value + GOAL_SPREAD for value in pose],
        )

    # -- what is rewarded ---------------------------------------------------

    def rewards(self, ctx: TaskContext) -> object:
        """Track the goal, and pay for the way there.

        Two tracking terms, as a reach task has: one that is never satisfied and
        keeps pulling, and one that is sharp near the goal and makes the last
        centimetre worth reaching. The penalties are what keeps a policy that
        reaches the pose from arriving at it shaking.
        """
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg
        from isaaclab.utils import configclass

        from .rewards import joint_pose_error, joint_pose_error_tanh

        params = {"asset_cfg": ctx.robot, "command_name": ctx.command}

        @configclass
        class Rewards:
            tracking = RewardTermCfg(func=joint_pose_error, weight=-1.0, params=params)
            tracking_fine = RewardTermCfg(
                func=joint_pose_error_tanh,
                weight=0.5,
                params={**params, "std": 0.1},
            )
            joint_velocity = RewardTermCfg(
                func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": ctx.robot}
            )
            action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3)

        return Rewards()

    def terminations(self, ctx: TaskContext) -> object:
        """Only the horizon. An episode that ended early would teach the policy that
        reaching the goal is where the world stops, and a run does not stop there."""
        from isaaclab.envs import mdp
        from isaaclab.managers import TerminationTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

        return Terminations()

    def events(self, ctx: TaskContext) -> object:
        """Scatter the joints around the reference pose on every reset."""
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Events:
            reset_joints = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-RESET_SPREAD, RESET_SPREAD),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": ctx.robot,
                },
            )

        return Events()

    def agent(self, ctx: TaskContext) -> Mapping[str, Any] | None:
        """The backend's own agent. A 64x64 network is what the vendor's reach task
        learns with, and this task is no harder."""
        return None
