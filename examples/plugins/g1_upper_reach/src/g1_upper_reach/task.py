"""Bring the mounted G1's upper body to a joint pose: the training task of phase 6, part two.

The task of ADR-0005, item 1, as far as a world of the platform can hold it: a policy of
the humanoid's upper body, with balance held by something else. In a simulator that
something is a mount -- the pelvis bolted to a stand at 1.0 m, the legs hanging without
torque (ADR-0063). That is an ideal balance controller, not the robot's own, and the task is
trained under that condition and says so.

The goal is a pose of the fifteen joints of the contract, like the goal of
``move_joints``, so that swapping the coded skill for the trained one is one line of the
profile (ADR-0043, ADR-0046). What this file decides and what it cannot:

* it says what is rewarded, when an episode ends, how a reset scatters the arms and puts
  the legs back, and the box a goal is drawn from;
* it cannot say what the policy emits, what it observes, at what rate, or through what
  drive: those are the contract's (ADR-0044).

It knows it is written for a G1, which the platform never does: the names of the legs it
resets and which joint is the waist are this package's knowledge, not the core's
(invariant 1).

Isaac Lab is imported inside the methods, never at module level: the platform reads
``backend``, ``spec`` and ``completion`` where there is no simulator -- a dry run, a
validator, a test on Windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from heron.core.capabilities import Capability
from heron.core.enums import ControlMode
from heron.core.models import Completion, SkillSpec
from heron.train.isaaclab_task import BACKEND, JointBox, TaskContext

WAIST_SPREAD: Final = 0.3
"""How far from the reference pose a goal of the waist may be drawn, radians.

Inside the action's targets for the waist (reference +- 0.5) with margin: the torso turns
the whole upper body, and a goal far round it is a different task."""

ARM_SPREAD: Final = 0.5
"""How far from the reference pose a goal of an arm joint may be drawn, radians.

Half of what the action can command there (reference +- 1.0): the rest is room for the
policy to hold against gravity, which a position drive of these gains leaves as an offset
-- at the shoulder up to 0.13 rad, 5.2 N m of an outstretched arm on kp 40. The platform
checks the containment (``train.goal_unreachable``) rather than trusting this comment."""

RESET_SPREAD: Final = 0.8
"""How far a reset scatters the upper body around the reference pose, radians.

The world of execution starts the arms hanging at zeros -- nothing drives them before the
first command -- and the largest element of the reference pose is the elbow's 0.6, so the
pose of the first tick of a run is inside what training saw. A test holds it."""

RESET_SPEED: Final = 0.2
"""Joint speed a reset may start the upper body with, rad/s: the arms of the world of
execution are not quite still when the first command arrives."""

LEGS: Final = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
"""The joints of the description outside the contract. Nothing drives them (ADR-0063), so
nothing but a reset puts them back: without it, legs set swinging by one episode would
still swing in the next."""


class G1UpperReachTask:
    """Bring the waist and both arms to a commanded pose and stay there."""

    backend = BACKEND

    # Written as data, as a profile writes it: the closed registry of quantities parses
    # ``requires`` (ADR-0008). Exactly fifteen: a policy of this width drives this upper
    # body and no other.
    spec = SkillSpec.model_validate(
        {
            "name": "g1_upper_reach",
            "required_capabilities": [Capability("arm.joint_position")],
            "required_control_mode": ControlMode.STREAMING,
            "requires": {"dof": "==15"},
        }
    )

    # Radians, the contract's unit; judged on the worst of fifteen joints (ADR-0043). Looser
    # than the arm's 0.05: the wrists are driven softly (kp 10) because nothing heavier is
    # stable without a rotor's inertia at 500 Hz (ADR-0063, item 5). The horizon is the
    # length of an episode, and one second longer than the coded skill's four.
    completion = Completion(tolerance=0.08, hold_s=0.3, horizon_s=5.0)

    # -- what the goal is ---------------------------------------------------

    def goal_range(self, ctx: TaskContext) -> JointBox:
        """A box around the contract's reference pose, in ``joint_order``."""
        pose = ctx.contract.default_pose or [0.0] * ctx.contract.dof
        spread = [
            WAIST_SPREAD if name.startswith("waist_") else ARM_SPREAD
            for name in ctx.contract.joint_order
        ]
        return JointBox(
            min=[value - width for value, width in zip(pose, spread, strict=True)],
            max=[value + width for value, width in zip(pose, spread, strict=True)],
        )

    # -- what is rewarded ---------------------------------------------------

    def rewards(self, ctx: TaskContext) -> object:
        """Track the goal, finish on the worst joint, pay for the way there."""
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg
        from isaaclab.utils import configclass

        from .rewards import joint_pose_error, worst_joint_bonus

        params = {"asset_cfg": ctx.robot, "command_name": ctx.command}

        @configclass
        class Rewards:
            tracking = RewardTermCfg(func=joint_pose_error, weight=-1.0, params=params)
            tracking_worst = RewardTermCfg(
                func=worst_joint_bonus,
                weight=0.5,
                params={**params, "std": 0.1},
            )
            joint_velocity = RewardTermCfg(
                func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": ctx.robot}
            )
            action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3)

        return Rewards()

    def terminations(self, ctx: TaskContext) -> object:
        """Only the horizon: a run does not stop at the goal, so an episode does not either."""
        from isaaclab.envs import mdp
        from isaaclab.managers import TerminationTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

        return Terminations()

    def events(self, ctx: TaskContext) -> object:
        """Put the legs back still; scatter the upper body around the reference pose."""
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab.utils import configclass

        legs = SceneEntityCfg(ctx.asset, joint_names=list(LEGS), preserve_order=True)

        @configclass
        class Events:
            reset_legs = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": legs,
                },
            )
            reset_upper_body = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-RESET_SPREAD, RESET_SPREAD),
                    "velocity_range": (-RESET_SPEED, RESET_SPEED),
                    "asset_cfg": ctx.robot,
                },
            )

        return Events()

    def agent(self, ctx: TaskContext) -> Mapping[str, Any] | None:
        """A wider network than the arm's: 60 inputs and 15 outputs, against 24 and 6.

        The backend merges this over its own agent key by key, so the policy is stated
        whole; everything else stays the backend's.
        """
        return {
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [128, 128],
                "critic_hidden_dims": [128, 128],
                "activation": "elu",
            },
        }
