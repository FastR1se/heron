"""Keep the G1 standing on a free base: the training task of ADR-0075.

Nothing holds the robot but the policy. The pelvis is free (the vendor's description
has no ``world`` link, ADR-0048), the robot appears where the scenario places it
(ADR-0077), and it feels its body through the pelvis IMU -- how the pelvis turns and
which way is down (ADR-0076).

The goal is the standing pose itself, the contract's ``default_pose``: a box of zero
width. The goal term is still the platform's (ADR-0043) -- it is what completion is
judged against -- so the trained skill is done when every joint has stayed near that
pose for ``hold_s`` of world time. A fall is not an outcome of its own: the pose
leaves the tolerance and the skill fails at the horizon.

What this file decides and what it cannot:

* it says what is rewarded, that a fall ends an episode, how a reset scatters the
  robot and how it is pushed while it stands;
* it cannot say what the policy emits, what it observes, at what rate, through what
  drive or where the robot appears: those are the contract's and the scenario's
  (ADR-0044, ADR-0077).

It knows it is written for a G1, which the platform never does: the name of the
pelvis and the height a fallen pelvis sinks below are this package's knowledge, not
the core's (invariant 1).

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

PELVIS: Final = "pelvis"
"""The root link of the vendor's description: what falls, tilts and is pushed."""

FALL_HEIGHT: Final = 0.5
"""Height of the pelvis, metres, below which the robot has fallen. It stands at about
0.78 m (0.793 m on straight legs in the vendor's MuJoCo scene); on its knees it is
lower than this."""

TILT_LIMIT: Final = 0.8
"""Angle between the pelvis's up and the world's, radians, past which it has fallen."""

RESET_JOINTS: Final = 0.1
"""How far a reset scatters every joint around the standing pose, radians. Inside the
narrowest frame of the contract (ankle roll +-0.2618 about 0)."""

RESET_PLACE: Final = 0.5
"""How far a reset moves the robot along the floor, metres, around the scenario's pose."""

RESET_SPEED: Final = 0.2
"""Linear and angular speed a reset may start the pelvis with."""

PUSH_SPEED: Final = 0.5
"""Horizontal speed a push gives the pelvis, m/s -- a firm shove, not a stumble."""

PUSH_EVERY: Final = (5.0, 10.0)
"""Seconds between pushes, drawn per environment."""

FRICTION: Final = (0.4, 1.0)
"""Range of friction of every body with the floor -- the feet are what touch it. The
floor of the world of execution is not the training floor."""

PELVIS_MASS: Final = (-1.0, 3.0)
"""Mass added to the pelvis, kg: a battery, a payload, an error of the description."""


class G1StandTask:
    """Hold the standing pose on a free base, and stay up when pushed."""

    backend = BACKEND

    # Written as data, as a profile writes it: the closed registry of quantities parses
    # ``requires`` (ADR-0008). Exactly twenty-seven: the whole of body 6, legs included.
    spec = SkillSpec.model_validate(
        {
            "name": "g1_stand",
            "required_capabilities": [Capability("arm.joint_position")],
            "required_control_mode": ControlMode.STREAMING,
            "requires": {"dof": "==27"},
        }
    )

    # Radians, on the worst of twenty-seven joints (ADR-0043). Loose: to stand against a
    # push the policy moves its hips and arms away from the pose, and that is standing.
    # Ten seconds of world time held is the evidence; the episode is twice that.
    completion = Completion(tolerance=0.25, hold_s=10.0, horizon_s=20.0)

    # -- what the goal is ---------------------------------------------------

    def goal_range(self, ctx: TaskContext) -> JointBox:
        """The standing pose and nothing around it: the contract's reference pose."""
        pose = list(ctx.contract.default_pose or [0.0] * ctx.contract.dof)
        return JointBox(min=pose, max=list(pose))

    # -- what is rewarded ---------------------------------------------------

    def rewards(self, ctx: TaskContext) -> object:
        """Stay alive and upright, stay put, stay in the pose, and do it smoothly."""
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg
        from isaaclab.utils import configclass

        from .rewards import base_planar_speed_l2, joint_pose_error, worst_joint_bonus

        pose = {"asset_cfg": ctx.robot, "command_name": ctx.command}

        @configclass
        class Rewards:
            alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
            fall = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
            upright = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0)
            vertical_speed = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
            tilt_rate = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
            drift = RewardTermCfg(
                func=base_planar_speed_l2, weight=-1.0, params={"asset_name": ctx.asset}
            )
            pose_mean = RewardTermCfg(func=joint_pose_error, weight=-1.0, params=pose)
            pose_worst = RewardTermCfg(
                func=worst_joint_bonus, weight=0.5, params={**pose, "std": 0.2}
            )
            torques = RewardTermCfg(
                func=mdp.joint_torques_l2, weight=-1e-5, params={"asset_cfg": ctx.robot}
            )
            joint_acc = RewardTermCfg(
                func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": ctx.robot}
            )
            action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

        return Rewards()

    def terminations(self, ctx: TaskContext) -> object:
        """The horizon, or a fall: the pelvis too low or too tilted."""
        from isaaclab.envs import mdp
        from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
        from isaaclab.utils import configclass

        body = SceneEntityCfg(ctx.asset)

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
            fell = TerminationTermCfg(
                func=mdp.root_height_below_minimum,
                params={"minimum_height": FALL_HEIGHT, "asset_cfg": body},
            )
            tilted = TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": TILT_LIMIT, "asset_cfg": body},
            )

        return Terminations()

    def events(self, ctx: TaskContext) -> object:
        """Scatter the robot around the standing pose; push it; vary the floor and the load."""
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab.utils import configclass

        body = SceneEntityCfg(ctx.asset)
        bodies = SceneEntityCfg(ctx.asset, body_names=".*")
        pelvis = SceneEntityCfg(ctx.asset, body_names=PELVIS)
        speed = (-RESET_SPEED, RESET_SPEED)

        @configclass
        class Events:
            friction = EventTermCfg(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": bodies,
                    "static_friction_range": FRICTION,
                    "dynamic_friction_range": FRICTION,
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 64,
                },
            )
            load = EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": pelvis,
                    "mass_distribution_params": PELVIS_MASS,
                    "operation": "add",
                },
            )
            reset_base = EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": body,
                    "pose_range": {
                        "x": (-RESET_PLACE, RESET_PLACE),
                        "y": (-RESET_PLACE, RESET_PLACE),
                        "yaw": (-3.14159, 3.14159),
                    },
                    "velocity_range": {
                        "x": speed, "y": speed, "z": speed,
                        "roll": speed, "pitch": speed, "yaw": speed,
                    },
                },
            )
            reset_joints = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "asset_cfg": ctx.robot,
                    "position_range": (-RESET_JOINTS, RESET_JOINTS),
                    "velocity_range": (0.0, 0.0),
                },
            )
            push = EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=PUSH_EVERY,
                params={
                    "asset_cfg": body,
                    "velocity_range": {
                        "x": (-PUSH_SPEED, PUSH_SPEED),
                        "y": (-PUSH_SPEED, PUSH_SPEED),
                    },
                },
            )

        return Events()

    def agent(self, ctx: TaskContext) -> Mapping[str, Any] | None:
        """A network for 114 inputs and 27 outputs; everything else is the backend's.

        The backend merges this over its own agent key by key, so the policy is stated
        whole. No observation normalizer: the scales of the contract do that job, and the
        export path is the one both earlier tasks proved.
        """
        return {
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 0.8,
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [256, 128, 128],
                "critic_hidden_dims": [256, 128, 128],
                "activation": "elu",
            },
        }
