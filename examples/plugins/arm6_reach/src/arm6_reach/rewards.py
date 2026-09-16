"""Reward terms of this task, written against a goal in joint space.

The vendor's reach task rewards a pose of the end effector, and computes the
error from forward kinematics the simulator already has. This one rewards a pose
of the joints, because that is what the goal of this task is (ADR-0043) and what
the observation carries.

Both terms read the goal through the command manager, by the name the platform
gave the command term, and the measured pose through the entity the platform
resolved over the contract's joints in the contract's order. Neither resolves
anything itself: a term that found its own joints could find them in another
order than the policy commands them in, and the mistake would show up as a
policy that almost works.

Torch is imported at module level here, unlike in ``task.py``: these functions are
called from inside the environment and never read by the platform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - the framework is absent outside training
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def joint_pose_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """Mean absolute deviation from the commanded pose, over the contract's joints.

    Never satisfied, so it keeps pulling from anywhere in the box. Absolute rather
    than squared: a squared error stops teaching anything once it is small, which
    is exactly the region the tolerance lives in.
    """
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(measured - goal), dim=1)


def joint_pose_error_tanh(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float
) -> torch.Tensor:
    """A bonus that is sharp near the goal: ``1 - tanh(error / std)``.

    What makes the last fraction of a radian worth reaching. ``std`` sets where
    "near" begins, and is of the order of the tolerance the task completes on.
    """
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.max(torch.abs(measured - goal), dim=1).values
    return 1.0 - torch.tanh(error / std)
