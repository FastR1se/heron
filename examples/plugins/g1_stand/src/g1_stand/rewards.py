"""Reward terms of this task that the framework does not ship.

The pose terms read the goal through the command manager, by the name the platform
gave the command term, and the measured pose through the entity the platform
resolved over the contract's joints in the contract's order (ADR-0044). Neither
resolves joints itself: a term that found its own could find them in another order
than the policy commands them in.

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
    """Mean absolute deviation from the standing pose, over the contract's joints.

    Absolute rather than squared: a squared error stops teaching once it is small,
    and that is where the tolerance lives.
    """
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(measured - goal), dim=1)


def worst_joint_bonus(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float
) -> torch.Tensor:
    """``1 - tanh(max_i |q_i - goal_i| / std)``: completion is judged on the worst joint
    (ADR-0043), so the worst joint is what is paid for."""
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.max(torch.abs(measured - goal), dim=1).values
    return 1.0 - torch.tanh(error / std)


def base_planar_speed_l2(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """Squared speed of the pelvis along the floor, in its own frame: standing is not walking."""
    asset = env.scene[asset_name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
