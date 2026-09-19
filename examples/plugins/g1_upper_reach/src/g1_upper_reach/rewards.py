"""Reward terms of this task, written against a goal in joint space.

Both read the goal through the command manager, by the name the platform gave the
command term, and the measured pose through the entity the platform resolved over the
contract's joints in the contract's order (ADR-0044). Neither resolves joints itself: a
term that found its own could find them in another order than the policy commands them
in, and the mistake would show up as a policy that almost works.

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

    Never satisfied, so it keeps pulling from anywhere in the box. Absolute rather than
    squared: a squared error stops teaching once it is small, and that is where the
    tolerance lives.
    """
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(measured - goal), dim=1)


def worst_joint_bonus(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float
) -> torch.Tensor:
    """``1 - tanh(max_i |q_i - goal_i| / std)``: sharp near the goal, on the worst joint.

    The worst joint and not the mean, because completion is judged on the worst joint
    (``max |q - target| <= tolerance``, ADR-0043): fifteen joints close on average and
    one far off is a run that never completes.
    """
    asset = env.scene[asset_cfg.name]
    goal = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.max(torch.abs(measured - goal), dim=1).values
    return 1.0 - torch.tanh(error / std)
