# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination functions for lego_g1.

Add custom termination conditions here. They will be available as `mdp.<func_name>`
in your task config's TerminationsCfg.

Base terminations (time_out, etc.) are already available from isaaclab.envs.mdp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_pos(env: "ManagerBasedRLEnv", object_name: str) -> torch.Tensor:
    """Object root position in env-local coordinates."""
    return env.scene[object_name].data.root_pos_w - env.scene.env_origins


def _proximal_joint_mean(env: "ManagerBasedRLEnv", joint_pattern: str) -> torch.Tensor:
    """Mean joint position for the hand joints matched by ``joint_pattern``."""
    robot = env.scene["unitree_g1"]
    idxs, _ = robot.find_joints(joint_pattern)
    if not idxs:
        return torch.zeros(env.num_envs, device=env.device)
    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=env.device)
    return robot.data.joint_pos[:, idx_tensor].mean(dim=1)


def bricks_released_at_targets(
    env: "ManagerBasedRLEnv",
    left_object_name: str,
    right_object_name: str,
    left_target_pos: tuple[float, float, float],
    right_target_pos: tuple[float, float, float],
    target_dist_threshold: float,
    left_gripper_joint_pattern: str,
    right_gripper_joint_pattern: str,
    gripper_open_threshold: float,
) -> torch.Tensor:
    """Task success for data generation: both bricks are at targets and both hands are open."""
    left_obj = _object_pos(env, left_object_name)
    right_obj = _object_pos(env, right_object_name)
    left_target = torch.as_tensor(left_target_pos, dtype=torch.float, device=env.device).expand_as(left_obj)
    right_target = torch.as_tensor(right_target_pos, dtype=torch.float, device=env.device).expand_as(right_obj)

    left_at_target = torch.norm(left_obj - left_target, dim=-1) <= target_dist_threshold
    right_at_target = torch.norm(right_obj - right_target, dim=-1) <= target_dist_threshold
    left_open = _proximal_joint_mean(env, left_gripper_joint_pattern) <= gripper_open_threshold
    right_open = _proximal_joint_mean(env, right_gripper_joint_pattern) <= gripper_open_threshold

    left_success = left_at_target & left_open
    right_success = right_at_target & right_open
    return left_success & right_success
