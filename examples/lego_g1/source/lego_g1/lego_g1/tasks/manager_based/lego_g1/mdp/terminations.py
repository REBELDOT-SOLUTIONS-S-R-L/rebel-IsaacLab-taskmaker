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

from lego_g1.base_il_env.mdp.observations import get_object_pos, get_proximal_joint_mean, to_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    left_obj = get_object_pos(env, left_object_name)
    right_obj = get_object_pos(env, right_object_name)
    left_target = to_tensor(left_target_pos, env.device).expand_as(left_obj)
    right_target = to_tensor(right_target_pos, env.device).expand_as(right_obj)

    left_at_target = torch.norm(left_obj - left_target, dim=-1) <= target_dist_threshold
    right_at_target = torch.norm(right_obj - right_target, dim=-1) <= target_dist_threshold
    left_open = get_proximal_joint_mean(env, left_gripper_joint_pattern) <= gripper_open_threshold
    right_open = get_proximal_joint_mean(env, right_gripper_joint_pattern) <= gripper_open_threshold

    left_success = left_at_target & left_open
    right_success = right_at_target & right_open
    return left_success & right_success
