# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation functions for lego_g1.

This module defines the **subtask termination signals** that the Mimic data
generation pipeline uses to slice a demonstration into reusable segments.
Each signal returns a ``(num_envs, 1)`` float tensor that is ``1.0`` when the
associated subtask is complete for that env and ``0.0`` otherwise.

The signals are wired into the Mimic recorder via a ``SubtaskTermsCfg``
observation group (group name ``subtask_terms``) in ``lego_g1_cfg.py``, read
out by ``BaseILEnv.get_subtask_term_predicates``, and referenced by name
(e.g. ``grasp_brick_left``) from ``lego_g1_mimic_cfg.py``'s ``SubTaskConfig``
entries.

Thresholds here are placeholder values — refine them once a few demos have been
recorded and you can inspect the actual EEF/object trajectories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lego_g1.base_il_env.mdp.observations import (
    get_eef_pos,
    get_object_pos,
    to_tensor,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_proximal_joint_mean(env: "ManagerBasedRLEnv", joint_pattern: str) -> torch.Tensor:
    """Mean joint position over the Inspire-hand proximal joints matching ``joint_pattern``.

    Used by the subtask predicates below and by ``mdp/terminations.py`` to decide
    whether a hand is "closed" (mean ≥ threshold) or "open" (mean ≤ threshold).
    Task-specific because it assumes the G1 robot's Inspire FTP hand naming
    scheme; other tasks (no hands, different hand topology) define their own.

    Args:
        env: The environment instance.
        joint_pattern: Joint name pattern forwarded to ``find_joints``.

    Returns:
        Mean joint position tensor of shape (num_envs,).
    """
    robot = env.scene["unitree_g1"]
    indexes, _ = robot.find_joints(joint_pattern)
    if not indexes:
        return torch.zeros(env.num_envs, device=env.device)
    indexes = torch.tensor(indexes, dtype=torch.long, device=env.device)
    return robot.data.joint_pos[:, indexes].mean(dim=1)


# ---------------------------------------------------------------------------
# Subtask termination signals (subtask_terms observation group)
# ---------------------------------------------------------------------------
def grasp_brick_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    object_name: str,
    dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_closed_threshold: float,
) -> torch.Tensor:
    """Fires when the EEF is within ``dist_threshold`` of the brick AND the
    fingers' proximal joints are at least ``gripper_closed_threshold``."""
    eef = get_eef_pos(env, eef_link)
    obj = get_object_pos(env, object_name)
    eef_obj_dist = torch.norm(eef - obj, dim=-1)
    grip = get_proximal_joint_mean(env, gripper_joint_pattern)
    near = eef_obj_dist <= dist_threshold
    closed = grip >= gripper_closed_threshold
    return (near & closed).float().unsqueeze(-1)


def move_brick_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    object_name: str,
    target_pos: tuple[float, float, float],
    eef_dist_threshold: float,
    target_dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_closed_threshold: float,
) -> torch.Tensor:
    """Fires when the EEF still holds the brick (near + closed) AND the brick
    has been carried within ``target_dist_threshold`` of ``target_pos``."""
    eef = get_eef_pos(env, eef_link)
    obj = get_object_pos(env, object_name)
    target = to_tensor(target_pos, env.device).expand_as(obj)
    eef_obj_dist = torch.norm(eef - obj, dim=-1)
    obj_tgt_dist = torch.norm(obj - target, dim=-1)
    grip = get_proximal_joint_mean(env, gripper_joint_pattern)
    near_eef = eef_obj_dist <= eef_dist_threshold
    near_target = obj_tgt_dist <= target_dist_threshold
    closed = grip >= gripper_closed_threshold
    return (near_eef & near_target & closed).float().unsqueeze(-1)


def release_brick_done(
    env: "ManagerBasedRLEnv",
    object_name: str,
    target_pos: tuple[float, float, float],
    target_dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_open_threshold: float,
) -> torch.Tensor:
    """Fires when the brick is within ``target_dist_threshold`` of ``target_pos``
    AND the fingers' proximal joints are at most ``gripper_open_threshold``."""
    obj = get_object_pos(env, object_name)
    target = to_tensor(target_pos, env.device).expand_as(obj)
    obj_tgt_dist = torch.norm(obj - target, dim=-1)
    grip = get_proximal_joint_mean(env, gripper_joint_pattern)
    near_target = obj_tgt_dist <= target_dist_threshold
    is_open = grip <= gripper_open_threshold
    return (near_target & is_open).float().unsqueeze(-1)


def idle_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    idle_pos: tuple[float, float, float],
    threshold: tuple[float, float, float],
) -> torch.Tensor:
    """Fires when the EEF is inside an axis-aligned box of half-extents
    ``threshold`` centred on ``idle_pos``."""
    eef = get_eef_pos(env, eef_link)
    target = to_tensor(idle_pos, env.device).expand_as(eef)
    thr = to_tensor(threshold, env.device)
    delta = (eef - target).abs()
    per_axis_ok = delta <= thr
    in_range = per_axis_ok.all(dim=-1)
    return in_range.float().unsqueeze(-1)
