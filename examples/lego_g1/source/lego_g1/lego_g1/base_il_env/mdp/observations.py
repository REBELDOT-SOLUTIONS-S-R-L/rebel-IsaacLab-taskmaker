# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for IL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Get end-effector position relative to env origin.

    Args:
        env: The environment instance.
        link_name: Name of the EEF body/link in the URDF/USD.

    Returns:
        EEF position tensor of shape (num_envs, 3).
    """
    body_pos_w = env.scene["unitree_g1"].data.body_pos_w
    eef_idx = env.scene["unitree_g1"].data.body_names.index(link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Get end-effector quaternion (w, x, y, z).

    Args:
        env: The environment instance.
        link_name: Name of the EEF body/link in the URDF/USD.

    Returns:
        EEF quaternion tensor of shape (num_envs, 4).
    """
    body_quat_w = env.scene["unitree_g1"].data.body_quat_w
    eef_idx = env.scene["unitree_g1"].data.body_names.index(link_name)
    eef_quat = body_quat_w[:, eef_idx]
    return eef_quat


def get_object_pos(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Get object root position relative to env origin.

    Args:
        env: The environment instance.
        object_name: Name of the object in the scene.

    Returns:
        Object position tensor of shape (num_envs, 3).
    """
    return env.scene[object_name].data.root_pos_w - env.scene.env_origins


def get_proximal_joint_mean(env: ManagerBasedRLEnv, joint_pattern: str) -> torch.Tensor:
    """Get mean joint position for matched proximal hand joints.

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


def to_tensor(values, device) -> torch.Tensor:
    """Convert values to a float tensor on the requested device."""
    return torch.as_tensor(values, dtype=torch.float, device=device)


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Object observations (in world frame).

    Returns concatenated tensor of:
        object pos (3), object quat (4), left_eef_to_object (3), right_eef_to_object (3).

    Args:
        env: The environment instance.
        left_eef_link_name: Left EEF body/link name.
        right_eef_link_name: Right EEF body/link name.

    Returns:
        Observation tensor of shape (num_envs, 13).
    """
    left_eef_pos = get_eef_pos(env, left_eef_link_name)
    right_eef_pos = get_eef_pos(env, right_eef_link_name)
    object_pos = get_object_pos(env, "object")
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (object_pos, object_quat, left_eef_to_object, right_eef_to_object),
        dim=1,
    )


def get_robot_joint_state(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    """Get a subset of robot joint positions by name (supports regex).

    Args:
        env: The environment instance.
        joint_names: List of joint name patterns (regex supported).

    Returns:
        Joint position tensor of shape (num_envs, num_matched_joints).
    """
    indexes, _ = env.scene["unitree_g1"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    return env.scene["unitree_g1"].data.joint_pos[:, indexes]


def get_all_robot_link_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all robot link states (pos + quat + vel).

    Args:
        env: The environment instance.

    Returns:
        All link state tensor of shape (num_envs, num_bodies, state_dim).
    """
    return env.scene["unitree_g1"].data.body_link_state_w[:, :, :]
