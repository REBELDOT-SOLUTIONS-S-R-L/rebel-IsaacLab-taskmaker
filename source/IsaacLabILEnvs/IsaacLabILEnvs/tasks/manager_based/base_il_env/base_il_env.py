# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base IL environment class implementing the 4 mandatory ManagerBasedRLMimicEnv methods.

This class is designed so that you NEVER need to subclass it for new tasks.
All task-specific behavior is driven by config (BaseILEnvCfg subclasses).

The 4 mandatory methods use config attributes to know how to slice the action tensor:
    - cfg.eef_names: list of EEF names, e.g. ["left", "right"]
    - cfg.eef_action_slices: dict mapping eef_name -> (pos_start, quat_start, quat_end)
    - cfg.eef_gripper_slices: dict mapping eef_name -> (gripper_start, gripper_end)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class BaseILEnv(ManagerBasedRLMimicEnv):
    """Shared IL environment for all tasks.

    Implements the 4 mandatory methods of ManagerBasedRLMimicEnv generically.
    Works for single-arm and dual-arm robots — the number of arms and action
    layout is configured entirely through cfg attributes.

    Required config attributes:
        eef_names (list[str]): Names of end-effectors, e.g. ["left", "right"].
        eef_action_slices (dict): Per-EEF action tensor layout.
            Each entry: eef_name -> {"pos": (start, end), "quat": (start, end)}
        eef_gripper_slices (dict): Per-EEF gripper tensor layout.
            Each entry: eef_name -> (start, end)
    """

    # ------------------------------------------------------------------
    # 1) get_robot_eef_pose
    # ------------------------------------------------------------------
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end-effector pose as a 4x4 matrix.

        Reads ``{eef_name}_eef_pos`` and ``{eef_name}_eef_quat`` from the
        observation buffer. Observations must follow this naming convention
        (set via ObsTerm in your config).

        Args:
            eef_name: Name of the end effector (must match a key in cfg.eef_names).
            env_ids: Environment indices. If None, all envs are returned.

        Returns:
            Pose matrix of shape (len(env_ids), 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    # ------------------------------------------------------------------
    # 2) target_eef_pose_to_action
    # ------------------------------------------------------------------
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses + gripper actions into an env action tensor.

        Builds the action by iterating over ``cfg.eef_names`` and concatenating
        ``[pos(3), quat(4), gripper(N)]`` for each arm.

        Args:
            target_eef_pose_dict: Maps eef_name -> 4x4 target pose.
            gripper_action_dict: Maps eef_name -> gripper action tensor.
            action_noise_dict: Optional per-EEF noise scale.
            env_id: Environment index (unused for absolute actions, kept for API compliance).

        Returns:
            Flat action tensor compatible with env.step().
        """
        parts = []

        for eef_name in self.cfg.eef_names:
            target_pose = target_eef_pose_dict[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
            target_quat = PoseUtils.quat_from_matrix(target_rot)

            gripper_action = gripper_action_dict[eef_name]

            if action_noise_dict is not None and eef_name in action_noise_dict:
                noise_scale = action_noise_dict[eef_name]
                pos_noise = noise_scale * torch.randn_like(target_pos)
                quat_noise = noise_scale * torch.randn_like(target_quat)
                target_pos = target_pos + pos_noise
                target_quat = target_quat + quat_noise

            parts.append(torch.cat((target_pos, target_quat, gripper_action), dim=0))

        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # 3) action_to_target_eef_pose
    # ------------------------------------------------------------------
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert an action tensor back into per-EEF target poses.

        Uses ``cfg.eef_action_slices`` to know where pos and quat live
        inside the action tensor for each arm.

        Args:
            action: Action tensor of shape (num_envs, action_dim).

        Returns:
            Dict mapping eef_name -> 4x4 pose tensor of shape (num_envs, 4, 4).
        """
        target_poses = {}

        for eef_name in self.cfg.eef_names:
            slices = self.cfg.eef_action_slices[eef_name]
            pos_s, pos_e = slices["pos"]
            quat_s, quat_e = slices["quat"]

            pos = action[:, pos_s:pos_e]
            quat = action[:, quat_s:quat_e]
            rot_mat = PoseUtils.matrix_from_quat(quat)
            target_poses[eef_name] = PoseUtils.make_pose(pos, rot_mat)

        return target_poses

    # ------------------------------------------------------------------
    # 4) actions_to_gripper_actions
    # ------------------------------------------------------------------
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-EEF gripper actions from the full action tensor.

        Uses ``cfg.eef_gripper_slices`` to know where gripper dimensions live.

        Args:
            actions: Action tensor of shape (num_envs, num_steps, action_dim)
                     or (num_envs, action_dim).

        Returns:
            Dict mapping eef_name -> gripper action tensor.
        """
        result = {}
        for eef_name in self.cfg.eef_names:
            g_start, g_end = self.cfg.eef_gripper_slices[eef_name]
            result[eef_name] = actions[:, g_start:g_end]
        return result
