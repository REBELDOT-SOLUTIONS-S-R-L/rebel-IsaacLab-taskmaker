# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic data-generation config for LegoG1.

Mixes the task cfg with isaaclab's :class:`MimicEnvCfg`, which adds the
``datagen_config`` and ``subtask_configs`` fields consumed by the Mimic data
generation pipeline. The generated ``IL-LEGO-G1-v0-Mimic`` gym id (see this
package's ``__init__.py``) wires this cfg to ``BaseILEnv``.

Each arm has four subtasks in order:
    1. ``grasp_brick``    — close the hand around the brick.
    2. ``move_brick``     — carry the brick to its placement target.
    3. ``release_brick``  — open the hand once the brick is at the target.
    4. ``idle``           — return the EEF to the per-arm idle pose.

The ``subtask_term_signal`` names match attributes on the ``SubtaskTermsCfg``
observation group (group name ``subtask_terms``) in ``lego_g1_cfg.py``.
During annotated teleop, ``BaseILEnv.get_subtask_term_predicates`` reads the
raw 0/1 values out of ``obs_buf["subtask_terms"]`` and feeds them to
``record_annotated_demos.py``'s dwell-and-latch logic; the latched signals
are then written into the recorded HDF5 and consumed by data generation.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .lego_g1_cfg import LegoG1TaskCfg


@configclass
class LegoG1MimicEnvCfg(LegoG1TaskCfg, MimicEnvCfg):
    """Mimic-enabled configuration for LegoG1."""

    def __post_init__(self):
        # Run both parents' post-inits first so all task scaffolding (scene,
        # actions, observations, eef_names → recorders) is in place.
        super().__post_init__()

        # ── datagen_config ────────────────────────────────────────────────
        self.datagen_config.name = "demo_src_lego_g1_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # ── subtask_configs (per end-effector) ────────────────────────────
        # The final subtask of each arm uses ``subtask_term_signal=None`` per
        # IsaacLab convention — it runs until end-of-demo. Swap to
        # ``"idle_<arm>"`` if you'd rather gate it on the explicit idle obs.

        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="red_brick",
                subtask_term_signal="grasp_brick_left",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                first_subtask_start_offset_range=(0, 0),
            ),
            SubTaskConfig(
                object_ref="red_brick",
                subtask_term_signal="move_brick_left",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="red_brick",
                subtask_term_signal="release_brick_left",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="red_brick",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]

        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref="blue_brick",
                subtask_term_signal="grasp_brick_right",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                first_subtask_start_offset_range=(0, 0),
            ),
            SubTaskConfig(
                object_ref="blue_brick",
                subtask_term_signal="move_brick_right",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="blue_brick",
                subtask_term_signal="release_brick_right",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="blue_brick",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
