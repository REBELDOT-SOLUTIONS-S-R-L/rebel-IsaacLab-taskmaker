# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base IL environment configuration.

This module provides ``BaseILEnvCfg``, a base config class designed for imitation
learning tasks with humanoid robots. To create a new task, subclass this config
and override only what changes (robot, scene, joint names, EEF config).

See ``example_task_cfg.py`` for a full working example.
"""

import tempfile

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import StandardAnnotatedMimicRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Default Scene
##


@configclass
class BaseILSceneCfg(InteractiveSceneCfg):
    """Minimal scene with ground, lights, and a robot placeholder.

    Override ``robot`` in your task config to swap the robot articulation.
    Add objects (e.g. ``RigidObjectCfg``) as needed.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Robot — override in your task config with .replace()
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=""),  # set in task config
    )


##
# Default MDP components (IL-oriented: no rewards, no curriculum)
##


@configclass
class BaseActionsCfg:
    """Action specifications — override in your task config.

    Actions are typically configured with PinkInverseKinematicsActionCfg
    for humanoid robots. This base leaves it as None so that each task
    config sets it up with the correct joint names and IK controller.
    """

    pass


@configclass
class BaseObservationsCfg:
    """Default observation group for IL.

    Override the inner PolicyCfg in your task config to add robot-specific
    observations (different link names, joint groups, etc.).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Base observations: actions, joint pos, root state."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseEventCfg:
    """Default events: reset scene to initial state."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class BaseTerminationsCfg:
    """Default terminations: time-out only.

    Add task-specific terminations (e.g. object dropping, success) in your
    task config.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Base Environment Config
##


@configclass
class BaseILEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for all IL tasks.

    Subclass this and override:
        - ``scene`` — swap robot, add objects, change USD scene
        - ``actions`` — configure IK controller with robot-specific joints
        - ``observations`` — add EEF obs with robot-specific link names
        - ``eef_names`` / ``eef_action_slices`` / ``eef_gripper_slices``

    See ``example_task_cfg.py`` for a working example.
    """

    # Scene
    scene: BaseILSceneCfg = BaseILSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # MDP
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: BaseActionsCfg = BaseActionsCfg()
    events: BaseEventCfg = BaseEventCfg()
    terminations: BaseTerminationsCfg = BaseTerminationsCfg()

    # Mimic-annotated recorder is the default for all IL tasks. Override
    # `dataset_export_dir_path` / `dataset_filename` per-run from the launcher.
    # `entity_order` is wired to `eef_names` automatically in __post_init__.
    recorders: StandardAnnotatedMimicRecorderManagerCfg = StandardAnnotatedMimicRecorderManagerCfg()

    # IL tasks don't use these
    commands = None
    rewards = None
    curriculum = None

    # ------------------------------------------------------------------
    # EEF configuration — used by BaseILEnv to slice the action tensor.
    # Override these in your task config.
    # ------------------------------------------------------------------

    # List of end-effector names, e.g. ["left", "right"] or ["ee"]
    eef_names: list[str] = []

    # Per-EEF action tensor slicing.
    # Maps eef_name -> {"pos": (start, end), "quat": (start, end)}
    # Example for dual-arm:
    #   "left":  {"pos": (0, 3),  "quat": (3, 7)}
    #   "right": {"pos": (18, 21), "quat": (21, 25)}
    eef_action_slices: dict = {}

    # Per-EEF gripper slicing.
    # Maps eef_name -> (start, end) indexing into the action tensor.
    # Example: "left": (7, 18), "right": (25, 36)
    eef_gripper_slices: dict = {}

    # Temporary directory for URDF conversion (used by Pink IK)
    temp_urdf_dir: str = tempfile.gettempdir()

    # Optional callable invoked by IsaacLab's teleop_se3_agent.py after the XR
    # pipeline strips CameraCfg attributes via `remove_camera_configs`. Tasks
    # with cameras set this to the module-level `attach_cameras(scene_cfg)`
    # helper generated alongside the task cfg, so the cameras can be put back
    # before `gym.make` runs.
    xr_camera_reattach: object | None = None

    def __post_init__(self):
        """Post initialization — override in task configs, calling super().__post_init__()."""
        # Simulation defaults
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120  # 120 Hz
        self.sim.render_interval = 2

        # Mimic recorder needs the same EEF list that BaseILEnv's API methods iterate.
        # Subclasses that set `self.eef_names` should call super().__post_init__()
        # AFTER they assign it (or assign it before the super call — either works).
        if self.eef_names:
            self.recorders.entity_order = list(self.eef_names)
