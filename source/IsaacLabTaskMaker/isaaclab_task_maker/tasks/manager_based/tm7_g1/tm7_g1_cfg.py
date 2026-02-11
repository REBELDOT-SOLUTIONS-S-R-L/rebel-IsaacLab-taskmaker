# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example IL task config — GR1T2 humanoid pick-and-place.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO CREATE A NEW TASK                                          ║
║                                                                    ║
║  1. Copy this file → my_task_cfg.py                                ║
║  2. Swap the robot:  change GR1T2_HIGH_PD_CFG to your robot cfg    ║
║  3. Swap the scene:  change UsdFileCfg paths, add/remove objects   ║
║  4. Update joints:   set pink_controlled_joint_names, hand_joints  ║
║  5. Update EEF:      set eef_names, eef_action_slices,             ║
║                      eef_gripper_slices, and link names in obs     ║
║  6. Register:        add gym.register() in __init__.py             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import tempfile

import torch
from pink.tasks import DampingTask, FrameTask

import carb

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_task_maker.tasks.manager_based.base_il_env import mdp
from isaaclab_task_maker.tasks.manager_based.base_il_env.base_il_env_cfg import BaseILEnvCfg

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG # isort: skip

SCENE_USD_PATH = "/home/voicandrei/Desktop/simulare_tm7_g1/scenes/scene.usd"
TM7_BOWL_USD_PATH = "/home/voicandrei/Desktop/simulare_tm7_g1/objects/tm7_bowl_collisions.usd"
TM7_LID_USD_PATH = "/home/voicandrei/Desktop/simulare_tm7_g1/objects/tm7_lid_collisions.usd"

# =====================================================================
# 1) SCENE — swap robot, add objects, change USD paths
# =====================================================================
@configclass
class TM7G1SceneCfg(InteractiveSceneCfg):
    """GR1T2 pick-place scene. Swap robot/objects here for new tasks."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(
            usd_path=SCENE_USD_PATH,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    tm7_bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/tm7_bowl",
        spawn=UsdFileCfg(
            usd_path=TM7_BOWL_USD_PATH,
            scale=(0.35, 0.35, 0.35),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 0.185), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    tm7_lid: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/tm7_lid",
        spawn=UsdFileCfg(
            usd_path=TM7_LID_USD_PATH,
            scale=(0.35, 0.35, 0.35),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 0.2), rot=(0.5, 0.5, -0.5, 0.5)),
    )

    # ---- SWAP THIS for a different robot ----
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.9, 0.0),
            rot=(0.7071, 0, 0, -0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # -- left/right hand
                ".*_thumb_.*": 0.0,
                ".*_index_.*": 0.0,
                ".*_middle_.*": 0.0,
                ".*_ring_.*": 0.0,
                ".*_pinky_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


# =====================================================================
# 2) ACTIONS — IK controller config with robot-specific joints
# =====================================================================
@configclass
class TM7G1ActionsCfg:
    """Pink IK actions for GR1T2. Change joint/link names for another robot."""
    # To be populated by agent env cfg
    # Temporarily control only right arm (7 joints) to match 7-action teleop device
    # Right arm joints: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=[
            # All the drive and mimic joints, total 24 joints
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "R_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )

    


# =====================================================================
# 3) OBSERVATIONS — EEF pos/quat following {eef_name}_eef_pos convention
# =====================================================================
@configclass
class TM7G1ObservationsCfg:
    """Observations for GR1T2. Change link_name params for another robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State observations for policy."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# =====================================================================
# 4) EVENTS & TERMINATIONS
# =====================================================================
@configclass
class TM7G1EventCfg:
    """Events for the pick-place task."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TM7G1TerminationsCfg:
    """Terminations: time-out + object dropping."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

# =====================================================================
# 5) FINAL CONFIG — inherits BaseILEnvCfg, sets all GR1T2-specific values
# =====================================================================
@configclass
class TM7G1TaskCfg(BaseILEnvCfg):

    # Scene
    scene: TM7G1SceneCfg = TM7G1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # MDP
    observations: TM7G1ObservationsCfg = TM7G1ObservationsCfg()
    actions: TM7G1ActionsCfg = TM7G1ActionsCfg()
    events: TM7G1EventCfg = TM7G1EventCfg()
    terminations: TM7G1TerminationsCfg = TM7G1TerminationsCfg()

    # XR anchor
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26

    # Idle action to hold the robot in default pose
    idle_action = torch.tensor(
        [
            # 14 hand joints for EEF control
            -0.1487,
            0.2038,
            1.0952,
            0.707,
            0.0,
            0.0,
            0.707,
            0.1487,
            0.2038,
            1.0952,
            0.707,
            0.0,
            0.0,
            0.707,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    def __post_init__(self):
        super().__post_init__()

        # Convert USD to URDF for Pink IK
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        # Teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
