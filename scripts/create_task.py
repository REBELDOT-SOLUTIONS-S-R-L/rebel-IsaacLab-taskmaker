#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a new IL task folder from a YAML definition file.

Usage:
    python scripts/create_task.py example.yaml
    python scripts/create_task.py example.yaml --dry-run
"""

import argparse
import os
import sys
import textwrap

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PACKAGE_DIR = os.path.join(ROOT_DIR, "source", "IsaacLabTaskMaker", "isaaclab_task_maker")
TASKS_DIR = os.path.join(PACKAGE_DIR, "tasks", "manager_based")
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
TEMPLATES_DIR = os.path.join(PACKAGE_DIR, "templates")
TASK_DEFS_DIR = os.path.join(TEMPLATES_DIR, "task_definitions")
SKELETON_PATH = os.path.join(TEMPLATES_DIR, "skeleton_task_cfg.py.template")


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase.  tm7_g1 → TM7G1"""
    return "".join(
        part.upper() if part.isalpha() and len(part) <= 3 else part.capitalize()
        for part in name.split("_")
    )


def format_list(items: list) -> str:
    """Format a Python list of strings as a single-line list literal."""
    inner = ", ".join(f'"{item}"' for item in items)
    return f"[{inner}]"


def format_dict(d: dict) -> str:
    """Format a Python dict of str→str as a single-line dict literal."""
    inner = ", ".join(f'"{k}": "{v}"' for k, v in d.items())
    return f"{{{inner}}}"


def build_asset_path_constants(task_name: str, cfg: dict) -> str:
    """Generate os.path.join constants for scene and object USDs."""
    lines = []

    # Scene USD path
    scene = cfg.get("scene", {})
    scene_file = scene.get("usd_file", "scene.usd")
    lines.append(f'SCENE_USD_PATH = os.path.join(SCENES_DIR, "{scene_file}")')

    # Object USD paths
    for obj in cfg.get("scene_objects", []):
        var_name = f'{obj["name"].upper()}_USD_PATH'
        usd_file = obj.get("usd_file", f'{obj["name"]}.usd')
        lines.append(f'{var_name} = os.path.join(OBJECTS_DIR, "{usd_file}")')

    return "\n".join(lines)


def build_scene_objects_block(objects_list: list) -> str:
    """Generate scene object declarations from YAML list."""
    if not objects_list:
        return ""
    lines = []
    for obj in objects_list:
        obj_type = obj.get("type", "RigidObjectCfg")
        pos = ", ".join(str(v) for v in obj["init_pos"])
        rot = ", ".join(str(v) for v in obj.get("init_rot", [1.0, 0.0, 0.0, 0.0]))
        scale = ", ".join(str(v) for v in obj.get("scale", [1.0, 1.0, 1.0]))
        var_name = f'{obj["name"].upper()}_USD_PATH'
        prim_path = obj['prim_path']
        lines.append(f"""\
    {obj['name']}: {obj_type} = {obj_type}(
        prim_path="{prim_path}",
        spawn=UsdFileCfg(
            usd_path={var_name},
            scale=({scale}),
        ),
        init_state={obj_type}.InitialStateCfg(pos=({pos}), rot=({rot})),
    )
""")
    return "\n".join(lines)


def build_frame_tasks(cfg: dict) -> str:
    """Generate FrameTask + NullSpacePostureTask blocks."""
    eef = cfg.get("eef", {})
    frame_names = eef.get("frame_names", {})
    ns_joints = cfg.get("ik_controller", {}).get("null_space_joints", [])

    lines = []
    for side, frame_name in frame_names.items():
        lines.append(textwrap.indent(textwrap.dedent(f"""\
            FrameTask(
                "{frame_name}",
                position_cost=8.0,
                orientation_cost=2.0,
                lm_damping=10,
                gain=0.5,
            ),"""), "                "))

    # NullSpacePostureTask — only if null_space_joints are specified
    if ns_joints:
        frame_list = list(frame_names.values())
        frames_str = ", ".join(f'"{f}"' for f in frame_list)
        joints_str = ",\n                        ".join(f'"{j}"' for j in ns_joints)

        lines.append(textwrap.indent(textwrap.dedent(f"""\
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        {frames_str},
                    ],
                    controlled_joints=[
                        {joints_str},
                    ],
                    gain=0.3,
                ),"""), "                "))

    return "\n".join(lines)


def generate_init_py(task_id: str, class_prefix: str, task_name: str) -> str:
    """Generate the __init__.py content for the task folder."""
    return textwrap.dedent(f"""\
        # Copyright (c) 2022-2026, The Isaac Lab Project Developers.
        # All rights reserved.
        #
        # SPDX-License-Identifier: BSD-3-Clause

        import gymnasium as gym

        gym.register(
            id="{task_id}",
            entry_point="isaaclab_task_maker.tasks.manager_based.base_il_env.base_il_env:BaseILEnv",
            disable_env_checker=True,
            kwargs={{
                "env_cfg_entry_point": f"{{__name__}}.{task_name}_cfg:{class_prefix}TaskCfg",
            }},
        )
    """)


def generate_task_cfg(cfg: dict) -> str:
    """Fill the skeleton template with values from the YAML config."""
    with open(SKELETON_PATH, "r") as f:
        template = f.read()

    task_name = cfg["task_name"]
    class_prefix = snake_to_pascal(task_name)

    robot = cfg.get("robot", {})
    scene = cfg.get("scene", {})
    ik = cfg.get("ik_controller", {})
    eef = cfg.get("eef", {})
    teleop = cfg.get("teleop", {})
    sim = cfg.get("sim", {})
    controller_type = ik.get("controller_type", "pink_ik")

    init_pos = ", ".join(str(v) for v in robot.get("init_pos", [0, 0, 0]))
    init_rot = ", ".join(str(v) for v in robot.get("init_rot", [1, 0, 0, 0]))
    scene_scale = ", ".join(str(v) for v in scene.get("scale", [1, 1, 1]))
    scene_pos = ", ".join(str(v) for v in scene.get("init_pos", [0, 0, 0]))
    scene_rot = ", ".join(str(v) for v in scene.get("init_rot", [1, 0, 0, 0]))
    robot_prim_path = robot.get("prim_path", "/World/envs/env_.*/Robot")

    # Robot import and block (always import mode now)
    robot_import_line = f"from {robot['import_path']} import {robot['config_name']}  # isort: skip"
    robot_block = (
        f'    robot: ArticulationCfg = {robot["config_name"]}.replace(\n'
        f'        prim_path="{robot_prim_path}",\n'
        f'        init_state=ArticulationCfg.InitialStateCfg(\n'
        f'            pos=({init_pos}),\n'
        f'            rot=({init_rot}),\n'
        f'            joint_pos={{".*": 0.0}},\n'
        f'            joint_vel={{".*": 0.0}},\n'
        f'        ),\n'
        f'    )'
    )

    replacements = {
        "{class_prefix}": class_prefix,
        "{task_name}": task_name,
        "{robot_import}": robot_import_line,
        "{robot_block}": robot_block,
        "{scene_scale}": scene_scale,
        "{scene_pos}": scene_pos,
        "{scene_rot}": scene_rot,
        "{asset_path_constants}": build_asset_path_constants(task_name, cfg),
        "{scene_objects_block}": build_scene_objects_block(cfg.get("scene_objects", [])),
        "{env_spacing}": str(sim.get("env_spacing", 2.5)),
        # Controller-specific blocks
        "{extra_imports}": build_extra_imports(controller_type),
        "{controller_imports}": build_controller_imports(controller_type, cfg),
        "{actions_block}": build_actions_block(controller_type, class_prefix, cfg),
        "{post_init_block}": build_post_init_block(controller_type, class_prefix, cfg),
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


# ---------------------------------------------------------------------------
# Controller-specific builders
# ---------------------------------------------------------------------------

SUPPORTED_CONTROLLERS = (
    "pink_ik",
    "differential_ik",
    "operational_space",
    "rmpflow",
    "joint_position",
    "relative_joint_position",
    "joint_velocity",
    "joint_effort",
)


def _build_gripper_block(ik: dict, action_cfg_class: str = "JointPositionActionCfg") -> str:
    """Build an optional gripper JointPosition action block if hand joints exist."""
    hand_joints = ik.get("hand_joint_names", [])
    if not hand_joints:
        return ""
    gripper_joints = format_list(hand_joints)
    return (
        f"\n\n"
        f"    gripper_action = {action_cfg_class}(\n"
        f"        asset_name=\"robot\",\n"
        f"        joint_names={gripper_joints},\n"
        f"        scale=1.0,\n"
        f"        use_default_offset=True,\n"
        f"    )"
    )


def _get_body_name(eef: dict) -> str:
    """Get the first EEF body name from the eef config."""
    target_links = eef.get("target_links", {})
    return list(target_links.values())[0] if target_links else "ee_link"


def build_extra_imports(controller_type: str) -> str:
    """Generate extra imports needed by the controller type."""
    if controller_type == "pink_ik":
        return textwrap.dedent("""\
            import torch
            from pink.tasks import FrameTask

            import carb

            import isaaclab.controllers.utils as ControllerUtils""")
    return ""


def build_controller_imports(controller_type: str, cfg: dict) -> str:
    """Generate controller-specific import lines."""
    teleop = cfg.get("teleop", {})
    imports_map = {
        "pink_ik": [
            "from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg",
            "from isaaclab.devices.device_base import DevicesCfg",
            "from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg",
            "from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg",
            f"from {teleop.get('retargeter_import', 'isaaclab.devices')} import {teleop.get('retargeter_class', 'RetargeterCfg')}",
        ],
        "differential_ik": [
            "from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg",
            "from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg",
        ],
        "operational_space": [
            "from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg",
            "from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg",
        ],
        "rmpflow": [
            "from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg",
            "from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg",
        ],
        "joint_position": [
            "from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg",
        ],
        "relative_joint_position": [
            "from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg",
        ],
        "joint_velocity": [
            "from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg",
        ],
        "joint_effort": [
            "from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg",
        ],
    }

    # Add Device imports for non-pink_ik controllers
    if controller_type != "pink_ik":
        device = teleop.get("device", "keyboard")
        imports_map[controller_type].append("from isaaclab.devices.device_base import DevicesCfg")
        if device == "keyboard":
            imports_map[controller_type].append("from isaaclab.devices.keyboard import Se3KeyboardCfg")
        elif device == "spacemouse":
            imports_map[controller_type].append("from isaaclab.devices.spacemouse import Se3SpaceMouseCfg")
        elif device == "gamepad":
            imports_map[controller_type].append("from isaaclab.devices.gamepad import Se3GamepadCfg")

    # Add JointPositionActionCfg import for gripper on task-space controllers
    ik = cfg.get("ik_controller", {})
    if ik.get("hand_joint_names") and controller_type in (
        "differential_ik", "operational_space", "rmpflow",
    ):
        imports_map[controller_type].append(
            "from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg"
        )

    lines = imports_map.get(controller_type, [f"# Unknown controller_type: {controller_type}"])
    return "\n".join(lines)


def build_actions_block(controller_type: str, class_prefix: str, cfg: dict) -> str:
    """Generate the full @configclass ActionsCfg block."""
    ik = cfg.get("ik_controller", {})
    eef = cfg.get("eef", {})

    builders = {
        "pink_ik": lambda: _build_pink_ik_actions(class_prefix, ik, eef, cfg),
        "differential_ik": lambda: _build_diff_ik_actions(class_prefix, ik, eef, cfg),
        "operational_space": lambda: _build_osc_actions(class_prefix, ik, eef, cfg),
        "rmpflow": lambda: _build_rmpflow_actions(class_prefix, ik, eef, cfg),
        "joint_position": lambda: _build_simple_joint_actions(
            class_prefix, ik, "JointPositionActionCfg", "Joint position",
            extra_params="scale=1.0,\n        use_default_offset=True,"),
        "relative_joint_position": lambda: _build_simple_joint_actions(
            class_prefix, ik, "RelativeJointPositionActionCfg", "Relative joint position",
            extra_params="scale=1.0,\n        use_zero_offset=True,"),
        "joint_velocity": lambda: _build_simple_joint_actions(
            class_prefix, ik, "JointVelocityActionCfg", "Joint velocity",
            extra_params="scale=1.0,\n        use_default_offset=True,"),
        "joint_effort": lambda: _build_simple_joint_actions(
            class_prefix, ik, "JointEffortActionCfg", "Joint effort",
            extra_params=""),
    }

    builder = builders.get(controller_type)
    if builder:
        return builder()
    return f"# TODO: Implement actions for controller_type={controller_type}"


# --- Pink IK ---

def _build_pink_ik_actions(class_prefix: str, ik: dict, eef: dict, cfg: dict) -> str:
    """Generate PinkIK actions block."""
    controlled = format_list(ik.get("controlled_joint_names", []))
    hand = format_list(ik.get("hand_joint_names", []))
    target_links = format_dict(eef.get("target_links", {}))
    base_link = ik.get("base_link_name", "base_link")
    num_hand = ik.get("num_hand_joints", 0)
    frame_tasks = build_frame_tasks(cfg)

    return (
        f"@configclass\n"
        f"class {class_prefix}ActionsCfg:\n"
        f'    """Pink IK actions for {class_prefix}."""\n'
        f"\n"
        f"    pink_ik_cfg = PinkInverseKinematicsActionCfg(\n"
        f"        pink_controlled_joint_names={controlled},\n"
        f"        hand_joint_names={hand},\n"
        f"        target_eef_link_names={target_links},\n"
        f'        asset_name="robot",\n'
        f"        controller=PinkIKControllerCfg(\n"
        f'            articulation_name="robot",\n'
        f'            base_link_name="{base_link}",\n'
        f"            num_hand_joints={num_hand},\n"
        f"            show_ik_warnings=False,\n"
        f"            fail_on_joint_limit_violation=False,\n"
        f"            variable_input_tasks=[\n"
        f"{frame_tasks}\n"
        f"            ],\n"
        f"            fixed_input_tasks=[],\n"
        f'            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),\n'
        f"        ),\n"
        f"        enable_gravity_compensation=False,\n"
        f"    )"
    )


# --- Differential IK ---

def _build_diff_ik_actions(class_prefix: str, ik: dict, eef: dict, cfg: dict) -> str:
    """Generate DifferentialIK actions block."""
    joint_names = format_list(ik.get("controlled_joint_names", []))
    body_name = _get_body_name(eef)
    ik_method = ik.get("ik_method", "dls")
    command_type = ik.get("command_type", "pose")
    use_relative = ik.get("use_relative_mode", False)
    body_offset = ik.get("body_offset", [0.0, 0.0, 0.0])
    offset_str = ", ".join(str(v) for v in body_offset)

    gripper_block = _build_gripper_block(ik)

    return (
        f"@configclass\n"
        f"class {class_prefix}ActionsCfg:\n"
        f'    """Differential IK actions for {class_prefix}."""\n'
        f"\n"
        f"    arm_action = DifferentialInverseKinematicsActionCfg(\n"
        f'        asset_name="robot",\n'
        f"        joint_names={joint_names},\n"
        f'        body_name="{body_name}",\n'
        f"        controller=DifferentialIKControllerCfg(\n"
        f'            command_type="{command_type}",\n'
        f"            use_relative_mode={use_relative},\n"
        f'            ik_method="{ik_method}",\n'
        f"        ),\n"
        f"        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[{offset_str}]),\n"
        f"    ){gripper_block}"
    )


# --- Operational Space Controller ---

def _build_osc_actions(class_prefix: str, ik: dict, eef: dict, cfg: dict) -> str:
    """Generate OperationalSpaceController actions block."""
    joint_names = format_list(ik.get("controlled_joint_names", []))
    body_name = _get_body_name(eef)
    body_offset = ik.get("body_offset", [0.0, 0.0, 0.0])
    offset_str = ", ".join(str(v) for v in body_offset)
    nullspace_target = ik.get("nullspace_joint_pos_target", "zero")

    gripper_block = _build_gripper_block(ik)

    return (
        f"@configclass\n"
        f"class {class_prefix}ActionsCfg:\n"
        f'    """Operational space controller actions for {class_prefix}."""\n'
        f"\n"
        f"    arm_action = OperationalSpaceControllerActionCfg(\n"
        f'        asset_name="robot",\n'
        f"        joint_names={joint_names},\n"
        f'        body_name="{body_name}",\n'
        f"        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=({offset_str})),\n"
        f"        controller_cfg=OperationalSpaceControllerCfg(\n"
        f'            target_types=["pose_abs"],\n'
        f'            impedance_mode="fixed",\n'
        f"            inertial_dynamics_decoupling=False,\n"
        f"            partial_inertial_dynamics_decoupling=False,\n"
        f"            gravity_compensation=False,\n"
        f"            motion_stiffness_task=100.0,\n"
        f"            motion_damping_ratio_task=1.0,\n"
        f"            contact_wrench_stiffness_task=0.0,\n"
        f"        ),\n"
        f'        nullspace_joint_pos_target="{nullspace_target}",\n'
        f"    ){gripper_block}"
    )


# --- RMPFlow ---

def _build_rmpflow_actions(class_prefix: str, ik: dict, eef: dict, cfg: dict) -> str:
    """Generate RMPFlow actions block."""
    joint_names = format_list(ik.get("controlled_joint_names", []))
    body_name = _get_body_name(eef)
    body_offset = ik.get("body_offset", [0.0, 0.0, 0.0])
    offset_str = ", ".join(str(v) for v in body_offset)
    use_relative = ik.get("use_relative_mode", False)
    robot_prim = cfg.get("robot", {}).get("prim_path", "/World/envs/env_.*/Robot")

    gripper_block = _build_gripper_block(ik)

    return (
        f"@configclass\n"
        f"class {class_prefix}ActionsCfg:\n"
        f'    """RMPFlow actions for {class_prefix}."""\n'
        f"\n"
        f"    arm_action = RMPFlowActionCfg(\n"
        f'        asset_name="robot",\n'
        f"        joint_names={joint_names},\n"
        f'        body_name="{body_name}",\n'
        f"        body_offset=RMPFlowActionCfg.OffsetCfg(pos=({offset_str})),\n"
        f"        controller=RmpFlowControllerCfg(),\n"
        f'        articulation_prim_expr="{robot_prim}",\n'
        f"        use_relative_mode={use_relative},\n"
        f"    ){gripper_block}"
    )


# --- Simple joint-space controllers (JointPosition, RelativeJointPosition, JointVelocity, JointEffort) ---

def _build_simple_joint_actions(
    class_prefix: str, ik: dict, cfg_class: str, label: str, extra_params: str = "",
) -> str:
    """Generate a simple joint-space actions block."""
    joint_names = format_list(ik.get("controlled_joint_names", []))
    gripper_block = _build_gripper_block(ik)

    extra = f"\n        {extra_params}" if extra_params else ""

    return (
        f"@configclass\n"
        f"class {class_prefix}ActionsCfg:\n"
        f'    """{label} actions for {class_prefix}."""\n'
        f"\n"
        f"    arm_action = {cfg_class}(\n"
        f'        asset_name="robot",\n'
        f"        joint_names={joint_names},{extra}\n"
        f"    ){gripper_block}"
    )


# --- Post-init block ---

def build_post_init_block(controller_type: str, class_prefix: str, cfg: dict) -> str:
    """Generate the __post_init__ method for the TaskCfg."""
    teleop = cfg.get("teleop", {})
    teleop_device = teleop.get("device", "keyboard")  # Default to keyboard if missing

    if controller_type == "pink_ik":
        retargeter_class = teleop.get("retargeter_class", "RetargeterCfg")
        # teleop_device is already extracted above, but pink_ik block used to default to "handtracking" locally if missing which is fine to override or keep consistent.
        # Let's use the one from config or default.
        
        # If device was missing in yaml, pink_ik usually defaults to handtracking, others to keyboard.
        # To preserve exact old behavior for pink_ik we could reuse specific default, but standardizing is better.
        # Let's stick effectively to what the code did:
        if "device" not in teleop:
            teleop_device = "handtracking"

        return (
            f"    xr: XrCfg = XrCfg(\n"
            f"        anchor_pos=(0.0, 0.0, 0.0),\n"
            f"        anchor_rot=(1.0, 0.0, 0.0, 0.0),\n"
            f"    )\n"
            f"\n"
            f"    NUM_OPENXR_HAND_JOINTS = 26\n"
            f"\n"
            f"    def __post_init__(self):\n"
            f"        super().__post_init__()\n"
            f"\n"
            f"        # Convert USD to URDF for Pink IK\n"
            f"        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(\n"
            f"            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True\n"
            f"        )\n"
            f"        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path\n"
            f"        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path\n"
            f"\n"
            f"        # Teleop devices\n"
            f"        self.teleop_devices = DevicesCfg(\n"
            f"            devices={{\n"
            f'                \"{teleop_device}\": OpenXRDeviceCfg(\n'
            f"                    retargeters=[\n"
            f"                        {retargeter_class}(\n"
            f"                            enable_visualization=True,\n"
            f"                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,\n"
            f"                            sim_device=self.sim.device,\n"
            f"                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,\n"
            f"                        ),\n"
            f"                    ],\n"
            f"                    sim_device=self.sim.device,\n"
            f"                    xr_cfg=self.xr,\n"
            f"                ),\n"
            f"            }}\n"
            f"        )\n"
        )
    else:
        # Generic teleop for other controllers (keyboard, spacemouse, etc.)
        device_type = teleop_device # This is now "keyboard" by default if missing
        
        device_config = ""
        if device_type == "keyboard":
            device_config = (
                f"                \"keyboard\": Se3KeyboardCfg(\n"
                f"                    pos_sensitivity=0.05,\n"
                f"                    rot_sensitivity=0.05,\n"
                f"                ),"
            )
        elif device_type == "spacemouse":
            device_config = (
                f"                \"spacemouse\": Se3SpaceMouseCfg(\n"
                f"                    pos_sensitivity=0.05,\n"
                f"                    rot_sensitivity=0.05,\n"
                f"                ),"
            )
        elif device_type == "gamepad":
            device_config = (
                f"                \"gamepad\": Se3GamepadCfg(\n"
                f"                    pos_sensitivity=10.0,\n"
                f"                    rot_sensitivity=10.0,\n"
                f"                ),"
            )

        if not device_config:
             # Fallback if unknown device
             return (
                "    def __post_init__(self):\n"
                "        super().__post_init__()\n"
             )

        return (
            f"    def __post_init__(self):\n"
            f"        super().__post_init__()\n"
            f"\n"
            f"        # Teleop devices\n"
            f"        self.teleop_devices = DevicesCfg(\n"
            f"            devices={{\n"
            f"{device_config}\n"
            f"            }}\n"
            f"        )\n"
        )



def create_asset_folders(task_name: str, cfg: dict, dry_run: bool = False):
    """Create asset subfolders under assets/scenes/, assets/objects/."""
    folders = [
        os.path.join(ASSETS_DIR, "scenes"),
        os.path.join(ASSETS_DIR, "objects"),
        os.path.join(ASSETS_DIR, "robots"),
    ]
    for folder in folders:
        if dry_run:
            print(f"  Would ensure exists: {folder}/")
        else:
            os.makedirs(folder, exist_ok=True)
            # print(f"  Ensured: {folder}/") # reduce spam

    # Print reminders for which USD files to place
    scene = cfg.get("scene", {})
    robot = cfg.get("robot", {})
    scene_file = scene.get("usd_file", "scene.usd")
    objects = cfg.get("scene_objects", [])

    print(f"\n  📁 Place your USD files:")
    print(f"     Scene:   assets/scenes/{scene_file}")
    for obj in objects:
        usd_file = obj.get("usd_file", f'{obj["name"]}.usd')
        print(f"     Object:  assets/objects/{usd_file}")


# ---------------------------------------------------------------------------
# MDP folder generation
# ---------------------------------------------------------------------------

MDP_INIT_TEMPLATE = """\
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

\"\"\"MDP functions for the {task_name} task.

This module re-exports everything from the base IL MDP and adds
task-specific observations, events, and terminations.
\"\"\"

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Base IL observations (EEF helpers, joint state, etc.)
from isaaclab_task_maker.tasks.manager_based.base_il_env.mdp import *  # noqa: F401, F403

# Task-specific overrides
from .observations import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
"""

MDP_OBSERVATIONS_TEMPLATE = """\
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

\"\"\"Task-specific observation functions for {task_name}.

Add custom observation terms here. They will be available as `mdp.<func_name>`
in your task config's ObservationsCfg.

Base observations (get_eef_pos, get_eef_quat, object_obs, get_robot_joint_state,
get_all_robot_link_state) are already available from the base_il_env mdp.
\"\"\"

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Example: uncomment and customize
# def my_custom_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
#     \"\"\"Compute a task-specific observation.\"\"\"
#     return torch.zeros(env.num_envs, 1, device=env.device)
"""

MDP_EVENTS_TEMPLATE = """\
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

\"\"\"Task-specific event functions for {task_name}.

Add custom event/reset functions here. They will be available as `mdp.<func_name>`
in your task config's EventCfg.

Base events (reset_scene_to_default, etc.) are already available from isaaclab.envs.mdp.
\"\"\"

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Example: uncomment and customize
# def my_custom_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
#     \"\"\"Custom reset logic for specific objects.\"\"\"
#     pass
"""

MDP_TERMINATIONS_TEMPLATE = """\
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

\"\"\"Task-specific termination functions for {task_name}.

Add custom termination conditions here. They will be available as `mdp.<func_name>`
in your task config's TerminationsCfg.

Base terminations (time_out, etc.) are already available from isaaclab.envs.mdp.
\"\"\"

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Example: uncomment and customize
# def object_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
#     \"\"\"Terminate if the object falls below a threshold.\"\"\"
#     object_pos = env.scene["object"].data.root_pos_w
#     return object_pos[:, 2] < 0.1
"""


def generate_mdp_files(task_name: str) -> dict[str, str]:
    """Generate the contents of all mdp/ files for a task.

    Returns:
        Dict mapping relative filename → content.
    """
    return {
        "mdp/__init__.py": MDP_INIT_TEMPLATE.format(task_name=task_name),
        "mdp/observations.py": MDP_OBSERVATIONS_TEMPLATE.format(task_name=task_name),
        "mdp/events.py": MDP_EVENTS_TEMPLATE.format(task_name=task_name),
        "mdp/terminations.py": MDP_TERMINATIONS_TEMPLATE.format(task_name=task_name),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a new IL task from a YAML definition.")
    parser.add_argument("yaml_file", help="Path to the YAML task definition file.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated files instead of writing them.")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing task folder if it exists.")
    args = parser.parse_args()

    # Resolve YAML path — accept just a filename or a full path
    yaml_path = args.yaml_file
    if not os.path.isfile(yaml_path):
        yaml_path = os.path.join(TASK_DEFS_DIR, args.yaml_file)
    if not os.path.isfile(yaml_path):
        print(f"ERROR: YAML file not found: {args.yaml_file}")
        print(f"       Looked in: {TASK_DEFS_DIR}/")
        sys.exit(1)

    # Load YAML
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    task_name = cfg["task_name"]
    task_id = cfg["task_id"]
    class_prefix = snake_to_pascal(task_name)

    task_dir = os.path.join(TASKS_DIR, task_name)
    init_path = os.path.join(task_dir, "__init__.py")
    cfg_path = os.path.join(task_dir, f"{task_name}_cfg.py")

    # Generate content
    init_content = generate_init_py(task_id, class_prefix, task_name)
    cfg_content = generate_task_cfg(cfg)
    mdp_files = generate_mdp_files(task_name)

    if args.dry_run:
        print(f"=== Would create: {task_dir}/ ===\n")
        print(f"--- {init_path} ---")
        print(init_content)
        print(f"--- {cfg_path} ---")
        print(cfg_content)
        for rel_path, content in mdp_files.items():
            print(f"\n--- {os.path.join(task_dir, rel_path)} ---")
            print(content)
        print(f"\n=== Asset folders ===")
        create_asset_folders(task_name, cfg, dry_run=True)
        return

    # Check if task already exists
    if os.path.exists(task_dir):
        if args.force:
            import shutil
            shutil.rmtree(task_dir)
            print(f"  Removed existing: {task_dir}/")
        else:
            print(f"ERROR: Task folder already exists: {task_dir}")
            print("       Delete it first, choose a different task_name, or use --force to overwrite.")
            sys.exit(1)

    # Create task folder and files
    os.makedirs(task_dir, exist_ok=True)

    with open(init_path, "w") as f:
        f.write(init_content)
    print(f"  Created: {init_path}")

    with open(cfg_path, "w") as f:
        f.write(cfg_content)
    print(f"  Created: {cfg_path}")

    # Create mdp/ subfolder and files
    mdp_dir = os.path.join(task_dir, "mdp")
    os.makedirs(mdp_dir, exist_ok=True)
    for rel_path, content in mdp_files.items():
        full_path = os.path.join(task_dir, rel_path)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"  Created: {full_path}")

    # Create asset subfolders
    create_asset_folders(task_name, cfg)

    print(f"\n✓ Task '{task_id}' generated in {task_dir}/")
    print(f"  Config class: {class_prefix}TaskCfg")
    print(f"\nNext steps:")
    print(f"  1. Place USD files in the asset folders listed above")
    print(f"  2. Review and customize: {cfg_path}")
    print(f"  3. Add task-specific MDP functions in: {mdp_dir}/")
    print(f"  4. Reinstall: pip install -e source/IsaacLabTaskMaker")
    print(f"  5. Run: ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task {task_id}")


if __name__ == "__main__":
    main()

