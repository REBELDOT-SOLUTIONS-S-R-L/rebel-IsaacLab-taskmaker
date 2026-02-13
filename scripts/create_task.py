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

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = ROOT_DIR / "source" / "IsaacLabTaskMaker" / "isaaclab_task_maker"
TASKS_DIR = PACKAGE_DIR / "tasks" / "manager_based"
ASSETS_DIR = PACKAGE_DIR / "assets"
TEMPLATES_DIR = PACKAGE_DIR / "templates"
TASK_DEFS_DIR = TEMPLATES_DIR / "task_definitions"

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


# ---------------------------------------------------------------------------
# Pydantic models for YAML validation
# ---------------------------------------------------------------------------
class RobotConfig(BaseModel):
    import_path: str
    config_name: str
    prim_path: str = "/World/envs/env_.*/Robot"
    init_pos: list[float] = [0.0, 0.0, 0.0]
    init_rot: list[float] = [1.0, 0.0, 0.0, 0.0]
    scale: list[float] = [1.0, 1.0, 1.0]


class SceneConfig(BaseModel):
    usd_file: str = "scene.usd"
    scale: list[float] = [1.0, 1.0, 1.0]
    init_pos: list[float] = [0.0, 0.0, 0.0]
    init_rot: list[float] = [1.0, 0.0, 0.0, 0.0]


class SceneObject(BaseModel):
    name: str
    type: str = "RigidObjectCfg"
    prim_path: str
    usd_file: Optional[str] = None  # defaults to {name}.usd
    scale: list[float] = [1.0, 1.0, 1.0]
    init_pos: list[float]
    init_rot: list[float] = [1.0, 0.0, 0.0, 0.0]


class IKControllerConfig(BaseModel):
    controller_type: str = "pink_ik"
    base_link_name: str = "base_link"
    controlled_joint_names: list[str] = []
    hand_joint_names: list[str] = []
    num_hand_joints: int = 0
    null_space_joints: list[str] = []
    # Differential IK specific
    ik_method: str = "dls"
    command_type: str = "pose"
    use_relative_mode: bool = False
    body_offset: list[float] = [0.0, 0.0, 0.0]
    # OSC specific
    nullspace_joint_pos_target: str = "zero"

    @field_validator("controller_type")
    @classmethod
    def validate_controller_type(cls, v: str) -> str:
        if v not in SUPPORTED_CONTROLLERS:
            raise ValueError(
                f"Unsupported controller_type: '{v}'. Must be one of: {SUPPORTED_CONTROLLERS}"
            )
        return v


class EEFConfig(BaseModel):
    names: list[str] = []
    target_links: dict[str, str] = {}
    frame_names: dict[str, str] = {}


class TeleopConfig(BaseModel):
    device: Optional[str] = None  # default depends on controller type
    retargeter_import: str = "isaaclab.devices"
    retargeter_class: str = "RetargeterCfg"


class SimConfig(BaseModel):
    decimation: int = 6
    episode_length_s: float = 20.0
    dt: float = 0.008333
    render_interval: int = 2
    env_spacing: float = 2.5


class TaskDefinition(BaseModel):
    task_name: str
    task_id: str
    robot: RobotConfig
    scene: SceneConfig = SceneConfig()
    scene_objects: list[SceneObject] = []
    ik_controller: IKControllerConfig = IKControllerConfig()
    eef: EEFConfig = EEFConfig()
    teleop: TeleopConfig = TeleopConfig()
    sim: SimConfig = SimConfig()
    observations: dict = {}


# ---------------------------------------------------------------------------
# Jinja2 setup
# ---------------------------------------------------------------------------
def _pylist(items: list) -> str:
    """Jinja2 filter: format Python list of strings as a list literal."""
    inner = ", ".join(f'"{item}"' for item in items)
    return f"[{inner}]"


def _pydict(d: dict) -> str:
    """Jinja2 filter: format Python dict of str→str as a dict literal."""
    inner = ", ".join(f'"{k}": "{v}"' for k, v in d.items())
    return "{" + inner + "}"


def _create_jinja_env() -> Environment:
    """Create and configure the Jinja2 template environment."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    env.filters["pylist"] = _pylist
    env.filters["pydict"] = _pydict
    return env


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase.  tm7_g1 → TM7G1"""
    return "".join(
        part.upper() if part.isalpha() and len(part) <= 3 else part.capitalize()
        for part in name.split("_")
    )


def _join_floats(values: list[float]) -> str:
    """Join a list of floats as a comma-separated string."""
    return ", ".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_context(cfg: TaskDefinition) -> dict:
    """Build the Jinja2 template context from a validated TaskDefinition."""
    class_prefix = snake_to_pascal(cfg.task_name)
    controller_type = cfg.ik_controller.controller_type

    # Resolve teleop device default (pink_ik → handtracking, others → keyboard)
    if cfg.teleop.device is None:
        teleop_device = "handtracking" if controller_type == "pink_ik" else "keyboard"
    else:
        teleop_device = cfg.teleop.device

    # First EEF body name (used by differential_ik, osc, rmpflow)
    target_links = cfg.eef.target_links
    body_name = list(target_links.values())[0] if target_links else "ee_link"

    # Pre-process scene objects
    scene_objects = []
    for obj in cfg.scene_objects:
        usd_file = obj.usd_file or f"{obj.name}.usd"
        scene_objects.append({
            "name": obj.name,
            "type": obj.type,
            "prim_path": obj.prim_path,
            "usd_file": usd_file,
            "usd_var": f"{obj.name.upper()}_USD_PATH",
            "scale": _join_floats(obj.scale),
            "init_pos": _join_floats(obj.init_pos),
            "init_rot": _join_floats(obj.init_rot),
        })

    return {
        "class_prefix": class_prefix,
        "task_name": cfg.task_name,
        "task_id": cfg.task_id,
        "controller_type": controller_type,
        "robot": {
            "import_path": cfg.robot.import_path,
            "config_name": cfg.robot.config_name,
            "prim_path": cfg.robot.prim_path,
            "init_pos": _join_floats(cfg.robot.init_pos),
            "init_rot": _join_floats(cfg.robot.init_rot),
        },
        "scene": {
            "usd_file": cfg.scene.usd_file,
            "scale": _join_floats(cfg.scene.scale),
            "pos": _join_floats(cfg.scene.init_pos),
            "rot": _join_floats(cfg.scene.init_rot),
        },
        "env_spacing": cfg.sim.env_spacing,
        "scene_objects": scene_objects,
        "ik": {
            "controlled_joint_names": cfg.ik_controller.controlled_joint_names,
            "hand_joint_names": cfg.ik_controller.hand_joint_names,
            "base_link_name": cfg.ik_controller.base_link_name,
            "num_hand_joints": cfg.ik_controller.num_hand_joints,
            "null_space_joints": cfg.ik_controller.null_space_joints,
            "ik_method": cfg.ik_controller.ik_method,
            "command_type": cfg.ik_controller.command_type,
            "use_relative_mode": cfg.ik_controller.use_relative_mode,
            "body_offset": _join_floats(cfg.ik_controller.body_offset),
            "nullspace_joint_pos_target": cfg.ik_controller.nullspace_joint_pos_target,
        },
        "eef": {
            "target_links": cfg.eef.target_links,
            "frame_names": cfg.eef.frame_names,
            "body_name": body_name,
        },
        "teleop": {
            "device": teleop_device,
            "retargeter_import": cfg.teleop.retargeter_import,
            "retargeter_class": cfg.teleop.retargeter_class,
        },
    }


# ---------------------------------------------------------------------------
# Asset folders
# ---------------------------------------------------------------------------
def create_asset_folders(cfg: TaskDefinition, dry_run: bool = False):
    """Create asset subfolders under assets/scenes/, assets/objects/."""
    folders = [ASSETS_DIR / "scenes", ASSETS_DIR / "objects", ASSETS_DIR / "robots"]
    for folder in folders:
        if dry_run:
            print(f"  Would ensure exists: {folder}/")
        else:
            folder.mkdir(parents=True, exist_ok=True)

    # Print reminders for which USD files to place
    scene_file = cfg.scene.usd_file
    print(f"\n  📁 Place your USD files:")
    print(f"     Scene:   assets/scenes/{scene_file}")
    for obj in cfg.scene_objects:
        usd_file = obj.usd_file or f"{obj.name}.usd"
        print(f"     Object:  assets/objects/{usd_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a new IL task from a YAML definition.")
    parser.add_argument("yaml_file", help="Path to the YAML task definition file.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated files instead of writing them.")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing task folder if it exists.")
    args = parser.parse_args()

    # Resolve YAML path — accept just a filename or a full path
    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_file():
        yaml_path = TASK_DEFS_DIR / args.yaml_file
    if not yaml_path.is_file():
        print(f"ERROR: YAML file not found: {args.yaml_file}")
        print(f"       Looked in: {TASK_DEFS_DIR}/")
        sys.exit(1)

    # Load and validate YAML
    raw = yaml.safe_load(yaml_path.read_text())
    try:
        cfg = TaskDefinition(**raw)
    except Exception as e:
        print(f"ERROR: Invalid task definition:\n{e}")
        sys.exit(1)

    # Set up Jinja2 and build template context
    jinja_env = _create_jinja_env()
    context = build_context(cfg)

    task_name = cfg.task_name
    class_prefix = context["class_prefix"]
    task_id = cfg.task_id
    task_dir = TASKS_DIR / task_name
    init_path = task_dir / "__init__.py"
    cfg_path = task_dir / f"{task_name}_cfg.py"

    # Render all templates
    init_content = jinja_env.get_template("init_py.py.j2").render(context)
    cfg_content = jinja_env.get_template("skeleton_task_cfg.py.j2").render(context)
    mdp_files = {
        "mdp/__init__.py": jinja_env.get_template("mdp_init.py.j2").render(context),
        "mdp/observations.py": jinja_env.get_template("mdp_observations.py.j2").render(context),
        "mdp/events.py": jinja_env.get_template("mdp_events.py.j2").render(context),
        "mdp/terminations.py": jinja_env.get_template("mdp_terminations.py.j2").render(context),
    }

    if args.dry_run:
        print(f"=== Would create: {task_dir}/ ===\n")
        print(f"--- {init_path} ---")
        print(init_content)
        print(f"--- {cfg_path} ---")
        print(cfg_content)
        for rel_path, content in mdp_files.items():
            print(f"\n--- {task_dir / rel_path} ---")
            print(content)
        print(f"\n=== Asset folders ===")
        create_asset_folders(cfg, dry_run=True)
        return

    # Check if task already exists
    if task_dir.exists():
        if args.force:
            shutil.rmtree(task_dir)
            print(f"  Removed existing: {task_dir}/")
        else:
            print(f"ERROR: Task folder already exists: {task_dir}")
            print("       Delete it first, choose a different task_name, or use --force to overwrite.")
            sys.exit(1)

    # Create task folder and write files
    task_dir.mkdir(parents=True, exist_ok=True)

    init_path.write_text(init_content)
    print(f"  Created: {init_path}")

    cfg_path.write_text(cfg_content)
    print(f"  Created: {cfg_path}")

    # Create mdp/ subfolder and files
    mdp_dir = task_dir / "mdp"
    mdp_dir.mkdir(exist_ok=True)
    for rel_path, content in mdp_files.items():
        full_path = task_dir / rel_path
        full_path.write_text(content)
        print(f"  Created: {full_path}")

    # Create asset subfolders
    create_asset_folders(cfg)

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
