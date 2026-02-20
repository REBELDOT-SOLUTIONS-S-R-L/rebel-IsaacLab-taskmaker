#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a standalone IsaacLab extension project from a YAML task definition.

The generated project is self-contained and can be installed into IsaacLab
with ``pip install -e source/<task_name>``.

Usage:
    python scripts/create_task.py example.yaml --isaaclab-path ~/isaac/IsaacLab
    python scripts/create_task.py example.yaml --isaaclab-path ~/isaac/IsaacLab --output-dir ~/my_tasks
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
# Paths (within THIS project — used to find templates and source files)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = ROOT_DIR / "source" / "IsaacLabTaskMaker" / "isaaclab_task_maker"
TEMPLATES_DIR = PACKAGE_DIR / "templates"
TASK_DEFS_DIR = TEMPLATES_DIR / "task_definitions"
BASE_IL_ENV_DIR = PACKAGE_DIR / "tasks" / "manager_based" / "base_il_env"
ASSETS_SRC_DIR = PACKAGE_DIR / "assets"

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
    """Jinja2 filter: format Python dict of str->str as a dict literal."""
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
    """Convert snake_case to PascalCase.  tm7_g1 -> TM7G1"""
    return "".join(
        part.upper() if part.isalpha() and len(part) <= 3 else part.capitalize()
        for part in name.split("_")
    )


def _join_floats(values: list[float]) -> str:
    """Join a list of floats as a comma-separated string."""
    return ", ".join(str(v) for v in values)


def _write(path: Path, content: str, dry_run: bool = False, label: str = "") -> None:
    """Write content to a file, or print in dry-run mode."""
    if dry_run:
        header = f"--- {label or path} ---"
        print(header)
        print(content)
        print()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"  Created: {path}")


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_context(cfg: TaskDefinition, package_name: str) -> dict:
    """Build the Jinja2 template context from a validated TaskDefinition."""
    class_prefix = snake_to_pascal(cfg.task_name)
    controller_type = cfg.ik_controller.controller_type

    if cfg.teleop.device is None:
        teleop_device = "handtracking" if controller_type == "pink_ik" else "keyboard"
    else:
        teleop_device = cfg.teleop.device

    target_links = cfg.eef.target_links
    body_name = list(target_links.values())[0] if target_links else "ee_link"

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

    title = f"IL Task: {class_prefix}"
    description = f"Imitation-learning task extension for {class_prefix} ({cfg.task_id})"

    return {
        "package_name": package_name,
        "project_dir_name": cfg.task_name,
        "title": title,
        "description": description,
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
# Script copying (from IsaacLab, with extension import injected)
# ---------------------------------------------------------------------------
IMPORT_LINE = "import isaaclab_tasks  # noqa: F401"
PLACEHOLDER_LINE = "# PLACEHOLDER: Extension template (do not remove this comment)"

SCRIPTS_TO_PATCH = {
    "scripts/environments/teleoperation/teleop_se3_agent.py": "scripts/teleop.py",
    "scripts/environments/random_agent.py": "scripts/random_agent.py",
    "scripts/environments/zero_agent.py": "scripts/zero_agent.py",
}


def _patch_script(content: str, package_name: str) -> str:
    """Inject ``import <package_name>`` into an IsaacLab script.

    For scripts with a PLACEHOLDER comment, replace it.
    For scripts without one (e.g. teleop), insert after ``import isaaclab_tasks``.
    """
    ext_import = f"import {package_name}  # noqa: F401"

    if PLACEHOLDER_LINE in content:
        return content.replace(PLACEHOLDER_LINE, ext_import)

    if IMPORT_LINE in content:
        return content.replace(
            IMPORT_LINE,
            f"{IMPORT_LINE}\n{ext_import}",
        )

    return content


def copy_isaaclab_scripts(
    isaaclab_path: Path,
    project_dir: Path,
    package_name: str,
    dry_run: bool = False,
) -> None:
    """Copy and patch IsaacLab scripts into the generated project."""
    for src_rel, dst_rel in SCRIPTS_TO_PATCH.items():
        src = isaaclab_path / src_rel
        dst = project_dir / dst_rel
        if not src.is_file():
            print(f"  WARNING: Script not found, skipping: {src}")
            continue
        patched = _patch_script(src.read_text(), package_name)
        _write(dst, patched, dry_run, label=dst_rel)


# ---------------------------------------------------------------------------
# Extension project generation
# ---------------------------------------------------------------------------
def generate_extension_project(
    cfg: TaskDefinition,
    output_dir: Path,
    isaaclab_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Generate a complete standalone IsaacLab extension project."""

    task_name = cfg.task_name
    package_name = task_name
    jinja_env = _create_jinja_env()
    context = build_context(cfg, package_name)
    class_prefix = context["class_prefix"]
    task_id = cfg.task_id

    # Top-level project directory
    project_dir = output_dir / task_name

    if not dry_run:
        if project_dir.exists():
            if force:
                shutil.rmtree(project_dir)
                print(f"  Removed existing: {project_dir}/")
            else:
                print(f"ERROR: Project folder already exists: {project_dir}")
                print("       Delete it first, choose a different task_name, or use --force to overwrite.")
                sys.exit(1)

    # Paths within the generated project
    ext_dir = project_dir / "source" / package_name        # source/<pkg>/
    pkg_dir = ext_dir / package_name                        # source/<pkg>/<pkg>/
    assets_dir = pkg_dir / "assets"
    base_env_dst = pkg_dir / "base_il_env"
    tasks_dir = pkg_dir / "tasks"
    mb_dir = tasks_dir / "manager_based"
    task_dir = mb_dir / task_name
    mdp_dir = task_dir / "mdp"

    if dry_run:
        print(f"=== Would generate extension project at: {project_dir}/ ===\n")

    # ----- 1) Root project files -----
    print("  [1/7] Project scaffold...")
    _write(
        project_dir / "pyproject.toml",
        jinja_env.get_template("extension/pyproject_toml.j2").render(context),
        dry_run,
    )
    _write(
        project_dir / "README.md",
        jinja_env.get_template("extension/readme_md.j2").render(context),
        dry_run,
    )

    # ----- 2) Extension package files (source/<pkg>/) -----
    print("  [2/7] Extension package files...")
    _write(
        ext_dir / "config" / "extension.toml",
        jinja_env.get_template("extension/extension_toml.j2").render(context),
        dry_run,
    )
    _write(
        ext_dir / "setup.py",
        jinja_env.get_template("extension/setup_py.j2").render(context),
        dry_run,
    )
    _write(
        ext_dir / "pyproject.toml",
        jinja_env.get_template("extension/pyproject_toml.j2").render(context),
        dry_run,
    )
    _write(
        ext_dir / f"{package_name}_register.pth",
        f"import {package_name}\n",
        dry_run,
    )

    # ----- 3) Python package root (source/<pkg>/<pkg>/) -----
    print("  [3/7] Python package...")
    _write(
        pkg_dir / "__init__.py",
        jinja_env.get_template("extension/ext_init_py.j2").render(context),
        dry_run,
    )
    _write(
        pkg_dir / "ui_extension_example.py",
        jinja_env.get_template("extension/ui_extension_example_py.j2").render(context),
        dry_run,
    )

    # ----- 4) Assets module -----
    print("  [4/7] Assets module...")
    _write(
        assets_dir / "__init__.py",
        jinja_env.get_template("extension/assets_init_py.j2").render(context),
        dry_run,
    )
    if not dry_run:
        for subdir in ("scenes", "objects", "robots"):
            (assets_dir / subdir).mkdir(parents=True, exist_ok=True)
            print(f"  Created: {assets_dir / subdir}/")
    else:
        print(f"  Would create subdirs: scenes/, objects/, robots/ in {assets_dir}/")

    # ----- 5) Copy base_il_env -----
    print("  [5/7] Base IL environment...")
    if dry_run:
        print(f"  Would copy {BASE_IL_ENV_DIR}/ -> {base_env_dst}/")
    else:
        shutil.copytree(BASE_IL_ENV_DIR, base_env_dst)
        print(f"  Copied:  {base_env_dst}/")

    # ----- 6) Task files -----
    print("  [6/7] Task files...")
    _write(
        tasks_dir / "__init__.py",
        jinja_env.get_template("extension/tasks_init_py.j2").render(context),
        dry_run,
    )
    _write(
        mb_dir / "__init__.py",
        "import gymnasium as gym  # noqa: F401\n",
        dry_run,
    )

    # Task-specific files (gym registration, config, MDP)
    _write(
        task_dir / "__init__.py",
        jinja_env.get_template("init_py.py.j2").render(context),
        dry_run,
    )
    _write(
        task_dir / f"{task_name}_cfg.py",
        jinja_env.get_template("skeleton_task_cfg.py.j2").render(context),
        dry_run,
    )
    _write(
        mdp_dir / "__init__.py",
        jinja_env.get_template("mdp_init.py.j2").render(context),
        dry_run,
    )
    _write(
        mdp_dir / "observations.py",
        jinja_env.get_template("mdp_observations.py.j2").render(context),
        dry_run,
    )
    _write(
        mdp_dir / "events.py",
        jinja_env.get_template("mdp_events.py.j2").render(context),
        dry_run,
    )
    _write(
        mdp_dir / "terminations.py",
        jinja_env.get_template("mdp_terminations.py.j2").render(context),
        dry_run,
    )

    # ----- 7) Copy & patch IsaacLab scripts -----
    if isaaclab_path is not None:
        print("  [7/7] IsaacLab scripts (with extension import)...")
        copy_isaaclab_scripts(isaaclab_path, project_dir, package_name, dry_run)
    else:
        print("  [7/7] Skipped IsaacLab scripts (--isaaclab-path not provided)")

    # ----- Summary -----
    scripts_dir = project_dir / "scripts"
    has_scripts = isaaclab_path is not None

    print(f"\n{'=' * 60}")
    print(f"Extension project '{task_id}' generated at:")
    print(f"  {project_dir}/")
    print(f"{'=' * 60}")
    print(f"  Config class: {class_prefix}TaskCfg")
    print(f"  Package name: {package_name}")

    scene_file = cfg.scene.usd_file
    print(f"\n  Place your USD files:")
    print(f"     Scene:   {assets_dir}/scenes/{scene_file}")
    for obj in cfg.scene_objects:
        usd_file = obj.usd_file or f"{obj.name}.usd"
        print(f"     Object:  {assets_dir}/objects/{usd_file}")

    print(f"\nNext steps:")
    print(f"  1. Place USD files in the asset folders listed above")
    print(f"  2. Review and customize: {task_dir / f'{task_name}_cfg.py'}")
    print(f"  3. Add task-specific MDP functions in: {mdp_dir}/")
    print(f"  4. Install: pip install -e {ext_dir}")
    if has_scripts:
        print(f"  5. Run: ./isaaclab.sh -p {scripts_dir / 'teleop.py'} --task {task_id}")
    else:
        print(f"  5. Run: ./isaaclab.sh -p {project_dir}/scripts/teleop.py --task {task_id}")
        print(f"\n  NOTE: Scripts were not generated because --isaaclab-path was not provided.")
        print(f"        Re-run with --isaaclab-path /path/to/IsaacLab to include launch scripts,"  )
        print(f"        or copy IsaacLab scripts manually and add: import {package_name}  # noqa: F401")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone IsaacLab IL extension from a YAML task definition."
    )
    parser.add_argument("yaml_file", help="Path to the YAML task definition file.")
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory where the extension project will be created (default: current directory).",
    )
    parser.add_argument(
        "--isaaclab-path",
        default=None,
        help=(
            "Path to IsaacLab installation. When provided, launch scripts (teleop, "
            "random_agent, zero_agent) are copied into the generated project with the "
            "extension import pre-configured. Falls back to ISAACLAB_PATH env var."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated files instead of writing them.")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing project if it exists.")
    args = parser.parse_args()

    # Resolve YAML path
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

    # Resolve IsaacLab path
    import os
    isaaclab_path_str = args.isaaclab_path or os.environ.get("ISAACLAB_PATH")
    isaaclab_path = Path(isaaclab_path_str).resolve() if isaaclab_path_str else None
    if isaaclab_path and not (isaaclab_path / "isaaclab.sh").is_file():
        print(f"WARNING: --isaaclab-path does not look like an IsaacLab installation: {isaaclab_path}")
        print(f"         (isaaclab.sh not found). Scripts will not be copied.")
        isaaclab_path = None

    output_dir = Path(args.output_dir).resolve()
    generate_extension_project(
        cfg, output_dir, isaaclab_path=isaaclab_path, dry_run=args.dry_run, force=args.force,
    )


if __name__ == "__main__":
    main()
