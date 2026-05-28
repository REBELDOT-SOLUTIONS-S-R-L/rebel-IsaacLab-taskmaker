#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a standalone IsaacLab extension project from a YAML task definition.

The generated project is self-contained and can be installed into IsaacLab
with ``pip install -e source/<task_name>``. The install writes a ``.pth`` file
that auto-imports the extension package on Python startup, so IsaacLab's stock
launch scripts (teleop_se3_agent, record_annotated_demos, random_agent,
zero_agent) can find the registered gym envs without any per-task customization.

Usage:
    python scripts/create_task.py example.yaml
    python scripts/create_task.py example.yaml --output-dir ~/my_tasks
    python scripts/create_task.py example.yaml --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Literal, Optional

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
BASE_IL_ENV_DIR = TEMPLATES_DIR / "base_il_env"
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
    name: str = "robot"
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


class PhysicsMaterialConfig(BaseModel):
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0


class ObjectResetConfig(BaseModel):
    """Optional per-object reset randomization emitted as EventTerm entries."""

    sampler: Literal["uniform", "sobol"] = "uniform"
    seed: Optional[int] = None
    pose_range: dict[str, tuple[float, float]] = {}
    pos_range: dict[str, tuple[float, float]] = {}
    rot_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}


class SceneObject(BaseModel):
    name: str
    type: str = "RigidObjectCfg"
    prim_path: str
    spawn: bool = True
    usd_file: Optional[str] = None  # defaults to {name}.usd
    scale: list[float] = [1.0, 1.0, 1.0]
    init_pos: list[float]
    init_rot: list[float] = [1.0, 0.0, 0.0, 0.0]
    # When True, the spawner overrides the mesh collision approximation with SDF.
    # When False, the collision approximation baked into the USD is preserved.
    use_default_sdf_collision: bool = True
    # Optional per-object physics material. If None, no material override is applied
    # and the object inherits the simulation's default friction/restitution.
    physics_material: Optional[PhysicsMaterialConfig] = None
    # Optional color/look name under the scene USD's Looks scope. When set, the
    # generated spawner binds ``{ENV_NS}/Scene/Looks/<visual_material>`` to the object.
    visual_material: Optional[str] = None
    # Optional reset randomization. When omitted, the object resets to init_state
    # through reset_scene_to_default only.
    reset: Optional[ObjectResetConfig] = None


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
    # Optional per-EEF prefix used to assign hand joints to arms when
    # `ik_controller.hand_joint_names` interleaves them. Required for multi-EEF
    # pink_ik tasks with hand joints (e.g. {"left": "L_", "right": "R_"} for the
    # Inspire hand). Unused for single-EEF tasks.
    hand_joint_prefixes: dict[str, str] = {}
    # Optional fixed child links inserted into the spawned robot USD and into
    # the URDF generated for Pink IK. Useful when the controller should track a
    # task-local frame that does not exist in the source robot asset.
    virtual_links: dict[str, "VirtualEEFLinkConfig"] = {}


class VirtualEEFLinkConfig(BaseModel):
    link_name: str
    usd_parent_path: str
    urdf_frame_name: str
    urdf_parent_link: str
    urdf_parent_fallback: Optional[str] = None
    offset: list[float] = [0.0, 0.0, 0.0]


class CameraConfig(BaseModel):
    """Pinhole camera attached to a robot link.

    Emitted as ``scene.<name>`` and also re-attached after the XR pipeline strips
    cameras (when running pink_ik teleop). Mirrors the upstream ``CameraCfg``
    surface — only the fields commonly tweaked per-task are exposed here.
    """

    name: str
    prim_path: str
    update_period: float = 0.0
    height: int = 480
    width: int = 640
    data_types: list[str] = ["rgb"]
    focal_length: float = 12.0
    focus_distance: float = 400.0
    horizontal_aperture: float = 20.955
    clipping_range: list[float] = [0.05, 10.0]
    offset_pos: list[float] = [0.0, 0.0, 0.0]
    offset_rot: list[float] = [1.0, 0.0, 0.0, 0.0]
    convention: str = "opengl"


class TeleopConfig(BaseModel):
    device: Optional[str] = None  # default depends on controller type
    retargeter_import: str = "isaaclab.devices"
    retargeter_class: str = "RetargeterCfg"
    # OpenXR anchor placement (pink_ik tasks only). Positions the operator's
    # tracking-space origin relative to the world frame so their hands land near
    # the robot's wrists. Both default to identity for backward compatibility,
    # but anything other than a robot sitting at (0, 0, 0) needs these set.
    xr_anchor_pos: list[float] = [0.0, 0.0, 0.0]
    xr_anchor_rot: list[float] = [1.0, 0.0, 0.0, 0.0]


class SimConfig(BaseModel):
    decimation: int = 6
    episode_length_s: float = 20.0
    dt: float = 0.008333
    render_interval: int = 2
    env_spacing: float = 2.5


class ResetDefaultsConfig(BaseModel):
    """Global defaults for generated reset EventTerms."""

    seed: int = 0
    sobol_advance_on_success_only: bool = False


class TaskDefinition(BaseModel):
    task_name: str
    task_id: str
    robot: RobotConfig
    scene: SceneConfig = SceneConfig()
    scene_objects: list[SceneObject] = []
    cameras: list[CameraConfig] = []
    ik_controller: IKControllerConfig = IKControllerConfig()
    eef: EEFConfig = EEFConfig()
    teleop: TeleopConfig = TeleopConfig()
    sim: SimConfig = SimConfig()
    observations: dict = {}
    resets: ResetDefaultsConfig = ResetDefaultsConfig()


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


def _pyrepr(value: Any) -> str:
    """Jinja2 filter: format an arbitrary YAML-derived value as a Python literal.

    Lists become Python lists, tuples stay tuples, dicts become dict literals,
    None becomes ``None``, bools become ``True``/``False``.
    """
    return repr(value)


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
    env.filters["pyrepr"] = _pyrepr
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


def _cleanup_installed_artifacts(package_name: str) -> None:
    """Remove pip package and the .pth file from site-packages."""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", package_name],
        capture_output=True,
    )
    pth_filename = f"{package_name}_register.pth"
    purelib = sysconfig.get_path("purelib")
    if purelib:
        pth_path = Path(purelib) / pth_filename
        if pth_path.is_file():
            pth_path.unlink()
            print(f"  Removed stale .pth: {pth_path}")


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
        leaf = obj.prim_path.rstrip("/").rsplit("/", 1)[-1]
        physics_material = None
        if obj.physics_material is not None:
            physics_material = {
                "static_friction": obj.physics_material.static_friction,
                "dynamic_friction": obj.physics_material.dynamic_friction,
                "restitution": obj.physics_material.restitution,
            }
        scene_objects.append({
            "name": obj.name,
            "type": obj.type,
            "prim_path": obj.prim_path,
            "leaf": leaf,
            "spawn": obj.spawn,
            "usd_file": usd_file,
            "usd_var": f"{obj.name.upper()}_USD_PATH",
            "scale": _join_floats(obj.scale),
            "init_pos": _join_floats(obj.init_pos),
            "init_rot": _join_floats(obj.init_rot),
            "use_default_sdf_collision": obj.use_default_sdf_collision,
            "physics_material": physics_material,
            "visual_material": obj.visual_material,
            "reset": _build_object_reset_context(obj, cfg.resets.seed),
        })

    has_sdf_objects = any(o["spawn"] and o["use_default_sdf_collision"] for o in scene_objects)
    has_baked_collision_objects = any(o["spawn"] and not o["use_default_sdf_collision"] for o in scene_objects)
    has_physics_materials = any(o["spawn"] and o["physics_material"] is not None for o in scene_objects)
    has_visual_materials = any(o["spawn"] and o["visual_material"] is not None for o in scene_objects)
    reset_events = _build_reset_events_context(
        scene_objects,
        cfg.resets.sobol_advance_on_success_only,
    )

    eef_slices = _build_eef_slices(cfg)
    cameras = _build_cameras_context(cfg)

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
            "name": cfg.robot.name,
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
        "sim": {
            "decimation": cfg.sim.decimation,
            "episode_length_s": cfg.sim.episode_length_s,
            "dt": cfg.sim.dt,
            "render_interval": cfg.sim.render_interval,
            "env_spacing": cfg.sim.env_spacing,
        },
        "env_spacing": cfg.sim.env_spacing,
        "scene_objects": scene_objects,
        "reset_events": reset_events,
        "cameras": cameras,
        "has_sdf_objects": has_sdf_objects,
        "has_baked_collision_objects": has_baked_collision_objects,
        "has_physics_materials": has_physics_materials,
        "has_visual_materials": has_visual_materials,
        "eef_slices": eef_slices,
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
            "names": cfg.eef.names,
            "target_links": cfg.eef.target_links,
            "frame_names": cfg.eef.frame_names,
            "body_name": body_name,
            "virtual_links": [
                {
                    "name": name,
                    "link_name": link.link_name,
                    "usd_parent_path": link.usd_parent_path,
                    "urdf_frame_name": link.urdf_frame_name,
                    "urdf_parent_link": link.urdf_parent_link,
                    "urdf_parent_fallback": link.urdf_parent_fallback,
                    "offset": tuple(link.offset),
                    "offset_literal": _py_value(link.offset),
                }
                for name, link in cfg.eef.virtual_links.items()
            ],
        },
        "observations": {
            # Mapping of eef_name -> robot link name. Drives EEF obs term generation.
            # Falls back to {} if the YAML doesn't declare it.
            "eef_link_names": (cfg.observations or {}).get("eef_link_names", {}),
        },
        "teleop": {
            "device": teleop_device,
            "retargeter_import": cfg.teleop.retargeter_import,
            "retargeter_class": cfg.teleop.retargeter_class,
            "xr_anchor_pos": _join_floats(cfg.teleop.xr_anchor_pos),
            "xr_anchor_rot": _join_floats(cfg.teleop.xr_anchor_rot),
        },
        "mimic_task_id": f"{cfg.task_id}-Mimic",
    }


def _build_eef_slices(cfg: TaskDefinition) -> Optional[dict]:
    """Compute the per-EEF action/gripper tensor layout for the BaseILEnv.

    Returns a dict consumed by ``skeleton_task_cfg.py.j2`` to emit
    ``self.eef_action_slices`` and ``self.eef_gripper_slices`` inside the
    generated ``__post_init__``. Returns ``None`` for joint-space controllers,
    which don't have pose-shaped actions for BaseILEnv to slice.
    """
    controller_type = cfg.ik_controller.controller_type
    eef_names = list(cfg.eef.names)
    if not eef_names:
        return None

    if controller_type == "pink_ik":
        # Action tensor layout (per `PinkInverseKinematicsActionCfg`):
        #     [pos(3), quat(4)] per EEF (in `frame_names` order), then `hand_joint_names`.
        action = {}
        for i, name in enumerate(eef_names):
            base = i * 7
            action[name] = {"pos": (base, base + 3), "quat": (base + 3, base + 7)}

        pose_offset = len(eef_names) * 7
        hand_joint_names = cfg.ik_controller.hand_joint_names

        if not hand_joint_names:
            return {"action": action, "gripper_mode": None}

        if len(eef_names) == 1:
            name = eef_names[0]
            return {
                "action": action,
                "gripper_mode": "contiguous",
                "gripper": {name: (pose_offset, pose_offset + len(hand_joint_names))},
            }

        prefixes = cfg.eef.hand_joint_prefixes
        if prefixes:
            missing = [n for n in eef_names if n not in prefixes]
            if missing:
                raise ValueError(
                    f"eef.hand_joint_prefixes is missing entries for {missing}; "
                    f"multi-EEF pink_ik tasks need a prefix per arm."
                )
            return {
                "action": action,
                "gripper_mode": "prefix",
                "pose_offset": pose_offset,
                "num_eefs": len(eef_names),
                "prefixes": {name: prefixes[name] for name in eef_names},
            }

        # Multi-EEF without prefixes — fall back to splitting evenly and warn
        # in the generated code via a TODO comment.
        per_arm = len(hand_joint_names) // len(eef_names)
        gripper = {
            name: (pose_offset + i * per_arm, pose_offset + (i + 1) * per_arm)
            for i, name in enumerate(eef_names)
        }
        return {
            "action": action,
            "gripper_mode": "contiguous_fallback",
            "gripper": gripper,
        }

    if controller_type in ("differential_ik", "operational_space", "rmpflow"):
        # Single-arm pose-space controllers: [pos(3), quat(4)] then optional
        # contiguous gripper joints.
        if len(eef_names) != 1:
            # Multi-arm not supported for these controllers in the action cfg
            # block above — skip emitting slices.
            return None
        name = eef_names[0]
        action = {name: {"pos": (0, 3), "quat": (3, 7)}}
        hand_joint_names = cfg.ik_controller.hand_joint_names
        if not hand_joint_names:
            return {"action": action, "gripper_mode": None}
        return {
            "action": action,
            "gripper_mode": "contiguous",
            "gripper": {name: (7, 7 + len(hand_joint_names))},
        }

    # joint_position / relative_joint_position / joint_velocity / joint_effort
    # operate directly in joint space — no pose slices to emit.
    return None


def _build_cameras_context(cfg: TaskDefinition) -> list[dict]:
    """Render cameras into a Jinja-friendly form (floats joined, lists kept)."""
    out: list[dict] = []
    for cam in cfg.cameras:
        out.append({
            "name": cam.name,
            "prim_path": cam.prim_path,
            "update_period": cam.update_period,
            "height": cam.height,
            "width": cam.width,
            "data_types": list(cam.data_types),
            "focal_length": cam.focal_length,
            "focus_distance": cam.focus_distance,
            "horizontal_aperture": cam.horizontal_aperture,
            "clipping_range": _join_floats(cam.clipping_range),
            "offset_pos": _join_floats(cam.offset_pos),
            "offset_rot": _join_floats(cam.offset_rot),
            "convention": cam.convention,
        })
    return out


def _py_value(value: Any) -> str:
    """Render a YAML-derived value as a Python literal.

    Numeric lists are upgraded to tuples so the rendered literals slot into
    ``tuple[float, ...]`` typed slots (e.g. reset pose ranges) without
    extra coercion in the generated code.
    """
    if isinstance(value, list) and value:
        inner_types = {type(v) for v in value}
        if inner_types <= {int, float} and inner_types != {bool}:
            return repr(tuple(value))
    return repr(value)


_POSE_KEY_ORDER = ("x", "y", "z", "roll", "pitch", "yaw")
_POSITION_KEYS = {"x", "y", "z"}
_ROTATION_KEY_ALIASES = {
    "rx": "roll",
    "ry": "pitch",
    "rz": "yaw",
    "roll": "roll",
    "pitch": "pitch",
    "yaw": "yaw",
}


def _normalize_reset_range_value(value: tuple[float, float], path: str) -> tuple[float, float]:
    """Validate and normalize a two-element reset range."""
    if len(value) != 2:
        raise ValueError(f"{path} must contain exactly two values: [min, max].")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError(f"{path} has max < min: {value}.")
    return (lo, hi)


def _add_pose_range_key(
    out: dict[str, tuple[float, float]],
    key: str,
    value: tuple[float, float],
    path: str,
) -> None:
    """Add a canonical pose randomization range, rejecting typos and duplicates."""
    if key not in _POSE_KEY_ORDER:
        raise ValueError(
            f"{path} uses unsupported pose key '{key}'. "
            "Expected one of: x, y, z, roll, pitch, yaw."
        )
    if key in out:
        raise ValueError(f"{path} duplicates pose key '{key}'.")
    out[key] = _normalize_reset_range_value(value, path)


def _build_object_reset_context(
    obj: SceneObject,
    global_seed: int,
) -> Optional[dict[str, Any]]:
    """Convert an optional object reset block into template-friendly values."""
    if obj.reset is None:
        return None
    if obj.type != "RigidObjectCfg":
        raise ValueError(
            f"scene_objects.{obj.name}.reset requires type: RigidObjectCfg; "
            f"got {obj.type!r}."
        )

    pose_range: dict[str, tuple[float, float]] = {}

    for key, value in obj.reset.pose_range.items():
        canonical = _ROTATION_KEY_ALIASES.get(key, key)
        _add_pose_range_key(pose_range, canonical, value, f"scene_objects.{obj.name}.reset.pose_range.{key}")

    for key, value in obj.reset.pos_range.items():
        if key not in _POSITION_KEYS:
            raise ValueError(
                f"scene_objects.{obj.name}.reset.pos_range.{key} is invalid; "
                "expected one of: x, y, z."
            )
        _add_pose_range_key(pose_range, key, value, f"scene_objects.{obj.name}.reset.pos_range.{key}")

    for key, value in obj.reset.rot_range.items():
        canonical = _ROTATION_KEY_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"scene_objects.{obj.name}.reset.rot_range.{key} is invalid; "
                "expected one of: rx, ry, rz, roll, pitch, yaw."
            )
        _add_pose_range_key(pose_range, canonical, value, f"scene_objects.{obj.name}.reset.rot_range.{key}")

    velocity_range = {
        key: _normalize_reset_range_value(value, f"scene_objects.{obj.name}.reset.velocity_range.{key}")
        for key, value in obj.reset.velocity_range.items()
    }

    ordered_pose_range = {key: pose_range[key] for key in _POSE_KEY_ORDER if key in pose_range}
    seed = global_seed if obj.reset.seed is None else obj.reset.seed

    return {
        "sampler": obj.reset.sampler,
        "seed": seed,
        "pose_range": ordered_pose_range,
        "pose_range_literal": _py_value(ordered_pose_range),
        "velocity_range": velocity_range,
        "velocity_range_literal": _py_value(velocity_range),
    }


def _build_reset_events_context(
    scene_objects: list[dict[str, Any]],
    sobol_advance_on_success_only: bool = False,
) -> dict[str, Any]:
    """Group object reset EventTerms by sampler.

    Sobol resets are emitted as one joint EventTerm so the Sobol engine covers
    the concatenated pose dimensions for all participating assets.
    """
    uniform_objects = [
        obj for obj in scene_objects
        if obj.get("reset") is not None and obj["reset"]["sampler"] == "uniform"
    ]
    sobol_objects = [
        obj for obj in scene_objects
        if obj.get("reset") is not None and obj["reset"]["sampler"] == "sobol"
    ]
    default_objects = [
        obj for obj in scene_objects
        if obj.get("reset") is None and obj["type"] == "RigidObjectCfg"
    ]
    has_object_resets = bool(uniform_objects or sobol_objects)

    sobol_seed = 0
    if sobol_objects:
        seeds = {obj["reset"]["seed"] for obj in sobol_objects}
        if len(seeds) != 1:
            raise ValueError(
                "Sobol object resets use one joint Sobol engine, so all "
                "scene_objects.*.reset.seed values must match."
            )
        sobol_seed = next(iter(seeds))

    return {
        "has_object_resets": has_object_resets,
        "default": default_objects if has_object_resets else [],
        "uniform": uniform_objects,
        "sobol": sobol_objects,
        "sobol_seed": sobol_seed,
        "sobol_advance_on_success_only": sobol_advance_on_success_only,
    }


# ---------------------------------------------------------------------------
# Extension project generation
# ---------------------------------------------------------------------------
def generate_extension_project(
    cfg: TaskDefinition,
    output_dir: Path,
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
                _cleanup_installed_artifacts(package_name)
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
    print("  [1/6] Project scaffold...")
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
    print("  [2/6] Extension package files...")
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
    print("  [3/6] Python package...")
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
    print("  [4/6] Assets module...")
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

    # ----- 5) Render / copy base_il_env -----
    print("  [5/6] Base IL environment...")
    for src_file in sorted(BASE_IL_ENV_DIR.rglob("*")):
        if not src_file.is_file():
            continue
        rel_path = src_file.relative_to(BASE_IL_ENV_DIR)
        if src_file.suffix == ".j2":
            template_name = (Path("base_il_env") / rel_path).as_posix()
            rendered = jinja_env.get_template(template_name).render(context)
            dst_file = base_env_dst / rel_path.with_suffix("")
            _write(dst_file, rendered, dry_run)
        else:
            dst_file = base_env_dst / rel_path
            if dry_run:
                print(f"  Would copy {src_file} -> {dst_file}")
            else:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"  Copied: {dst_file}")

    # ----- 6) Task files -----
    print("  [6/6] Task files...")
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
        task_dir / f"{task_name}_mimic_cfg.py",
        jinja_env.get_template("mimic_env_cfg.py.j2").render(context),
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

    # ----- Summary -----
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
        if not obj.spawn:
            continue
        usd_file = obj.usd_file or f"{obj.name}.usd"
        print(f"     Object:  {assets_dir}/objects/{usd_file}")

    print(f"\nNext steps:")
    print(f"  1. Place USD files in the asset folders listed above")
    print(f"  2. Review and customize: {task_dir / f'{task_name}_cfg.py'}")
    print(f"  3. Add task-specific MDP functions in: {mdp_dir}/")
    print(f"  4. Install: pip install -e {ext_dir}")
    print(f"     (the editable install writes a .pth that auto-imports {package_name},")
    print(f"     so IsaacLab's stock scripts can find the registered gym envs.)")
    print(f"  5. Run, e.g.:")
    print(f"     ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task {task_id}")
    print(f"     ./isaaclab.sh -p scripts/tools/record_annotated_demos.py --task {context['mimic_task_id']}")
    print(f"  6. Fill in task-specific subtask_configs in: {task_dir / f'{task_name}_mimic_cfg.py'}")


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

    output_dir = Path(args.output_dir).resolve()
    generate_extension_project(
        cfg, output_dir, dry_run=args.dry_run, force=args.force,
    )


if __name__ == "__main__":
    main()
