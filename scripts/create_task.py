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
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict, field_validator


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


class PhysicsMaterialConfig(BaseModel):
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0


class SceneObject(BaseModel):
    name: str
    type: str = "RigidObjectCfg"
    prim_path: str
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


class SubTaskSpec(BaseModel):
    """A single SubTaskConfig entry. Extra keys are forwarded to SubTaskConfig as-is."""

    model_config = ConfigDict(extra="allow")

    object_ref: Optional[str] = None
    subtask_term_signal: Optional[str] = None


class SubtaskTermSpec(BaseModel):
    """Declarative binding for a subtask_term_signal predicate.

    Lets multiple signals share one MDP function differentiated by params (so
    a single ``grasp_brick_done`` can back both ``grasp_brick_left`` and
    ``grasp_brick_right`` ObsTerms). If ``func`` is omitted it defaults to the
    signal name, preserving the one-function-per-signal stub layout.
    """

    func: Optional[str] = None
    params: dict[str, Any] = {}


class MimicConfig(BaseModel):
    """Optional Mimic data-generation configuration.

    When present, create_task.py generates a `<task_name>_mimic_cfg.py` alongside
    the task cfg and registers a sibling `<task_id>-Mimic` gym env.
    """

    model_config = ConfigDict(extra="forbid")

    # When None, the mimic task_id defaults to `<task_id>-Mimic` in the context builder.
    mimic_task_id: Optional[str] = None
    # Free-form passthrough into self.datagen_config.<k> = <v>. Only keys present
    # in this dict are emitted, so users can stay close to upstream IsaacLab examples
    # without us hard-coding the full DataGenConfig schema.
    datagen: dict[str, Any] = {}
    # eef_name -> ordered list of SubTaskConfig entries.
    subtasks: dict[str, list[SubTaskSpec]] = {}


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
    mimic: Optional[MimicConfig] = None
    # Optional binding from subtask_term_signal name to a (function, params)
    # pair. Used to flesh out the SubtaskTermsCfg ObsTerms and the MDP stub
    # signatures. When omitted, each declared signal gets its own zero-stub
    # function and an ObsTerm with only the signal_name param.
    subtask_terms: dict[str, SubtaskTermSpec] = {}


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
    None becomes ``None``, bools become ``True``/``False``. Used to forward
    Mimic config values verbatim into the generated Python.
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
            "usd_file": usd_file,
            "usd_var": f"{obj.name.upper()}_USD_PATH",
            "scale": _join_floats(obj.scale),
            "init_pos": _join_floats(obj.init_pos),
            "init_rot": _join_floats(obj.init_rot),
            "use_default_sdf_collision": obj.use_default_sdf_collision,
            "physics_material": physics_material,
        })

    has_sdf_objects = any(o.use_default_sdf_collision for o in cfg.scene_objects)
    has_baked_collision_objects = any(not o.use_default_sdf_collision for o in cfg.scene_objects)
    has_physics_materials = any(o.physics_material is not None for o in cfg.scene_objects)

    eef_slices = _build_eef_slices(cfg)
    cameras = _build_cameras_context(cfg)
    subtask_terms = _build_subtask_signal_context(cfg)

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
        "sim": {
            "decimation": cfg.sim.decimation,
            "episode_length_s": cfg.sim.episode_length_s,
            "dt": cfg.sim.dt,
            "render_interval": cfg.sim.render_interval,
            "env_spacing": cfg.sim.env_spacing,
        },
        "env_spacing": cfg.sim.env_spacing,
        "scene_objects": scene_objects,
        "cameras": cameras,
        "has_sdf_objects": has_sdf_objects,
        "has_baked_collision_objects": has_baked_collision_objects,
        "has_physics_materials": has_physics_materials,
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
        "subtask_terms": subtask_terms,
        "mimic": _build_mimic_context(cfg, class_prefix),
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


def _infer_param_type(value: Any) -> Optional[str]:
    """Best-effort Python type annotation for a YAML-derived value.

    Returns ``None`` when no useful annotation can be inferred — the generator
    then emits the parameter without a type hint, which the user can refine.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        inner_types = {type(v) for v in value}
        if inner_types == {bool}:
            return "tuple[bool, ...]"
        if inner_types <= {int, float} and inner_types != {bool}:
            return "tuple[float, ...]"
        if inner_types == {str}:
            return "tuple[str, ...]"
        return None
    return None


def _py_value(value: Any) -> str:
    """Render a YAML-derived value as a Python literal.

    Numeric lists are upgraded to tuples so they line up with the
    ``tuple[float, ...]`` annotation produced by ``_infer_param_type``.
    """
    if isinstance(value, list) and value:
        inner_types = {type(v) for v in value}
        if inner_types <= {int, float} and inner_types != {bool}:
            return repr(tuple(value))
    return repr(value)


def _build_subtask_signal_context(cfg: TaskDefinition) -> dict[str, Any]:
    """Build the subtask-signal context block for the templates.

    Output shape::

        {
            "signals": [
                {
                    "name": "grasp_brick_left",
                    "func": "grasp_brick_done",
                    "params": [("eef_link", "'left_wrist_yaw_link'"), ...],
                },
                ...
            ],
            "funcs": [
                {
                    "name": "grasp_brick_done",
                    "params": [{"name": "eef_link", "type": "str"}, ...],
                },
                ...
            ],
        }

    ``signals`` is the unique ordered list of ``subtask_term_signal`` values
    declared on ``mimic.subtasks`` (terminal ``None`` entries are skipped).
    For each signal, ``params`` are pre-rendered ``(key, py_literal)`` pairs
    that include ``signal_name`` plus any user-supplied entries from
    ``subtask_terms``.

    ``funcs`` is the unique set of underlying MDP function names; each carries
    the merged set of parameter names with inferred types so the stub
    generator can emit a single function whose signature accepts every param
    used by any signal pointing at it.
    """
    if cfg.mimic is None:
        return {"signals": [], "funcs": []}

    ordered_signals: list[str] = []
    seen: set[str] = set()
    for entries in cfg.mimic.subtasks.values():
        for entry in entries:
            sig = entry.subtask_term_signal
            if sig and sig not in seen:
                seen.add(sig)
                ordered_signals.append(sig)

    signals_out: list[dict[str, Any]] = []
    func_params: dict[str, dict[str, Optional[str]]] = {}
    func_order: list[str] = []

    for sig in ordered_signals:
        binding = cfg.subtask_terms.get(sig)
        func_name = (binding.func if binding and binding.func else sig)
        user_params = (binding.params if binding else {}) or {}

        signal_params: list[tuple[str, str]] = [("signal_name", repr(sig))]
        for k, v in user_params.items():
            signal_params.append((k, _py_value(v)))
        signals_out.append({
            "name": sig,
            "func": func_name,
            "params": signal_params,
        })

        if func_name not in func_params:
            func_params[func_name] = {}
            func_order.append(func_name)
        merged = func_params[func_name]
        for k, v in user_params.items():
            inferred = _infer_param_type(v)
            existing = merged.get(k)
            # First declaration wins; later signals only fill in a type that
            # was previously unknown.
            if k not in merged:
                merged[k] = inferred
            elif existing is None and inferred is not None:
                merged[k] = inferred

    funcs_out: list[dict[str, Any]] = []
    for fn in func_order:
        params_def: list[dict[str, Optional[str]]] = [
            {"name": name, "type": ty} for name, ty in func_params[fn].items()
        ]
        funcs_out.append({"name": fn, "params": params_def})

    return {"signals": signals_out, "funcs": funcs_out}


# SubTaskConfig fields that are declared as tuples upstream. YAML reads them as
# lists, so coerce them so the generated Python keeps the tuple syntax that
# matches the upstream IsaacLab examples.
_TUPLE_SUBTASK_FIELDS = (
    "first_subtask_start_offset_range",
    "subtask_start_offset_range",
    "subtask_term_offset_range",
)


def _build_mimic_context(cfg: TaskDefinition, class_prefix: str) -> Optional[dict]:
    """Build the Mimic section of the Jinja context, or return None if no mimic config."""
    if cfg.mimic is None:
        return None

    mimic_task_id = cfg.mimic.mimic_task_id or f"{cfg.task_id}-Mimic"

    # Render datagen kv pairs as (key, python-literal) so the template can emit
    # `self.datagen_config.<k> = <v>` without needing per-field knowledge.
    datagen_items = [(k, repr(v)) for k, v in cfg.mimic.datagen.items()]

    subtasks: dict[str, list[list[tuple[str, str]]]] = {}
    for eef_name, entries in cfg.mimic.subtasks.items():
        rendered_entries: list[list[tuple[str, str]]] = []
        for entry in entries:
            raw = entry.model_dump()
            # Drop unset Nones so the generated code only mentions fields the
            # user actually specified.
            kv: list[tuple[str, str]] = []
            for k, v in raw.items():
                if v is None and k not in ("object_ref", "subtask_term_signal"):
                    continue
                if k in _TUPLE_SUBTASK_FIELDS and isinstance(v, list):
                    v = tuple(v)
                kv.append((k, repr(v)))
            rendered_entries.append(kv)
        subtasks[eef_name] = rendered_entries

    return {
        "mimic_task_id": mimic_task_id,
        "class_prefix": class_prefix,
        "datagen_items": datagen_items,
        "subtasks": subtasks,
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
    "scripts/tools/record_annotated_demos.py": "scripts/record_annotated_demos.py",
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


def _patch_teleop_script(
    content: str,
    package_name: str,
    task_name: str,
    has_cameras: bool,
) -> str:
    """Customize the upstream teleop script for the generated task.

    Adds ``--dataset_dir`` / ``--dataset_file`` CLI args, wires those into the
    recorder, removes the upstream ``terminations.time_out = None`` line so
    episodes auto-reset at ``episode_length_s``, and (when cameras are
    configured) re-attaches them after IsaacLab's XR pipeline strips them via
    ``remove_camera_configs``.
    """
    content = _patch_script(content, package_name)

    # CLI args after --task.
    task_arg = 'parser.add_argument("--task", type=str, default=None, help="Name of the task.")'
    dataset_args = (
        f'{task_arg}\n'
        f'parser.add_argument(\n'
        f'    "--dataset_dir",\n'
        f'    type=str,\n'
        f'    default="./datasets/{task_name}",\n'
        f'    help="Directory where the recorded HDF5 dataset will be written.",\n'
        f')\n'
        f'parser.add_argument(\n'
        f'    "--dataset_file",\n'
        f'    type=str,\n'
        f'    default="dataset",\n'
        f'    help="Filename (without extension) for the recorded dataset.",\n'
        f')'
    )
    content = content.replace(task_arg, dataset_args, 1)

    # Keep the time_out termination so the env auto-resets at episode_length_s.
    content = content.replace(
        "    # modify configuration\n    env_cfg.terminations.time_out = None\n",
        "",
        1,
    )

    # Camera re-attach after XR strip + recorder path plumbing.
    xr_anchor = (
        '    if args_cli.xr:\n'
        '        env_cfg = remove_camera_configs(env_cfg)\n'
        '        env_cfg.sim.render.antialiasing_mode = "DLSS"\n'
    )
    if xr_anchor in content:
        replacement_lines = [xr_anchor.rstrip("\n")]
        if has_cameras:
            replacement_lines.append(
                "        # Re-attach cameras after the XR strip — upstream removes\n"
                "        # them because the XR compositor can fight with replicator\n"
                "        # over the RTX render queue.\n"
                "        attach_cameras(env_cfg.scene)"
            )
        replacement_lines.append(
            "\n    # Override the recorder's per-run dataset target. The recorder\n"
            "    # itself, entity_order, and eef_names are set by BaseILEnvCfg.\n"
            "    env_cfg.recorders.dataset_export_dir_path = args_cli.dataset_dir\n"
            "    env_cfg.recorders.dataset_filename = args_cli.dataset_file\n"
        )
        content = content.replace(xr_anchor, "\n".join(replacement_lines) + "\n", 1)

    # attach_cameras import (after the package import that _patch_script just
    # injected).
    if has_cameras:
        pkg_import = f"import {package_name}  # noqa: F401"
        attach_import = (
            f"from {package_name}.tasks.manager_based.{task_name}.{task_name}_cfg "
            f"import attach_cameras"
        )
        if pkg_import in content and attach_import not in content:
            content = content.replace(
                pkg_import,
                f"{pkg_import}\n{attach_import}",
                1,
            )

    return content


def _patch_record_annotated_demos(content: str, package_name: str) -> str:
    """Patch the upstream annotated-demo recorder.

    Publishes the per-EEF queue heads as ``env._debug_subtask_heads`` so
    subtask predicates in ``mdp/observations.py`` can gate any debug printing
    on the signal each EEF is currently dwelling on.
    """
    content = _patch_script(content, package_name)

    anchor = "                _, _, terminated, truncated, _ = env.step(action)"
    inject = (
        "                # Publish the current queue heads so subtask\n"
        "                # observation functions only print debug info for\n"
        "                # the signal each EEF is actively waiting on.\n"
        "                env._debug_subtask_heads = {\n"
        "                    signal for signal in annotator.current_signal_heads().values()\n"
        "                    if signal is not None\n"
        "                }\n"
    )
    if anchor in content and "env._debug_subtask_heads" not in content:
        content = content.replace(anchor, inject + anchor, 1)

    return content


def copy_isaaclab_scripts(
    isaaclab_path: Path,
    project_dir: Path,
    package_name: str,
    task_name: str,
    has_cameras: bool,
    dry_run: bool = False,
) -> None:
    """Copy and patch IsaacLab scripts into the generated project."""
    for src_rel, dst_rel in SCRIPTS_TO_PATCH.items():
        src = isaaclab_path / src_rel
        dst = project_dir / dst_rel
        if not src.is_file():
            print(f"  WARNING: Script not found, skipping: {src}")
            continue
        raw = src.read_text()
        if dst_rel == "scripts/teleop.py":
            patched = _patch_teleop_script(raw, package_name, task_name, has_cameras)
        elif dst_rel == "scripts/record_annotated_demos.py":
            patched = _patch_record_annotated_demos(raw, package_name)
        else:
            patched = _patch_script(raw, package_name)
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
    if context["mimic"] is not None:
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

    # ----- 7) Copy & patch IsaacLab scripts -----
    if isaaclab_path is not None:
        print("  [7/7] IsaacLab scripts (with extension import)...")
        copy_isaaclab_scripts(
            isaaclab_path,
            project_dir,
            package_name,
            task_name,
            has_cameras=bool(context["cameras"]),
            dry_run=dry_run,
        )
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
