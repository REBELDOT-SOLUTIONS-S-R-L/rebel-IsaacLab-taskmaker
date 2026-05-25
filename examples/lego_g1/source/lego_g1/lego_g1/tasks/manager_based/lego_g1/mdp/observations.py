# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation functions for lego_g1.

This module defines the **subtask termination signals** that the Mimic data
generation pipeline uses to slice a demonstration into reusable segments.
Each signal returns a ``(num_envs, 1)`` float tensor that is ``1.0`` when the
associated subtask is complete for that env and ``0.0`` otherwise.

The signals are wired into the Mimic recorder via a ``SubtaskTermsCfg``
observation group (group name ``subtask_terms``) in ``lego_g1_cfg.py``, read
out by ``BaseILEnv.get_subtask_term_predicates``, and referenced by name
(e.g. ``grasp_brick_left``) from ``lego_g1_mimic_cfg.py``'s ``SubTaskConfig``
entries.

Thresholds here are placeholder values — refine them once a few demos have been
recorded and you can inspect the actual EEF/object trajectories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _link_pos(env: "ManagerBasedRLEnv", link_name: str) -> torch.Tensor:
    """World-frame position of a named robot link, minus the env origin."""
    robot = env.scene["unitree_g1"]
    idx = robot.data.body_names.index(link_name)
    return robot.data.body_pos_w[:, idx] - env.scene.env_origins


def _object_pos(env: "ManagerBasedRLEnv", object_name: str) -> torch.Tensor:
    """Object root position, minus the env origin."""
    return env.scene[object_name].data.root_pos_w - env.scene.env_origins


def _proximal_joint_mean(env: "ManagerBasedRLEnv", joint_pattern: str) -> torch.Tensor:
    """Mean joint position for an Inspire hand's proximal joints.

    Larger value ⇒ fingers curled inwards (closed). The pattern is a regex
    forwarded to ``ArticulationData.find_joints``, e.g. ``L_.*_proximal_joint``.
    """
    robot = env.scene["unitree_g1"]
    idxs, _ = robot.find_joints(joint_pattern)
    if not idxs:
        return torch.zeros(env.num_envs, device=env.device)
    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=env.device)
    return robot.data.joint_pos[:, idx_tensor].mean(dim=1)


def _tensor(values, device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float, device=device)


# Throttle subtask predicate prints so they don't flood stdout at sim rate.
# Set LEGO_G1_DEBUG_SUBTASKS=0 to silence, or to an integer N to print every Nth call.
_DEBUG_SUBTASKS_ENV = "LEGO_G1_DEBUG_SUBTASKS"
_DEBUG_DEFAULT_EVERY = 0
_print_counters: dict[str, int] = {}


def _check_mark(ok: bool) -> str:
    return "OK  " if ok else "FAIL"


def _debug_subtask(env, signal_name: str | None, header: str, rows: list[tuple[str, float, str, float, bool]]) -> None:
    """Pretty-print a subtask predicate's evaluation, gated to the current queue head.

    The script populates ``env._debug_subtask_heads`` with the active signal names per
    EEF; if that attribute is missing we print every signal (useful when running
    outside the annotated-teleop script).

    Args:
        signal_name: ObsTerm-level name (e.g. ``"grasp_brick_left"``). Pass ``None`` to
            disable printing for that ObsTerm entirely.
        header: First line, typically ``"[<signal_name>] done=OK|FAIL"``.
        rows: List of ``(label, value, op, threshold, passed)`` tuples to render aligned.
    """
    if signal_name is None:
        return
    heads = getattr(env, "_debug_subtask_heads", None)
    if heads is not None and signal_name not in heads:
        return

    import os

    raw = os.environ.get(_DEBUG_SUBTASKS_ENV, str(_DEBUG_DEFAULT_EVERY))
    try:
        every = int(raw)
    except ValueError:
        every = _DEBUG_DEFAULT_EVERY
    if every <= 0:
        return
    count = _print_counters.get(signal_name, 0) + 1
    _print_counters[signal_name] = count
    if count % every != 0:
        return

    label_width = max((len(label) for label, *_ in rows), default=0)
    lines = [header]
    for label, value, op, threshold, passed in rows:
        lines.append(
            f"    {label:<{label_width}}  {value:>7.3f}   need {op} {threshold:<7.3f}  {_check_mark(passed)}"
        )
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Subtask termination signals (subtask_terms observation group)
# ---------------------------------------------------------------------------
def grasp_brick_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    object_name: str,
    dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_closed_threshold: float,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Fires when the EEF is within ``dist_threshold`` of the brick AND the
    fingers' proximal joints are at least ``gripper_closed_threshold``."""
    eef = _link_pos(env, eef_link)
    obj = _object_pos(env, object_name)
    eef_obj_dist = torch.norm(eef - obj, dim=-1)
    grip = _proximal_joint_mean(env, gripper_joint_pattern)
    near = eef_obj_dist <= dist_threshold
    closed = grip >= gripper_closed_threshold
    done = (near & closed).float().unsqueeze(-1)
    _debug_subtask(
        env,
        signal_name,
        f"[{signal_name}] done={_check_mark(bool(done[0, 0].item()))}",
        [
            ("eef → object distance", eef_obj_dist[0].item(), "≤", float(dist_threshold), bool(near[0].item())),
            ("gripper closure", grip[0].item(), "≥", float(gripper_closed_threshold), bool(closed[0].item())),
        ],
    )
    return done


def move_brick_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    object_name: str,
    target_pos: tuple[float, float, float],
    eef_dist_threshold: float,
    target_dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_closed_threshold: float,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Fires when the EEF still holds the brick (near + closed) AND the brick
    has been carried within ``target_dist_threshold`` of ``target_pos``."""
    eef = _link_pos(env, eef_link)
    obj = _object_pos(env, object_name)
    target = _tensor(target_pos, env.device).expand_as(obj)
    eef_obj_dist = torch.norm(eef - obj, dim=-1)
    obj_tgt_dist = torch.norm(obj - target, dim=-1)
    grip = _proximal_joint_mean(env, gripper_joint_pattern)
    near_eef = eef_obj_dist <= eef_dist_threshold
    near_target = obj_tgt_dist <= target_dist_threshold
    closed = grip >= gripper_closed_threshold
    done = (near_eef & near_target & closed).float().unsqueeze(-1)
    _debug_subtask(
        env,
        signal_name,
        f"[{signal_name}] done={_check_mark(bool(done[0, 0].item()))}",
        [
            ("eef → object distance", eef_obj_dist[0].item(), "≤", float(eef_dist_threshold), bool(near_eef[0].item())),
            ("object → target distance", obj_tgt_dist[0].item(), "≤", float(target_dist_threshold), bool(near_target[0].item())),
            ("gripper closure", grip[0].item(), "≥", float(gripper_closed_threshold), bool(closed[0].item())),
        ],
    )
    return done


def release_brick_done(
    env: "ManagerBasedRLEnv",
    object_name: str,
    target_pos: tuple[float, float, float],
    target_dist_threshold: float,
    gripper_joint_pattern: str,
    gripper_open_threshold: float,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Fires when the brick is within ``target_dist_threshold`` of ``target_pos``
    AND the fingers' proximal joints are at most ``gripper_open_threshold``."""
    obj = _object_pos(env, object_name)
    target = _tensor(target_pos, env.device).expand_as(obj)
    obj_tgt_dist = torch.norm(obj - target, dim=-1)
    grip = _proximal_joint_mean(env, gripper_joint_pattern)
    near_target = obj_tgt_dist <= target_dist_threshold
    is_open = grip <= gripper_open_threshold
    done = (near_target & is_open).float().unsqueeze(-1)
    _debug_subtask(
        env,
        signal_name,
        f"[{signal_name}] done={_check_mark(bool(done[0, 0].item()))}",
        [
            ("object → target distance", obj_tgt_dist[0].item(), "≤", float(target_dist_threshold), bool(near_target[0].item())),
            ("gripper openness", grip[0].item(), "≤", float(gripper_open_threshold), bool(is_open[0].item())),
        ],
    )
    return done


def idle_done(
    env: "ManagerBasedRLEnv",
    eef_link: str,
    idle_pos: tuple[float, float, float],
    threshold: tuple[float, float, float],
    signal_name: str | None = None,
) -> torch.Tensor:
    """Fires when the EEF is inside an axis-aligned box of half-extents
    ``threshold`` centred on ``idle_pos``."""
    eef = _link_pos(env, eef_link)
    target = _tensor(idle_pos, env.device).expand_as(eef)
    thr = _tensor(threshold, env.device)
    delta = (eef - target).abs()
    per_axis_ok = delta <= thr
    in_range = per_axis_ok.all(dim=-1)
    done = in_range.float().unsqueeze(-1)
    _debug_subtask(
        env,
        signal_name,
        f"[{signal_name}] done={_check_mark(bool(done[0, 0].item()))}",
        [
            ("|Δx| (eef − idle)", delta[0, 0].item(), "≤", float(threshold[0]), bool(per_axis_ok[0, 0].item())),
            ("|Δy| (eef − idle)", delta[0, 1].item(), "≤", float(threshold[1]), bool(per_axis_ok[0, 1].item())),
            ("|Δz| (eef − idle)", delta[0, 2].item(), "≤", float(threshold[2]), bool(per_axis_ok[0, 2].item())),
        ],
    )
    return done
