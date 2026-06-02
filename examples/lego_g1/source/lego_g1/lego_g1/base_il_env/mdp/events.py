# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for generated IL environments."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_VELOCITY_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_robot_to_default(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reset_joint_targets: bool = False,
) -> None:
    """Reset one robot articulation to its configured default state.

    This is the articulation-only subset of Isaac Lab's ``reset_scene_to_default``.
    Use it when objects have separate reset terms and should not be written twice
    in the same reset pass.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    if reset_joint_targets:
        robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)
        robot.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)


def _canonical_pose_range(pose_range: Mapping[str, tuple[float, float]] | None) -> dict[str, tuple[float, float]]:
    if not pose_range:
        return {}
    aliases = {"rx": "roll", "ry": "pitch", "rz": "yaw"}
    out: dict[str, tuple[float, float]] = {}
    for key, value in pose_range.items():
        canonical = aliases.get(key, key)
        if canonical in _POSE_KEYS:
            out[canonical] = (float(value[0]), float(value[1]))
    return {key: out[key] for key in _POSE_KEYS if key in out}


def _canonical_velocity_range(
    velocity_range: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    if not velocity_range:
        return {}
    return {
        key: (float(value[0]), float(value[1]))
        for key, value in velocity_range.items()
        if key in _VELOCITY_KEYS
    }


def _normalize_sobol_inputs(
    pose_range: Mapping[str, tuple[float, float]] | None,
    velocity_range: Mapping[str, tuple[float, float]] | None,
    asset_cfg: SceneEntityCfg | None,
    pose_ranges: Mapping[str, Mapping[str, tuple[float, float]]] | None,
    velocity_ranges: Mapping[str, Mapping[str, tuple[float, float]]] | None,
    asset_cfgs: Mapping[str, SceneEntityCfg] | Sequence[SceneEntityCfg] | None,
) -> tuple[list[tuple[str, SceneEntityCfg]], dict[str, dict[str, tuple[float, float]]], dict[str, dict[str, tuple[float, float]]]]:
    if asset_cfgs is None:
        cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")
        asset_items = [(cfg.name, cfg)]
        pose_by_asset = {cfg.name: _canonical_pose_range(pose_range)}
        velocity_by_asset = {cfg.name: _canonical_velocity_range(velocity_range)}
        return asset_items, pose_by_asset, velocity_by_asset

    if isinstance(asset_cfgs, Mapping):
        asset_items = list(asset_cfgs.items())
    else:
        asset_items = [(cfg.name, cfg) for cfg in asset_cfgs]

    pose_ranges = pose_ranges or {}
    velocity_ranges = velocity_ranges or {}
    pose_by_asset = {
        name: _canonical_pose_range(pose_ranges.get(name))
        for name, _ in asset_items
    }
    velocity_by_asset = {
        name: _canonical_velocity_range(velocity_ranges.get(name))
        for name, _ in asset_items
    }
    return asset_items, pose_by_asset, velocity_by_asset


def _sobol_state_key(
    seed: int,
    asset_items: list[tuple[str, SceneEntityCfg]],
    pose_by_asset: dict[str, dict[str, tuple[float, float]]],
) -> tuple:
    return (
        int(seed),
        tuple(
            (
                name,
                tuple((key, *pose_by_asset[name][key]) for key in pose_by_asset[name]),
            )
            for name, _ in asset_items
        ),
    )


def _env_id_list(env_ids: torch.Tensor) -> list[int]:
    return [int(env_id) for env_id in env_ids.detach().cpu().flatten().tolist()]


def _counter_value(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _explicit_sobol_success_mask(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> torch.Tensor | None:
    succeeded = getattr(env, "_sobol_episode_succeeded", None)
    if succeeded is None:
        return None

    if not isinstance(succeeded, torch.Tensor):
        succeeded = torch.as_tensor(succeeded, dtype=torch.bool)
        index_ids = torch.as_tensor(_env_id_list(env_ids), dtype=torch.long)
        return succeeded[index_ids].detach().to(device="cpu", dtype=torch.bool).flatten()

    index_ids = env_ids.detach().to(device=succeeded.device, dtype=torch.long)
    mask = succeeded[index_ids].detach().to(device="cpu", dtype=torch.bool).flatten()
    try:
        succeeded[index_ids] = False
    except RuntimeError:
        pass
    return mask


def _sobol_success_mask(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    entry: dict,
) -> torch.Tensor:
    explicit_mask = _explicit_sobol_success_mask(env, env_ids)
    if explicit_mask is not None:
        return explicit_mask

    recorder_manager = getattr(env, "recorder_manager", None)
    exported_count = getattr(recorder_manager, "exported_successful_episode_count", None)
    if exported_count is None:
        if not getattr(env, "_sobol_no_recorder_warning_emitted", False):
            warnings.warn(
                "reset_root_state_sobol(advance_on_success_only=True) could not "
                "find env._sobol_episode_succeeded or "
                "env.recorder_manager.exported_successful_episode_count; "
                "advancing Sobol samples on every reset.",
                stacklevel=2,
            )
            setattr(env, "_sobol_no_recorder_warning_emitted", True)
        return torch.ones(len(env_ids), dtype=torch.bool)

    current_count = _counter_value(exported_count)
    previous_count = entry.get("last_exported_successful_episode_count")
    entry["last_exported_successful_episode_count"] = current_count
    if previous_count is None:
        return torch.zeros(len(env_ids), dtype=torch.bool)
    return torch.full((len(env_ids),), current_count > previous_count, dtype=torch.bool)


def _draw_sobol_samples(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    seed: int,
    asset_items: list[tuple[str, SceneEntityCfg]],
    pose_by_asset: dict[str, dict[str, tuple[float, float]]],
    advance_on_success_only: bool = False,
) -> tuple[torch.Tensor | None, dict[str, slice]]:
    slices: dict[str, slice] = {}
    cursor = 0
    for name, _ in asset_items:
        dim = len(pose_by_asset[name])
        slices[name] = slice(cursor, cursor + dim)
        cursor += dim

    if cursor == 0:
        return None, slices

    state = getattr(env, "_sobol_reset_state", None)
    if state is None:
        state = {}
        setattr(env, "_sobol_reset_state", state)

    key = _sobol_state_key(seed, asset_items, pose_by_asset)
    entry = state.get(key)
    if entry is None:
        entry = {
            "engine": torch.quasirandom.SobolEngine(cursor, scramble=True, seed=int(seed)),
            "count": 0,
        }
        state[key] = entry

    if not advance_on_success_only:
        samples = entry["engine"].draw(len(env_ids))
        entry["count"] += len(env_ids)
        setattr(env, "_sobol_count", entry["count"])
        return samples, slices

    env_ids_list = _env_id_list(env_ids)
    cache: dict[int, torch.Tensor] = entry.setdefault("cache", {})
    success_mask = _sobol_success_mask(env, env_ids, entry)
    refresh_rows = [
        row
        for row, env_id in enumerate(env_ids_list)
        if bool(success_mask[row]) or env_id not in cache
    ]

    samples = torch.empty((len(env_ids_list), cursor), dtype=torch.float32)
    refreshed_rows = set(refresh_rows)
    if refresh_rows:
        fresh_samples = entry["engine"].draw(len(refresh_rows))
        entry["count"] += len(refresh_rows)
        for row, sample in zip(refresh_rows, fresh_samples):
            cache[env_ids_list[row]] = sample.clone()
            samples[row] = sample

    for row, env_id in enumerate(env_ids_list):
        if row not in refreshed_rows:
            samples[row] = cache[env_id]

    setattr(env, "_sobol_count", entry["count"])
    return samples, slices


def _stable_asset_seed(seed: int, asset_name: str) -> int:
    name_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(asset_name))
    return (int(seed) + name_offset) % (2**63 - 1)


def _draw_velocity_offsets(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    seed: int,
    asset_name: str,
    velocity_range: dict[str, tuple[float, float]],
    device: str | torch.device,
) -> torch.Tensor:
    ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in _VELOCITY_KEYS], device="cpu")
    if not velocity_range or torch.all(ranges[:, 0] == ranges[:, 1]):
        return torch.zeros((len(env_ids), 6), device=device)

    state = getattr(env, "_sobol_velocity_reset_state", None)
    if state is None:
        state = {}
        setattr(env, "_sobol_velocity_reset_state", state)

    key = (
        int(seed),
        asset_name,
        tuple((vel_key, *velocity_range[vel_key]) for vel_key in velocity_range),
    )
    entry = state.get(key)
    if entry is None:
        generator = torch.Generator()
        generator.manual_seed(_stable_asset_seed(seed, asset_name))
        entry = {"generator": generator, "count": 0}
        state[key] = entry

    unit = torch.rand((len(env_ids), 6), generator=entry["generator"], device="cpu")
    entry["count"] += len(env_ids)
    velocity_offsets = ranges[:, 0] + unit * (ranges[:, 1] - ranges[:, 0])
    return velocity_offsets.to(device=device)


def _apply_asset_root_state(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset: RigidObject | Articulation,
    pose_offsets: torch.Tensor,
    velocity_offsets: torch.Tensor,
) -> None:
    root_states = asset.data.default_root_state[env_ids].clone()

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_offsets[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_offsets[:, 3],
        pose_offsets[:, 4],
        pose_offsets[:, 5],
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    velocities = root_states[:, 7:13] + velocity_offsets

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_sobol(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg | None = None,
    *,
    pose_ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
    velocity_ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
    asset_cfgs: dict[str, SceneEntityCfg] | Sequence[SceneEntityCfg] | None = None,
    seed: int = 0,
    advance_on_success_only: bool = False,
) -> None:
    """Reset root poses from a scrambled, seeded Sobol sequence.

    The single-asset call shape mirrors Isaac Lab's ``reset_root_state_uniform``.
    For joint object coverage, pass ``asset_cfgs`` plus per-asset
    ``pose_ranges``/``velocity_ranges``. One Sobol engine is built over the
    concatenated pose dimensions, then each asset receives its slice.

    If ``advance_on_success_only`` is true, each env reuses its cached Sobol
    sample until that env succeeds. Set ``env._sobol_episode_succeeded`` to a
    per-env bool tensor for exact vectorized gating. Without that tensor, the
    recorder manager's global successful-export counter is used as a single-env
    fallback; if no recorder is present, samples advance on every reset.
    """
    asset_items, pose_by_asset, velocity_by_asset = _normalize_sobol_inputs(
        pose_range,
        velocity_range,
        asset_cfg,
        pose_ranges,
        velocity_ranges,
        asset_cfgs,
    )
    unit_samples, sample_slices = _draw_sobol_samples(
        env,
        env_ids,
        seed,
        asset_items,
        pose_by_asset,
        advance_on_success_only,
    )

    for name, cfg in asset_items:
        asset: RigidObject | Articulation = env.scene[cfg.name]
        pose_offsets = torch.zeros((len(env_ids), 6), device=asset.device)
        keys = list(pose_by_asset[name])
        if keys and unit_samples is not None:
            samples = unit_samples[:, sample_slices[name]].to(device=asset.device)
            ranges = torch.tensor([pose_by_asset[name][key] for key in keys], device=asset.device)
            scaled = ranges[:, 0] + samples * (ranges[:, 1] - ranges[:, 0])
            for dim, key in enumerate(keys):
                pose_offsets[:, _POSE_KEYS.index(key)] = scaled[:, dim]

        velocity_offsets = _draw_velocity_offsets(
            env,
            env_ids,
            seed,
            name,
            velocity_by_asset[name],
            asset.device,
        )
        _apply_asset_root_state(env, env_ids, asset, pose_offsets, velocity_offsets)
