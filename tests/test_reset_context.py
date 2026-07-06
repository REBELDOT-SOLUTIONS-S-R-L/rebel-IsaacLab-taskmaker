# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-range normalization and Sobol grouping (goal §3, plus reset validation
from goal §1 which lives in these builders rather than in Pydantic)."""

from __future__ import annotations

import pytest


def _obj(create_task, **reset_kwargs):
    """A minimal RigidObjectCfg SceneObject with a reset block."""
    return create_task.SceneObject(
        name="cube",
        prim_path="/World/envs/env_.*/Cube",
        init_pos=[0.0, 0.0, 0.0],
        reset=reset_kwargs,
    )


# ---------------------------------------------------------------------------
# _build_object_reset_context — normalization
# ---------------------------------------------------------------------------
def test_reset_none_returns_none(create_task):
    obj = create_task.SceneObject(
        name="cube", prim_path="/World/Cube", init_pos=[0.0, 0.0, 0.0]
    )
    assert create_task._build_object_reset_context(obj, global_seed=0) is None


def test_pose_range_normalized_and_ordered(create_task):
    obj = _obj(
        create_task,
        pose_range={"yaw": [-1.0, 1.0], "x": [-0.1, 0.1], "z": [0.0, 0.2]},
    )
    ctx = create_task._build_object_reset_context(obj, global_seed=0)
    # keys follow x, y, z, roll, pitch, yaw ordering
    assert list(ctx["pose_range"].keys()) == ["x", "z", "yaw"]
    assert ctx["pose_range"]["x"] == (-0.1, 0.1)


def test_pos_range_and_rot_range_merge_into_pose_range(create_task):
    obj = _obj(
        create_task,
        pos_range={"x": [-0.1, 0.1]},
        rot_range={"yaw": [-3.14, 3.14]},
    )
    ctx = create_task._build_object_reset_context(obj, global_seed=0)
    assert ctx["pose_range"] == {"x": (-0.1, 0.1), "yaw": (-3.14, 3.14)}


def test_rotation_aliases_map_to_canonical(create_task):
    obj = _obj(
        create_task,
        rot_range={"rx": [0.0, 1.0], "ry": [0.0, 1.0], "rz": [0.0, 1.0]},
    )
    ctx = create_task._build_object_reset_context(obj, global_seed=0)
    assert set(ctx["pose_range"].keys()) == {"roll", "pitch", "yaw"}


def test_pose_range_alias_in_pose_range_too(create_task):
    obj = _obj(create_task, pose_range={"rx": [0.0, 1.0]})
    ctx = create_task._build_object_reset_context(obj, global_seed=0)
    assert "roll" in ctx["pose_range"]


# ---------------------------------------------------------------------------
# _build_object_reset_context — validation
# ---------------------------------------------------------------------------
def test_reset_on_non_rigid_object_rejected(create_task):
    obj = create_task.SceneObject(
        name="arm",
        type="ArticulationCfg",
        prim_path="/World/Arm",
        init_pos=[0.0, 0.0, 0.0],
        reset={"pose_range": {"x": [-0.1, 0.1]}},
    )
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "RigidObjectCfg" in str(exc.value)


def test_bad_pose_range_key_rejected(create_task):
    obj = _obj(create_task, pose_range={"nope": [0.0, 1.0]})
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "nope" in str(exc.value)


def test_bad_pos_range_key_rejected(create_task):
    obj = _obj(create_task, pos_range={"w": [0.0, 1.0]})
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "pos_range.w" in str(exc.value)


def test_bad_rot_range_key_rejected(create_task):
    obj = _obj(create_task, rot_range={"nope": [0.0, 1.0]})
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "rot_range.nope" in str(exc.value)


def test_duplicate_alias_rejected(create_task):
    # "roll" and its alias "rx" both land on the canonical key "roll".
    obj = _obj(create_task, pose_range={"roll": [0.0, 1.0], "rx": [0.0, 1.0]})
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "duplicates pose key" in str(exc.value)


def test_max_less_than_min_rejected(create_task):
    obj = _obj(create_task, pose_range={"x": [1.0, -1.0]})
    with pytest.raises(ValueError) as exc:
        create_task._build_object_reset_context(obj, global_seed=0)
    assert "max < min" in str(exc.value)


def test_wrong_length_range_rejected(create_task):
    # The Pydantic models type ranges as tuple[float, float], so a wrong length
    # is normally caught at construction. This guards the defensive check in the
    # normalizer itself, which is reachable if a caller passes a raw range.
    with pytest.raises(ValueError) as exc:
        create_task._normalize_reset_range_value((0.0, 1.0, 2.0), path="obj.reset.x")
    assert "exactly two values" in str(exc.value)


# ---------------------------------------------------------------------------
# _build_object_reset_context — seed resolution
# ---------------------------------------------------------------------------
def test_object_seed_overrides_global(create_task):
    obj = _obj(create_task, seed=42, pose_range={"x": [-0.1, 0.1]})
    ctx = create_task._build_object_reset_context(obj, global_seed=7)
    assert ctx["seed"] == 42


def test_object_seed_inherits_global_when_none(create_task):
    obj = _obj(create_task, pose_range={"x": [-0.1, 0.1]})
    ctx = create_task._build_object_reset_context(obj, global_seed=7)
    assert ctx["seed"] == 7


# ---------------------------------------------------------------------------
# _build_reset_events_context — grouping
# ---------------------------------------------------------------------------
def _scene_obj(sampler=None, seed=0, obj_type="RigidObjectCfg"):
    reset = None if sampler is None else {"sampler": sampler, "seed": seed}
    return {"type": obj_type, "reset": reset}


def test_grouping_by_sampler(create_task):
    scene_objects = [
        _scene_obj("uniform"),
        _scene_obj("sobol", seed=3),
        _scene_obj(None),  # default
    ]
    ctx = create_task._build_reset_events_context(scene_objects)
    assert len(ctx["uniform"]) == 1
    assert len(ctx["sobol"]) == 1
    assert len(ctx["default"]) == 1
    assert ctx["has_object_resets"] is True
    assert ctx["sobol_seed"] == 3


def test_default_bucket_empty_without_object_resets(create_task):
    # No uniform/sobol resets -> default objects are not emitted.
    scene_objects = [_scene_obj(None), _scene_obj(None)]
    ctx = create_task._build_reset_events_context(scene_objects)
    assert ctx["has_object_resets"] is False
    assert ctx["default"] == []


def test_mixed_sobol_seeds_rejected(create_task):
    scene_objects = [_scene_obj("sobol", seed=1), _scene_obj("sobol", seed=2)]
    with pytest.raises(ValueError) as exc:
        create_task._build_reset_events_context(scene_objects)
    assert "must match" in str(exc.value)


def test_sobol_advance_on_success_only_preserved(create_task):
    scene_objects = [_scene_obj("sobol", seed=5)]
    ctx = create_task._build_reset_events_context(
        scene_objects, sobol_advance_on_success_only=True
    )
    assert ctx["sobol_advance_on_success_only"] is True
