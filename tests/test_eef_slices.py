# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``_build_eef_slices`` — per-EEF action/gripper tensor layout (goal §2)."""

from __future__ import annotations

import pytest


def _cfg(make_task_def, controller, names, hand_joint_names=None, prefixes=None):
    return make_task_def(
        ik_controller={
            "controller_type": controller,
            "hand_joint_names": hand_joint_names or [],
        },
        eef={
            "names": names,
            "hand_joint_prefixes": prefixes or {},
        },
    )


# ---------------------------------------------------------------------------
# pink_ik
# ---------------------------------------------------------------------------
def test_pink_ik_single_eef_no_hand_joints(create_task, make_task_def):
    cfg = _cfg(make_task_def, "pink_ik", ["gripper"])
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] is None
    assert slices["action"]["gripper"] == {"pos": (0, 3), "quat": (3, 7)}


def test_pink_ik_single_eef_with_hand_joints(create_task, make_task_def):
    cfg = _cfg(make_task_def, "pink_ik", ["gripper"], hand_joint_names=["j1", "j2"])
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] == "contiguous"
    # pose block is 7 wide, then two hand joints
    assert slices["gripper"] == {"gripper": (7, 9)}


def test_pink_ik_dual_eef_with_prefixes(create_task, make_task_def):
    cfg = _cfg(
        make_task_def,
        "pink_ik",
        ["left", "right"],
        hand_joint_names=["L_a", "R_a", "L_b", "R_b"],
        prefixes={"left": "L_", "right": "R_"},
    )
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] == "prefix"
    assert slices["pose_offset"] == 14  # 2 EEFs * 7
    assert slices["num_eefs"] == 2
    assert slices["prefixes"] == {"left": "L_", "right": "R_"}
    # Both EEFs get their own pose block.
    assert slices["action"]["left"] == {"pos": (0, 3), "quat": (3, 7)}
    assert slices["action"]["right"] == {"pos": (7, 10), "quat": (10, 14)}


def test_pink_ik_dual_eef_missing_prefix_raises(create_task, make_task_def):
    cfg = _cfg(
        make_task_def,
        "pink_ik",
        ["left", "right"],
        hand_joint_names=["a", "b"],
        prefixes={"left": "L_"},  # right is missing
    )
    with pytest.raises(ValueError) as exc:
        create_task._build_eef_slices(cfg)
    assert "missing entries" in str(exc.value)
    assert "right" in str(exc.value)


def test_pink_ik_dual_eef_no_prefixes_contiguous_fallback(create_task, make_task_def):
    cfg = _cfg(
        make_task_def,
        "pink_ik",
        ["left", "right"],
        hand_joint_names=["a", "b", "c", "d"],  # 4 joints, split evenly
    )
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] == "contiguous_fallback"
    # pose_offset = 14; 2 joints per arm
    assert slices["gripper"] == {"left": (14, 16), "right": (16, 18)}


# ---------------------------------------------------------------------------
# differential_ik / operational_space / rmpflow (single-arm pose space)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("controller", ["differential_ik", "operational_space", "rmpflow"])
def test_single_arm_pose_controllers_no_hand_joints(create_task, make_task_def, controller):
    cfg = _cfg(make_task_def, controller, ["ee"])
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] is None
    assert slices["action"] == {"ee": {"pos": (0, 3), "quat": (3, 7)}}


@pytest.mark.parametrize("controller", ["differential_ik", "operational_space", "rmpflow"])
def test_single_arm_pose_controllers_with_hand_joints(create_task, make_task_def, controller):
    cfg = _cfg(make_task_def, controller, ["ee"], hand_joint_names=["g1"])
    slices = create_task._build_eef_slices(cfg)
    assert slices["gripper_mode"] == "contiguous"
    assert slices["gripper"] == {"ee": (7, 8)}


@pytest.mark.parametrize("controller", ["differential_ik", "operational_space", "rmpflow"])
def test_single_arm_pose_controllers_reject_multi_eef(create_task, make_task_def, controller):
    cfg = _cfg(make_task_def, controller, ["left", "right"])
    assert create_task._build_eef_slices(cfg) is None


# ---------------------------------------------------------------------------
# Joint-space controllers + empty EEF names -> None
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "controller",
    ["joint_position", "relative_joint_position", "joint_velocity", "joint_effort"],
)
def test_joint_space_controllers_return_none(create_task, make_task_def, controller):
    cfg = _cfg(make_task_def, controller, ["ee"])
    assert create_task._build_eef_slices(cfg) is None


def test_empty_eef_names_returns_none(create_task, make_task_def):
    cfg = _cfg(make_task_def, "pink_ik", [])
    assert create_task._build_eef_slices(cfg) is None
