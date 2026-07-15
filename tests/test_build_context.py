# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``build_context`` — the contract between validated YAML and templates (goal §4)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("franka_panda", "FrankaPanda"),
        ("tm7_g1", "Tm7G1"),
        ("franka_osc", "FrankaOsc"),
        ("so100_joint_pos", "So100JointPos"),
        ("g1", "G1"),
    ],
)
def test_task_name_to_class_prefix_is_pascal_case(create_task, make_task_def, task_name, expected):
    cfg = make_task_def(task_name=task_name)
    ctx = create_task.build_context(cfg, task_name)
    assert ctx["class_prefix"] == expected


def test_class_prefix_is_always_a_valid_identifier(create_task, make_task_def):
    # Empty tokens from leading/trailing/repeated underscores are dropped.
    for task_name in ("_leading", "trailing_", "double__underscore", "tm7_gr1t"):
        cfg = make_task_def(task_name=task_name)
        prefix = create_task.build_context(cfg, task_name)["class_prefix"]
        assert prefix.isidentifier(), f"{task_name!r} -> {prefix!r} is not a valid identifier"


def test_default_teleop_device_pink_ik_is_handtracking(create_task, make_task_def):
    cfg = make_task_def(ik_controller={"controller_type": "pink_ik"})
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["teleop"]["device"] == "handtracking"


def test_default_teleop_device_other_is_keyboard(create_task, make_task_def):
    cfg = make_task_def(ik_controller={"controller_type": "differential_ik"})
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["teleop"]["device"] == "keyboard"


def test_explicit_teleop_device_wins(create_task, make_task_def):
    cfg = make_task_def(
        ik_controller={"controller_type": "pink_ik"},
        teleop={"device": "spacemouse"},
    )
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["teleop"]["device"] == "spacemouse"


def test_mimic_task_id_suffix(create_task, make_task_def):
    cfg = make_task_def(task_id="Isaac-Pick-v0")
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["mimic_task_id"] == "Isaac-Pick-v0-Mimic"


def test_scene_object_usd_defaults_to_object_name(create_task, make_task_def):
    cfg = make_task_def(
        scene_objects=[
            {"name": "red_cube", "prim_path": "/World/envs/env_.*/Cube", "init_pos": [0, 0, 0]}
        ]
    )
    ctx = create_task.build_context(cfg, "demo_task")
    obj = ctx["scene_objects"][0]
    assert obj["usd_file"] == "red_cube.usd"
    assert obj["usd_var"] == "RED_CUBE_USD_PATH"
    assert obj["leaf"] == "Cube"


def test_scene_object_explicit_usd_preserved(create_task, make_task_def):
    cfg = make_task_def(
        scene_objects=[
            {
                "name": "cube",
                "prim_path": "/World/Cube",
                "init_pos": [0, 0, 0],
                "usd_file": "custom.usd",
            }
        ]
    )
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["scene_objects"][0]["usd_file"] == "custom.usd"


def test_material_and_collision_flags(create_task, make_task_def):
    cfg = make_task_def(
        scene_objects=[
            {
                "name": "sdf_obj",
                "prim_path": "/World/A",
                "init_pos": [0, 0, 0],
                "use_default_sdf_collision": True,
            },
            {
                "name": "baked_obj",
                "prim_path": "/World/B",
                "init_pos": [0, 0, 0],
                "use_default_sdf_collision": False,
                "physics_material": {"static_friction": 0.9},
                "visual_material": "Red",
            },
        ]
    )
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["has_sdf_objects"] is True
    assert ctx["has_baked_collision_objects"] is True
    assert ctx["has_physics_materials"] is True
    assert ctx["has_visual_materials"] is True


def test_camera_populates_context(create_task, make_task_def):
    cfg = make_task_def(
        cameras=[{"name": "wrist_cam", "prim_path": "/World/Robot/wrist/cam"}]
    )
    ctx = create_task.build_context(cfg, "demo_task")
    assert len(ctx["cameras"]) == 1
    assert ctx["cameras"][0]["name"] == "wrist_cam"


def test_robot_custom_name_propagates(create_task, make_task_def):
    cfg = make_task_def(
        robot={
            "name": "g1_humanoid",
            "import_path": "isaaclab_assets.robots.g1",
            "config_name": "G1_CFG",
        }
    )
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["robot"]["name"] == "g1_humanoid"


def test_context_contract_keys_present(create_task, make_task_def):
    cfg = make_task_def()
    ctx = create_task.build_context(cfg, "demo_task")
    for key in (
        "package_name",
        "class_prefix",
        "task_name",
        "task_id",
        "controller_type",
        "robot",
        "scene",
        "sim",
        "scene_objects",
        "reset_events",
        "eef_slices",
        "teleop",
        "mimic_task_id",
    ):
        assert key in ctx, f"missing context key: {key}"
    assert ctx["package_name"] == "demo_task"


def test_body_name_defaults_when_no_target_links(create_task, make_task_def):
    cfg = make_task_def()
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["eef"]["body_name"] == "ee_link"


def test_body_name_from_target_links(create_task, make_task_def):
    cfg = make_task_def(eef={"names": ["ee"], "target_links": {"ee": "panda_hand"}})
    ctx = create_task.build_context(cfg, "demo_task")
    assert ctx["eef"]["body_name"] == "panda_hand"
