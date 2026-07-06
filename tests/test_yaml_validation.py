# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pydantic validation + defaults for the YAML task schema (goal §1)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Required fields fail cleanly
# ---------------------------------------------------------------------------
def test_robot_config_requires_import_path_and_config_name(create_task):
    with pytest.raises(ValidationError):
        create_task.RobotConfig(config_name="X")  # missing import_path
    with pytest.raises(ValidationError):
        create_task.RobotConfig(import_path="pkg")  # missing config_name


def test_task_definition_requires_robot(create_task):
    with pytest.raises(ValidationError):
        create_task.TaskDefinition(task_name="t", task_id="Isaac-T-v0")


def test_scene_object_requires_name_prim_path_init_pos(create_task):
    with pytest.raises(ValidationError):
        create_task.SceneObject(name="cube", prim_path="/World/Cube")  # no init_pos


# ---------------------------------------------------------------------------
# Controller type validation
# ---------------------------------------------------------------------------
def test_unsupported_controller_type_rejected(create_task):
    with pytest.raises(ValidationError) as exc:
        create_task.IKControllerConfig(controller_type="bogus_controller")
    message = str(exc.value)
    assert "Unsupported controller_type" in message
    assert "bogus_controller" in message


@pytest.mark.parametrize(
    "controller",
    [
        "pink_ik",
        "differential_ik",
        "operational_space",
        "rmpflow",
        "joint_position",
        "relative_joint_position",
        "joint_velocity",
        "joint_effort",
    ],
)
def test_all_supported_controllers_accepted(create_task, controller):
    cfg = create_task.IKControllerConfig(controller_type=controller)
    assert cfg.controller_type == controller


def test_supported_controllers_constant_matches_parametrization(create_task):
    # Guards against the tuple drifting out of sync with the accepted-values test.
    assert set(create_task.SUPPORTED_CONTROLLERS) == {
        "pink_ik",
        "differential_ik",
        "operational_space",
        "rmpflow",
        "joint_position",
        "relative_joint_position",
        "joint_velocity",
        "joint_effort",
    }


# ---------------------------------------------------------------------------
# Defaults applied
# ---------------------------------------------------------------------------
def test_default_values_applied(create_task, make_task_def):
    cfg = make_task_def()
    assert cfg.robot.name == "robot"
    assert cfg.robot.prim_path == "/World/envs/env_.*/Robot"
    assert cfg.scene.usd_file == "scene.usd"
    assert cfg.sim.decimation == 6
    assert cfg.sim.episode_length_s == pytest.approx(20.0)
    assert cfg.teleop.device is None
    assert cfg.resets.seed == 0
    assert cfg.resets.sobol_advance_on_success_only is False
    assert cfg.ik_controller.controller_type == "pink_ik"
    assert cfg.scene_objects == []
    assert cfg.light is None


# ---------------------------------------------------------------------------
# Parsing YAML text
# ---------------------------------------------------------------------------
INLINE_YAML = """
task_name: pick_cube
task_id: Isaac-Pick-Cube-v0
robot:
  import_path: isaaclab_assets.robots.franka
  config_name: FRANKA_PANDA_CFG
sim:
  decimation: 4
ik_controller:
  controller_type: differential_ik
"""


def test_parse_inline_yaml_string(create_task):
    data = yaml.safe_load(INLINE_YAML)
    cfg = create_task.TaskDefinition(**data)
    assert cfg.task_name == "pick_cube"
    assert cfg.sim.decimation == 4
    assert cfg.ik_controller.controller_type == "differential_ik"


def test_parse_shipped_task_definition(create_task):
    # Guards the real, shipped schema against drift.
    yaml_path = create_task.TASK_DEFS_DIR / "franka_panda.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    cfg = create_task.TaskDefinition(**data)
    assert cfg.task_name
    assert cfg.task_id
    assert cfg.robot.import_path
