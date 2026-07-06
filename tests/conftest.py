# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the IsaacLab-Task-Maker unit tests.

The tests here exercise the YAML -> Isaac Lab task pipeline **without** launching
Isaac Sim, GPU physics, or an Omniverse app:

* ``scripts/create_task.py`` and ``scripts/remove_task.py`` import cleanly (stdlib
  + ``yaml``/``jinja2``/``pydantic``), so they are imported in-process.
* The generated ``base_il_env.py`` template imports ``isaaclab`` (which pulls in
  ``pxr`` and fails outside a full Isaac Sim install), so those modules are
  replaced with light fakes in ``sys.modules`` before the file is loaded.

Nothing here hardcodes an absolute path — the repo root is derived from this
file's location.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BASE_IL_ENV_FILE = (
    REPO_ROOT
    / "source"
    / "IsaacLabTaskMaker"
    / "isaaclab_task_maker"
    / "templates"
    / "base_il_env"
    / "base_il_env.py"
)


# ---------------------------------------------------------------------------
# Import the code-generation scripts in-process
# ---------------------------------------------------------------------------
def _import_from_scripts(module_name: str):
    """Import a module from ``scripts/`` regardless of the current working dir."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(module_name)


@pytest.fixture(scope="session")
def create_task():
    """The ``scripts/create_task.py`` module."""
    return _import_from_scripts("create_task")


@pytest.fixture(scope="session")
def remove_task():
    """The ``scripts/remove_task.py`` module."""
    return _import_from_scripts("remove_task")


# ---------------------------------------------------------------------------
# Minimal valid TaskDefinition factory
# ---------------------------------------------------------------------------
@pytest.fixture
def make_task_def(create_task):
    """Return a factory that builds a valid ``TaskDefinition`` from overrides.

    Only the truly required fields are filled by default; tests pass extra
    top-level blocks (``eef``, ``scene_objects``, ``ik_controller``, ...) as
    keyword overrides merged into the base dict.
    """

    def _factory(**overrides):
        data = {
            "task_name": "demo_task",
            "task_id": "Isaac-Demo-Task-v0",
            "robot": {
                "import_path": "isaaclab_assets.robots.franka",
                "config_name": "FRANKA_PANDA_CFG",
            },
        }
        data.update(overrides)
        return create_task.TaskDefinition(**data)

    return _factory


# ---------------------------------------------------------------------------
# Fake isaaclab modules + BaseILEnv loader (no Isaac Sim required)
# ---------------------------------------------------------------------------
def _install_fake_isaaclab():
    """Register light ``isaaclab`` fakes in ``sys.modules``.

    ``PoseUtils`` is implemented as **pass-through bijections** so the pose <->
    action round trip is exact and slice placement can be asserted directly:

        make_pose(pos, rot)   -> {"pos": pos, "rot": rot}
        unmake_pose(pose)     -> (pose["pos"], pose["rot"])
        matrix_from_quat(q)   -> q
        quat_from_matrix(r)   -> r
    """
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []  # mark as a package
    utils = types.ModuleType("isaaclab.utils")
    utils.__path__ = []
    math_mod = types.ModuleType("isaaclab.utils.math")
    envs = types.ModuleType("isaaclab.envs")

    def make_pose(pos, rot):
        return {"pos": pos, "rot": rot}

    def unmake_pose(pose):
        return pose["pos"], pose["rot"]

    def matrix_from_quat(quat):
        return quat

    def quat_from_matrix(rot):
        return rot

    math_mod.make_pose = make_pose
    math_mod.unmake_pose = unmake_pose
    math_mod.matrix_from_quat = matrix_from_quat
    math_mod.quat_from_matrix = quat_from_matrix

    class ManagerBasedRLMimicEnv:  # plain stand-in base class
        pass

    envs.ManagerBasedRLMimicEnv = ManagerBasedRLMimicEnv

    utils.math = math_mod
    isaaclab.utils = utils
    isaaclab.envs = envs

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["isaaclab.envs"] = envs


@pytest.fixture(scope="session")
def base_il_env_module():
    """Load the ``base_il_env.py`` template with fake ``isaaclab`` in place."""
    _install_fake_isaaclab()
    spec = importlib.util.spec_from_file_location(
        "base_il_env_under_test", BASE_IL_ENV_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_env(base_il_env_module):
    """Return a factory that builds a ``BaseILEnv`` without running ``__init__``.

    The methods under test only read ``self.cfg`` and ``self.obs_buf``, so an
    instance created via ``object.__new__`` with those two attributes attached
    is sufficient.
    """

    def _factory(eef_names, action_slices, gripper_slices, obs_policy=None):
        env = object.__new__(base_il_env_module.BaseILEnv)
        env.cfg = types.SimpleNamespace(
            eef_names=eef_names,
            eef_action_slices=action_slices,
            eef_gripper_slices=gripper_slices,
        )
        env.obs_buf = {"policy": obs_policy or {}}
        return env

    return _factory
