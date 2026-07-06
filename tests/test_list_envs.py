# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_list_envs_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "list_envs.py"
    spec = importlib.util.spec_from_file_location("list_envs_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_envs_import_does_not_launch_isaac_sim():
    module = _load_list_envs_module()

    assert module.DEFAULT_TASK_PREFIX == "IL-"


def test_default_filter_matches_generated_task_ids():
    module = _load_list_envs_module()

    assert module.matches_filters("IL-LEGO-G1-v0", keyword=None, prefix=module.DEFAULT_TASK_PREFIX)
    assert module.matches_filters("IL-LEGO-G1-v0-Mimic", keyword="Mimic", prefix=module.DEFAULT_TASK_PREFIX)
    assert not module.matches_filters("Isaac-Ant-v0", keyword=None, prefix=module.DEFAULT_TASK_PREFIX)
    assert module.matches_filters("Isaac-Ant-v0", keyword=None, prefix=None)
