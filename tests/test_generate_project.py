# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end project generation into ``tmp_path`` (goal §5).

These render the real Jinja2 templates but write only into a pytest temp dir —
no Isaac Sim, no pip, no network.
"""

from __future__ import annotations

import ast

import pytest


def _generate(create_task, cfg, out_dir, **kwargs):
    create_task.generate_extension_project(cfg, out_dir, **kwargs)
    return out_dir / cfg.task_name


def test_creates_directory_tree(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    project = _generate(create_task, cfg, tmp_path)
    pkg = project / "source" / "demo_task" / "demo_task"
    assert pkg.is_dir()
    assert (pkg / "base_il_env").is_dir()
    assert (pkg / "tasks" / "manager_based" / "demo_task").is_dir()
    assert (pkg / "tasks" / "manager_based" / "demo_task" / "mdp").is_dir()


def test_writes_expected_files(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    project = _generate(create_task, cfg, tmp_path)
    ext = project / "source" / "demo_task"
    pkg = ext / "demo_task"
    task = pkg / "tasks" / "manager_based" / "demo_task"

    assert (project / "pyproject.toml").is_file()
    assert (ext / "setup.py").is_file()
    assert (ext / "demo_task_register.pth").is_file()
    assert (task / "demo_task_cfg.py").is_file()
    assert (task / "demo_task_mimic_cfg.py").is_file()
    assert (task / "mdp" / "observations.py").is_file()
    assert (task / "mdp" / "events.py").is_file()
    assert (task / "mdp" / "terminations.py").is_file()


def test_creates_asset_subdirs(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    project = _generate(create_task, cfg, tmp_path)
    assets = project / "source" / "demo_task" / "demo_task" / "assets"
    for sub in ("scenes", "objects", "robots"):
        assert (assets / sub).is_dir()


def test_register_pth_content(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    project = _generate(create_task, cfg, tmp_path)
    pth = project / "source" / "demo_task" / "demo_task_register.pth"
    assert pth.read_text().strip() == "import demo_task"


def test_generated_python_is_syntactically_valid(create_task, make_task_def, tmp_path):
    cfg = make_task_def(
        task_name="demo_task",
        eef={"names": ["ee"], "target_links": {"ee": "panda_hand"}},
        scene_objects=[
            {"name": "cube", "prim_path": "/World/envs/env_.*/Cube", "init_pos": [0.4, 0, 0.1]}
        ],
    )
    project = _generate(create_task, cfg, tmp_path)
    py_files = list(project.rglob("*.py"))
    assert py_files, "no generated .py files found"
    for path in py_files:
        source = path.read_text()
        ast.parse(source)  # raises SyntaxError on malformed output


def test_key_snippets_present(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task", task_id="Isaac-Demo-v0")
    project = _generate(create_task, cfg, tmp_path)
    cfg_py = (
        project / "source" / "demo_task" / "demo_task"
        / "tasks" / "manager_based" / "demo_task" / "demo_task_cfg.py"
    ).read_text()
    assert "DemoTask" in cfg_py  # class_prefix baked into the config class name


def test_refuses_overwrite_without_force(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    _generate(create_task, cfg, tmp_path)  # first generation
    with pytest.raises(SystemExit):
        create_task.generate_extension_project(cfg, tmp_path, force=False)


def test_force_overwrites_existing(create_task, make_task_def, tmp_path, monkeypatch):
    # Stub out the pip/.pth cleanup so --force never touches the real environment.
    monkeypatch.setattr(create_task, "_cleanup_installed_artifacts", lambda name: None)
    cfg = make_task_def(task_name="demo_task")
    project = _generate(create_task, cfg, tmp_path)
    # Drop a marker; a forced regeneration should wipe the dir and rebuild.
    marker = project / "STALE_MARKER"
    marker.write_text("stale")
    create_task.generate_extension_project(cfg, tmp_path, force=True)
    assert not marker.exists()
    assert (project / "pyproject.toml").is_file()


def test_dry_run_writes_nothing(create_task, make_task_def, tmp_path):
    cfg = make_task_def(task_name="demo_task")
    create_task.generate_extension_project(cfg, tmp_path, dry_run=True)
    assert not (tmp_path / "demo_task").exists()
