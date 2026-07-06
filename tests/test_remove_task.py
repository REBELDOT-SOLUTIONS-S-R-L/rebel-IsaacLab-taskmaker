# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``scripts/remove_task.py`` cleanup helpers (goal §7).

All subprocess / filesystem side effects are monkeypatched or confined to
``tmp_path`` — no package is actually uninstalled and only test-created
directories are removed.
"""

from __future__ import annotations

import types


# ---------------------------------------------------------------------------
# find_pth_file
# ---------------------------------------------------------------------------
def test_find_pth_file_found(remove_task, tmp_path, monkeypatch):
    (tmp_path / "mytask_register.pth").write_text("import mytask\n")
    monkeypatch.setattr(remove_task.sysconfig, "get_path", lambda name: str(tmp_path))
    result = remove_task.find_pth_file("mytask")
    assert result == tmp_path / "mytask_register.pth"


def test_find_pth_file_missing(remove_task, tmp_path, monkeypatch):
    monkeypatch.setattr(remove_task.sysconfig, "get_path", lambda name: str(tmp_path))
    assert remove_task.find_pth_file("mytask") is None


# ---------------------------------------------------------------------------
# pip_uninstall
# ---------------------------------------------------------------------------
def _fake_run(returncode=0, stdout="", stderr=""):
    def _run(*args, **kwargs):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return _run


def test_pip_uninstall_success(remove_task, monkeypatch):
    monkeypatch.setattr(
        remove_task.subprocess, "run", _fake_run(0, stdout="Successfully uninstalled mytask")
    )
    assert remove_task.pip_uninstall("mytask") is True


def test_pip_uninstall_not_installed(remove_task, monkeypatch):
    monkeypatch.setattr(
        remove_task.subprocess, "run", _fake_run(0, stdout="WARNING: Skipping mytask as it is not installed.")
    )
    assert remove_task.pip_uninstall("mytask") is False


def test_pip_uninstall_failure(remove_task, monkeypatch):
    monkeypatch.setattr(
        remove_task.subprocess, "run", _fake_run(1, stderr="some pip error")
    )
    assert remove_task.pip_uninstall("mytask") is False


# ---------------------------------------------------------------------------
# remove_pth_file
# ---------------------------------------------------------------------------
def test_remove_pth_file_found(remove_task, tmp_path, monkeypatch):
    pth = tmp_path / "mytask_register.pth"
    pth.write_text("import mytask\n")
    monkeypatch.setattr(remove_task, "find_pth_file", lambda name: pth)
    assert remove_task.remove_pth_file("mytask") is True
    assert not pth.exists()


def test_remove_pth_file_missing(remove_task, monkeypatch):
    monkeypatch.setattr(remove_task, "find_pth_file", lambda name: None)
    assert remove_task.remove_pth_file("mytask") is False


# ---------------------------------------------------------------------------
# remove_project_dir
# ---------------------------------------------------------------------------
def test_remove_project_dir_removes_only_given_dir(remove_task, tmp_path):
    target = tmp_path / "mytask"
    target.mkdir()
    (target / "file.txt").write_text("x")
    sibling = tmp_path / "keepme"
    sibling.mkdir()

    assert remove_task.remove_project_dir(target) is True
    assert not target.exists()
    assert sibling.exists()  # untouched


def test_remove_project_dir_missing(remove_task, tmp_path):
    assert remove_task.remove_project_dir(tmp_path / "nope") is False


# ---------------------------------------------------------------------------
# main --keep-files
# ---------------------------------------------------------------------------
def _stub_main_helpers(remove_task, monkeypatch, calls):
    monkeypatch.setattr(remove_task, "pip_uninstall", lambda name: calls.append(("pip", name)))
    monkeypatch.setattr(remove_task, "remove_pth_file", lambda name: calls.append(("pth", name)))
    monkeypatch.setattr(
        remove_task, "remove_project_dir", lambda p: calls.append(("dir", p))
    )


def test_main_keep_files_skips_dir_removal(remove_task, monkeypatch):
    calls = []
    _stub_main_helpers(remove_task, monkeypatch, calls)
    monkeypatch.setattr(remove_task.sys, "argv", ["remove_task.py", "mytask", "--keep-files"])
    remove_task.main()
    kinds = [c[0] for c in calls]
    assert "pip" in kinds and "pth" in kinds
    assert "dir" not in kinds


def test_main_without_keep_files_removes_dir(remove_task, monkeypatch):
    calls = []
    _stub_main_helpers(remove_task, monkeypatch, calls)
    monkeypatch.setattr(remove_task.sys, "argv", ["remove_task.py", "mytask"])
    remove_task.main()
    kinds = [c[0] for c in calls]
    assert "dir" in kinds
