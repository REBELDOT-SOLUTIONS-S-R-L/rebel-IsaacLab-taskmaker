#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Remove a previously generated IsaacLab extension task.

This script:
  1. Runs ``pip uninstall <package_name>``
  2. Removes the ``<package_name>_register.pth`` file from site-packages
  3. Optionally removes the generated project directory

Usage:
    python scripts/remove_task.py tm7_g1
    python scripts/remove_task.py tm7_g1 --project-dir ~/my_tasks/tm7_g1
    python scripts/remove_task.py tm7_g1 --keep-files
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sysconfig
import sys
from pathlib import Path


def find_pth_file(package_name: str) -> Path | None:
    """Locate the register .pth file in site-packages."""
    pth_filename = f"{package_name}_register.pth"
    purelib = sysconfig.get_path("purelib")
    if purelib:
        candidate = Path(purelib) / pth_filename
        if candidate.is_file():
            return candidate
    return None


def pip_uninstall(package_name: str) -> bool:
    """Uninstall the package via pip. Returns True if it was installed."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", package_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        if "not installed" in result.stdout.lower() or "not installed" in result.stderr.lower():
            print(f"  Package '{package_name}' was not pip-installed.")
            return False
        print(f"  Uninstalled pip package: {package_name}")
        return True
    else:
        if "not installed" in result.stderr.lower():
            print(f"  Package '{package_name}' was not pip-installed.")
        else:
            print(f"  WARNING: pip uninstall failed: {result.stderr.strip()}")
        return False


def remove_pth_file(package_name: str) -> bool:
    """Remove the .pth file from site-packages. Returns True if removed."""
    pth_path = find_pth_file(package_name)
    if pth_path is None:
        print(f"  No .pth file found for '{package_name}'.")
        return False
    pth_path.unlink()
    print(f"  Removed: {pth_path}")
    return True


def remove_project_dir(project_dir: Path) -> bool:
    """Remove the generated project directory. Returns True if removed."""
    if not project_dir.is_dir():
        print(f"  Project directory not found: {project_dir}")
        return False
    shutil.rmtree(project_dir)
    print(f"  Removed: {project_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove a generated IsaacLab IL extension task and clean up all artifacts."
    )
    parser.add_argument(
        "task_name",
        help="Name of the task to remove (the snake_case task_name from the YAML, e.g. tm7_g1).",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help=(
            "Path to the generated project directory. "
            "If not provided, looks for ./<task_name> in the current directory."
        ),
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Only uninstall the pip package and remove the .pth file; keep project files on disk.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    package_name = task_name

    print(f"Removing task '{task_name}'...")

    print("\n[1/3] Pip uninstall...")
    pip_uninstall(package_name)

    print("\n[2/3] Cleaning .pth file...")
    remove_pth_file(package_name)

    if args.keep_files:
        print("\n[3/3] Skipped project directory removal (--keep-files)")
    else:
        project_dir = Path(args.project_dir).resolve() if args.project_dir else Path.cwd() / task_name
        print(f"\n[3/3] Removing project directory: {project_dir}")
        remove_project_dir(project_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
