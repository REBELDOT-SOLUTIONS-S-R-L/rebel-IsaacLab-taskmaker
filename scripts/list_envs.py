# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib

DEFAULT_TASK_PREFIX = "IL-"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="List registered Isaac Lab Task Maker environments.")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="PACKAGE",
        help=(
            "Python package to import before listing environments, e.g. 'lego_g1'. "
            "Repeat to import multiple packages. Installed generated extensions usually "
            "auto-register through their .pth file, so this is mainly for source checkouts."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_TASK_PREFIX,
        help=f"Only show environments whose IDs start with this prefix. Default: {DEFAULT_TASK_PREFIX!r}.",
    )
    parser.add_argument("--all", action="store_true", help="Show all registered Gymnasium environments.")
    return parser.parse_args()


def import_registration_packages(package_names: list[str]) -> None:
    """Import extension packages so their ``gym.register()`` calls run."""
    for package_name in package_names:
        module_name = package_name.strip()
        if not module_name:
            continue
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if module_name == missing_name or module_name.startswith(f"{missing_name}."):
                raise ModuleNotFoundError(
                    f"Could not import package '{module_name}'. Install the generated extension first "
                    f"or add its source directory to PYTHONPATH."
                ) from exc
            raise


def matches_filters(task_id: str, keyword: str | None, prefix: str | None) -> bool:
    """Return whether a Gymnasium env ID should be shown."""
    if prefix is not None and not task_id.startswith(prefix):
        return False
    return keyword is None or keyword in task_id


def main() -> None:
    """Print registered Task Maker environments."""
    args_cli = parse_args()

    # Launch Isaac Sim before importing task packages, matching Isaac Lab script conventions.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from prettytable import PrettyTable

    prefix = None if args_cli.all else args_cli.prefix

    try:
        import_registration_packages(args_cli.package)

        print_envs(gym.registry.values(), args_cli.keyword, prefix, PrettyTable)
    finally:
        simulation_app.close()


def print_envs(task_specs, keyword: str | None, prefix: str | None, table_cls) -> None:
    """Print matching Gymnasium environment specs."""
    table = table_cls(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Isaac Lab Task Maker Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in sorted(task_specs, key=lambda spec: spec.id):
        if matches_filters(task_spec.id, keyword, prefix):
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs.get("env_cfg_entry_point", ""),
                ]
            )
            index += 1

    print(table)


if __name__ == "__main__":
    main()
