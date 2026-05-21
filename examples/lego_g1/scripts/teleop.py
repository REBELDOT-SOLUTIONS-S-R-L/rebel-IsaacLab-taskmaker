# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LegoG1-aware teleop launcher.

Forked from `IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py`.
The only functional change vs upstream: when XR is enabled, IsaacLab strips
every CameraCfg from the scene (`remove_camera_configs`). For LegoG1 we want
the head + wrist cameras to remain, so we re-attach them after the strip via
`attach_cameras` from the task config module.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperation for LegoG1 (with cameras retained under XR).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default="IL-LEGO-G1-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./datasets/lego_g1",
    help="Directory where the recorded HDF5 dataset will be written.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="dataset",
    help="Filename (without extension) for the recorded dataset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# Ensure the LegoG1 task is registered with gym before parse_env_cfg looks it up.
import lego_g1.tasks  # noqa: F401
from lego_g1.tasks.manager_based.lego_g1.lego_g1_cfg import attach_cameras

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

logger = logging.getLogger(__name__)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # Keep the time_out termination enabled so episodes auto-reset at
    # `episode_length_s` (set in LegoG1TaskCfg.__post_init__).
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"
        # Re-attach LegoG1 cameras after the XR strip. Upstream removes them
        # because the XR compositor and replicator can fight over the RTX
        # render queue; if you hit frame drops / black frames, comment this out.
        attach_cameras(env_cfg.scene)
        logger.info("Re-attached LegoG1 cameras after XR camera strip.")

    # The recorder + entity_order + eef_names are now defaults set by
    # BaseILEnvCfg. We only override the per-run dataset path/filename here.
    env_cfg.recorders.dataset_export_dir_path = args_cli.dataset_dir
    env_cfg.recorders.dataset_filename = args_cli.dataset_file
    logger.info(f"Recording to {args_cli.dataset_dir}/{args_cli.dataset_file}.hdf5")

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    if args_cli.xr:
        teleoperation_active = False
    else:
        teleoperation_active = True

    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()

                if teleoperation_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
