# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record standard Mimic demonstrations with online subtask annotations.

The script records a standard-schema annotated Mimic dataset directly during
teleoperation.  Each end-effector owns an independent queue derived from
``env.cfg.subtask_configs``.  At every step only the head signal of each queue
is checked against raw environment predicates.  When a head signal remains true
for its configured dwell, that signal is latched and only that queue advances.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
import traceback
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record standard Mimic demos with online annotations.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here or via the environment config. Built-ins: keyboard, spacemouse, gamepad."
    ),
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/annotated_dataset.hdf5",
    help="File path to export recorded annotated demos.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Teleop sensitivity factor.")
parser.add_argument(
    "--default_signal_dwell",
    type=int,
    default=3,
    help="Default consecutive true steps required before latching a subtask signal.",
)
parser.add_argument(
    "--signal_dwell",
    action="append",
    default=[],
    metavar="SIGNAL=STEPS",
    help="Per-signal dwell override. Can be passed multiple times.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

if "handtracking" in args_cli.teleop_device.lower():
    args_cli.xr = True
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import StandardAnnotatedMimicRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import lego_g1  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


def log_status(level: int, message: str, *args) -> None:
    """Log and print operator-facing status messages."""
    logger.log(level, message, *args)
    if args:
        message = message % args
    print(message, flush=True)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


@dataclass
class OnlineSubtaskAnnotationState:
    """Per-episode online subtask annotation state."""

    eef_queues: dict[str, list[str]]
    device: torch.device | str
    default_signal_dwell: int = 3
    signal_dwells: dict[str, int] = field(default_factory=dict)
    latched_signals: dict[str, bool] = field(init=False)
    eef_queue_indices: dict[str, int] = field(init=False)
    consecutive_true_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        signal_names = [signal for queue in self.eef_queues.values() for signal in queue]
        duplicate_names = sorted({signal for signal in signal_names if signal_names.count(signal) > 1})
        if duplicate_names:
            raise ValueError(
                "Subtask term signal names must be globally unique for standard annotated teleop. "
                f"Duplicates: {duplicate_names}"
            )
        if not signal_names:
            raise ValueError("No subtask term signals found in env.cfg.subtask_configs.")
        self.latched_signals = {signal: False for signal in signal_names}
        self.eef_queue_indices = {eef_name: 0 for eef_name in self.eef_queues}
        self.consecutive_true_counts = {eef_name: 0 for eef_name in self.eef_queues}

    @classmethod
    def from_env(
        cls,
        env: ManagerBasedRLEnv,
        *,
        default_signal_dwell: int,
        signal_dwells: dict[str, int],
    ) -> "OnlineSubtaskAnnotationState":
        subtask_cfgs = getattr(getattr(env, "cfg", None), "subtask_configs", {})
        if not isinstance(subtask_cfgs, dict) or not subtask_cfgs:
            raise ValueError("Annotated teleop requires env.cfg.subtask_configs to define one queue per EEF.")

        eef_queues: dict[str, list[str]] = {}
        for eef_name, cfgs in subtask_cfgs.items():
            queue = [
                str(getattr(cfg, "subtask_term_signal"))
                for cfg in cfgs
                if getattr(cfg, "subtask_term_signal", None)
            ]
            if queue:
                eef_queues[str(eef_name)] = queue

        return cls(
            eef_queues=eef_queues,
            device=env.device,
            default_signal_dwell=default_signal_dwell,
            signal_dwells=signal_dwells,
        )

    def reset(self) -> None:
        for signal_name in self.latched_signals:
            self.latched_signals[signal_name] = False
        for eef_name in self.eef_queue_indices:
            self.eef_queue_indices[eef_name] = 0
            self.consecutive_true_counts[eef_name] = 0

    def current_signal_heads(self) -> dict[str, str | None]:
        return {eef_name: self._head_signal_for_eef(eef_name) for eef_name in self.eef_queues}

    def is_complete(self) -> bool:
        return all(
            int(self.eef_queue_indices.get(eef_name, 0)) >= len(queue)
            for eef_name, queue in self.eef_queues.items()
        )

    def as_tensor_dict(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            num_envs = 1
        elif isinstance(env_ids, torch.Tensor):
            num_envs = int(env_ids.numel())
        elif isinstance(env_ids, slice):
            num_envs = 1
        else:
            num_envs = len(env_ids)
        values = {}
        for signal_name, latched in self.latched_signals.items():
            values[signal_name] = torch.full((num_envs, 1), bool(latched), dtype=torch.bool, device=self.device)
        return values

    def advance(self, raw_signal_reader: Callable[[], dict[str, Any]]) -> list[str]:
        raw_signals = raw_signal_reader()
        newly_latched: list[str] = []

        for eef_name in self.eef_queues:
            signal_name = self._head_signal_for_eef(eef_name)
            if signal_name is None:
                self.consecutive_true_counts[eef_name] = 0
                continue

            if signal_name not in raw_signals:
                available = sorted(raw_signals.keys())
                raise KeyError(
                    f"Raw subtask predicates did not include queue head '{signal_name}' for EEF '{eef_name}'. "
                    f"Available signals: {available}"
                )

            is_true = bool(torch.as_tensor(raw_signals[signal_name], device=self.device).reshape(-1)[0].item())
            if is_true:
                self.consecutive_true_counts[eef_name] += 1
            else:
                self.consecutive_true_counts[eef_name] = 0

            if self.consecutive_true_counts[eef_name] < self._required_dwell(signal_name):
                continue

            self.latched_signals[signal_name] = True
            self.eef_queue_indices[eef_name] += 1
            self.consecutive_true_counts[eef_name] = 0
            newly_latched.append(signal_name)

        return newly_latched

    def _head_signal_for_eef(self, eef_name: str) -> str | None:
        queue = self.eef_queues.get(eef_name, [])
        index = int(self.eef_queue_indices.get(eef_name, 0))
        if index >= len(queue):
            return None
        return queue[index]

    def _required_dwell(self, signal_name: str) -> int:
        return max(1, int(self.signal_dwells.get(signal_name, self.default_signal_dwell)))


def parse_signal_dwells(entries: Sequence[str]) -> dict[str, int]:
    signal_dwells: dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected --signal_dwell entries as SIGNAL=STEPS, got: {entry!r}")
        signal_name, dwell_text = entry.split("=", maxsplit=1)
        signal_name = signal_name.strip()
        if not signal_name:
            raise ValueError(f"Signal name cannot be empty in --signal_dwell entry: {entry!r}")
        dwell = int(dwell_text)
        if dwell < 1:
            raise ValueError(f"Signal dwell must be >= 1 for {signal_name!r}, got {dwell}.")
        signal_dwells[signal_name] = dwell
    return signal_dwells


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str) -> ManagerBasedRLEnvCfg:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Annotated teleop recording is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}."
        )

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.recorders = StandardAnnotatedMimicRecorderManagerCfg()
    env_cfg.recorders.record_pre_step_subtask_start_signals = None
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg


def setup_teleop_device(env_cfg: ManagerBasedRLEnvCfg, callbacks: dict[str, Callable]) -> object:
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

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
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    for key, callback in callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to add callback for key %s: %s", key, exc)

    return teleop_interface


def is_unimplemented_mimic_method(env: ManagerBasedRLEnv, method_name: str) -> bool:
    method = getattr(env, method_name, None)
    method_func = getattr(method, "__func__", None)
    base_method = getattr(ManagerBasedRLMimicEnv, method_name, None)
    return method_func is not None and method_func is base_method


def validate_env_contract(env: ManagerBasedRLEnv) -> None:
    required_methods = ["get_robot_eef_pose", "get_object_poses", "action_to_target_eef_pose"]
    missing_methods = [method_name for method_name in required_methods if not hasattr(env, method_name)]
    if missing_methods:
        raise TypeError(
            "Annotated teleop requires a manager-based env with standard Mimic recording APIs. "
            f"Missing methods: {missing_methods}."
        )
    unimplemented_methods = [
        method_name
        for method_name in ("get_robot_eef_pose", "action_to_target_eef_pose")
        if is_unimplemented_mimic_method(env, method_name)
    ]
    if unimplemented_methods:
        raise NotImplementedError(
            "Annotated teleop requires environment-specific Mimic API implementations. "
            f"Unimplemented methods: {unimplemented_methods}."
        )
    has_raw_predicates = hasattr(env, "get_subtask_term_predicates")
    has_term_signals = hasattr(env, "get_subtask_term_signals") and not is_unimplemented_mimic_method(
        env, "get_subtask_term_signals"
    )
    if not has_raw_predicates and not has_term_signals:
        raise NotImplementedError(
            "Annotated teleop requires raw subtask predicates from get_subtask_term_predicates(env_ids=None), "
            "or an existing get_subtask_term_signals(env_ids=None) implementation to use as the raw source."
        )


def install_standard_mimic_method_adapters(env: ManagerBasedRLEnv) -> None:
    if not is_unimplemented_mimic_method(env, "get_object_poses"):
        return

    base_get_object_poses = env.get_object_poses

    def get_object_poses(_env, env_ids: Sequence[int] | None = None):
        return base_get_object_poses(env_ids=env_ids)

    env.get_object_poses = types.MethodType(get_object_poses, env)


def make_raw_signal_reader(env: ManagerBasedRLEnv) -> Callable[[], dict[str, Any]]:
    if hasattr(env, "get_subtask_term_predicates"):
        raw_signal_method = env.get_subtask_term_predicates
    else:
        raw_signal_method = env.get_subtask_term_signals

    def read_raw_signals() -> dict[str, Any]:
        return raw_signal_method(env_ids=[0])

    return read_raw_signals


def install_latched_signal_adapter(env: ManagerBasedRLEnv, annotator: OnlineSubtaskAnnotationState) -> None:
    def get_latched_subtask_term_signals(_env, env_ids: Sequence[int] | None = None):
        return annotator.as_tensor_dict(env_ids=env_ids)

    env.get_subtask_term_signals = types.MethodType(get_latched_subtask_term_signals, env)


def format_progress(annotator: OnlineSubtaskAnnotationState) -> str:
    parts = []
    for eef_name, queue in annotator.eef_queues.items():
        index = int(annotator.eef_queue_indices.get(eef_name, 0))
        head = annotator.current_signal_heads().get(eef_name)
        next_label = head if head is not None else "complete"
        parts.append(f"{eef_name}: {index}/{len(queue)} next={next_label}")
    return " | ".join(parts)


def reset_episode(env: ManagerBasedRLEnv, teleop_interface: object, annotator: OnlineSubtaskAnnotationState) -> None:
    env.sim.reset()
    env.recorder_manager.reset()
    annotator.reset()
    env.reset()
    teleop_interface.reset()


def normalize_action(env: ManagerBasedRLEnv, action: Any) -> torch.Tensor | None:
    if action is None:
        return None
    action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    return action_tensor.repeat(env.num_envs, 1)


def export_successful_episode(env: ManagerBasedRLEnv) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])


def main() -> int:
    if args_cli.default_signal_dwell < 1:
        raise ValueError(f"--default_signal_dwell must be >= 1, got {args_cli.default_signal_dwell}.")

    output_dir, output_file_name = setup_output_directories()
    log_status(logging.INFO, "Creating environment config for task: %s", args_cli.task)
    env_cfg = create_environment_config(output_dir, output_file_name)
    env: ManagerBasedRLEnv | None = None

    try:
        log_status(logging.INFO, "Creating environment instance.")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        log_status(logging.INFO, "Environment created: %s", type(env).__name__)
        if not isinstance(env, ManagerBasedRLEnv):
            raise ValueError(
                "Annotated teleop recording requires a ManagerBasedRLEnv environment. "
                f"Received: {type(env).__name__}."
            )
        log_status(logging.INFO, "Validating Mimic environment contract.")
        validate_env_contract(env)
        install_standard_mimic_method_adapters(env)

        signal_dwells = parse_signal_dwells(args_cli.signal_dwell)
        annotator = OnlineSubtaskAnnotationState.from_env(
            env,
            default_signal_dwell=args_cli.default_signal_dwell,
            signal_dwells=signal_dwells,
        )
        raw_signal_reader = make_raw_signal_reader(env)
        install_latched_signal_adapter(env, annotator)
        log_status(logging.INFO, "Annotation queues configured: %s", format_progress(annotator))

        recorded_demo_count = 0
        recording_active = False
        completion_announced = False
        flags = {"start": False, "save": False, "discard": False, "abort": False, "reset": False}

        def on_start() -> None:
            flags["start"] = True
            log_status(logging.INFO, "[START] Recording start requested.")

        def on_save() -> None:
            flags["save"] = True
            log_status(logging.INFO, "[SAVE] Save requested.")

        def on_discard() -> None:
            flags["discard"] = True
            log_status(logging.INFO, "[DISCARD] Discard requested.")

        def on_abort() -> None:
            flags["abort"] = True
            log_status(logging.WARNING, "[ESC] Abort requested.")

        def on_reset() -> None:
            if recording_active and annotator.is_complete():
                flags["save"] = True
                log_status(logging.INFO, "[RESET] Save requested because annotation queues are complete.")
                return
            flags["reset"] = True
            log_status(logging.INFO, "[RESET] Reset requested.")

        callbacks = {
            "S": on_start,
            "N": on_save,
            "D": on_discard,
            "R": on_reset,
            "RESET": on_reset,
            "START": on_start,
            "STOP": on_save,
            "ESCAPE": on_abort,
        }
        log_status(logging.INFO, "Creating teleop device: %s", args_cli.teleop_device)
        teleop_interface = setup_teleop_device(env_cfg, callbacks)

        rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None
        log_status(logging.INFO, "Resetting annotated recording episode.")
        reset_episode(env, teleop_interface, annotator)

        log_status(logging.INFO, "Using teleop device: %s", teleop_interface)
        if args_cli.xr:
            log_status(
                logging.INFO,
                "XR controls: START begins recording, RESET saves after queues complete or discards before completion.",
            )
        else:
            log_status(
                logging.INFO,
                "Press S to start, N to save after all queues complete, D/R to discard and reset, ESC to abort.",
            )
        log_status(logging.INFO, "Annotation queues: %s", format_progress(annotator))

        if not simulation_app.is_running():
            raise RuntimeError("Simulation app stopped before the teleop loop could start.")

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                if flags["abort"]:
                    break

                if flags["reset"] or flags["discard"]:
                    reset_episode(env, teleop_interface, annotator)
                    recording_active = False
                    completion_announced = False
                    flags["reset"] = False
                    flags["discard"] = False
                    flags["save"] = False
                    log_status(logging.INFO, "Episode discarded. Start a new attempt when ready.")
                    continue

                if flags["start"] and not recording_active:
                    recording_active = True
                    completion_announced = False
                    flags["start"] = False
                    log_status(logging.INFO, "Recording active. %s", format_progress(annotator))

                if flags["save"]:
                    if recording_active and annotator.is_complete():
                        export_successful_episode(env)
                        recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                        log_status(
                            logging.INFO,
                            "Saved annotated episode. Recorded %d successful demonstrations.",
                            recorded_demo_count,
                        )
                        if args_cli.num_demos > 0 and recorded_demo_count >= args_cli.num_demos:
                            break
                        reset_episode(env, teleop_interface, annotator)
                        recording_active = False
                        completion_announced = False
                        log_status(logging.INFO, "Ready for next episode. Start when ready.")
                    else:
                        log_status(
                            logging.WARNING,
                            "Save ignored because annotation queues are not complete. %s",
                            format_progress(annotator),
                        )
                    flags["save"] = False
                    continue

                if not recording_active:
                    env.sim.render()
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)
                    continue

                try:
                    action = normalize_action(env, teleop_interface.advance())
                except Exception as exc:
                    logger.error("Error in teleop interface: %s", exc, exc_info=True)
                    action = None

                if action is None:
                    env.sim.render()
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)
                    continue

                newly_latched = annotator.advance(raw_signal_reader)
                if newly_latched:
                    log_status(logging.INFO, "Latched: %s", ", ".join(newly_latched))
                    log_status(logging.INFO, "Annotation progress: %s", format_progress(annotator))

                # Publish the current queue heads so subtask obs functions only print
                # debug info for the signal each EEF is actively waiting on.
                env._debug_subtask_heads = {
                    signal for signal in annotator.current_signal_heads().values() if signal is not None
                }
                _, _, terminated, truncated, _ = env.step(action)

                if annotator.is_complete() and not completion_announced:
                    if args_cli.xr:
                        log_status(
                            logging.INFO,
                            "All annotation queues completed. Send RESET to save the episode.",
                        )
                    else:
                        log_status(
                            logging.INFO,
                            "All annotation queues completed. Press N to save the episode or D to re-record.",
                        )
                    completion_announced = True

                if bool(torch.any(terminated).item() or torch.any(truncated).item()):
                    log_status(logging.WARNING, "Episode terminated/truncated before save. Discarding current attempt.")
                    reset_episode(env, teleop_interface, annotator)
                    recording_active = False
                    completion_announced = False

                if rate_limiter is not None:
                    rate_limiter.sleep(env)

        if recording_active and annotator.is_complete():
            log_status(logging.INFO, "Saving completed episode before exit.")
            export_successful_episode(env)
            recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            log_status(
                logging.INFO,
                "Saved annotated episode. Recorded %d successful demonstrations.",
                recorded_demo_count,
            )

        return recorded_demo_count
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal recorder exit: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
