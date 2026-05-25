# IL Task: LegoG1

IsaacLab IL extension for the **IL-LEGO-G1-v0** task: a Unitree G1 humanoid
with Inspire FTP hands picks up two Lego bricks and places them on per-arm
targets.

Generated from `source/IsaacLabTaskMaker/.../templates/task_definitions/g1_lego.yaml`
via `scripts/create_task.py`, then hand-tuned (subtask thresholds, gripper
joint pattern, etc.).

## Installation

```bash
cd examples/lego_g1
pip install -e source/lego_g1
```

The editable install drops a `.pth` file that auto-imports `lego_g1` so
IsaacLab's stock scripts (and the ones in this folder) find the registered
gym envs without any extra wiring.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/teleop.py` | Drive the env interactively (keyboard / spacemouse / handtracking). |
| `scripts/record_annotated_demos.py` | Record standard-Mimic demos with online subtask annotations. Each EEF queue advances when its head predicate dwells true for `--default_signal_dwell` steps. Save with `N` (or XR `RESET` after completion). |

Examples:

```bash
# Keyboard teleop
./isaaclab.sh -p examples/lego_g1/scripts/teleop.py --task IL-LEGO-G1-v0

# Annotated dataset recording (XR handtracking)
./isaaclab.sh -p examples/lego_g1/scripts/record_annotated_demos.py \
    --task IL-LEGO-G1-v0-Mimic \
    --teleop_device dualhandtracking_abs \
    --enable_pinocchio \
    --dataset_file ./datasets/lego_g1_annotated.hdf5
```

## Scene assets

The scene attribute names differ from the IsaacLab defaults — match these
when reading from `env.scene[...]` or writing `SceneEntityCfg(...)` terms:

| Asset | Scene name | Notes |
|---|---|---|
| Robot | `unitree_g1` | Set via `robot.name` in the source YAML. |
| Left brick (red, y=+0.2) | `red_brick` | Assigned to the LEFT arm queue. |
| Right brick (blue, y=−0.2) | `blue_brick` | Assigned to the RIGHT arm queue. |

The gripper-closed predicate averages the thumb (`*_thumb_proximal_pitch_joint`),
index, and middle proximal joints per arm — see `tasks/manager_based/lego_g1/lego_g1_cfg.py`.

## Project structure

```
source/lego_g1/lego_g1/
├── assets/          # USD scene, object, and robot files (ignored by git)
├── base_il_env/     # BaseILEnv + BaseILEnvCfg, generated per-task
└── tasks/           # Task-specific cfg and MDP functions
    └── manager_based/lego_g1/
        ├── lego_g1_cfg.py        # Scene, actions, observations, subtask thresholds
        ├── lego_g1_mimic_cfg.py  # Per-arm SubTaskConfig list
        └── mdp/                  # Subtask predicates, custom obs/terminations
```

## Customization

- **Subtask thresholds and joint patterns**: `tasks/manager_based/lego_g1/lego_g1_cfg.py`
  (`_GRIPPER_CLOSED_THRESHOLD`, `_DIST_TO_BRICK_THRESHOLD`, `_LEFT_GRIPPER_JOINTS`, …).
- **Subtask predicates**: `tasks/manager_based/lego_g1/mdp/observations.py`
  (`grasp_brick_done`, `move_brick_done`, `release_brick_done`, `idle_done`).
- **Per-arm Mimic subtask layout**: `tasks/manager_based/lego_g1/lego_g1_mimic_cfg.py`.
- **USD payloads**: drop new files under `assets/scenes/` and `assets/objects/`
  (gitignored). Only `assets/__init__.py` (path constants) is tracked.
