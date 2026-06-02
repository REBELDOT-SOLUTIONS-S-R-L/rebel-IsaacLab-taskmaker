# IsaacLab Task Maker

A scaffolding tool for [Isaac Lab](https://github.com/isaac-sim/IsaacLab) imitation learning (IL) tasks. Instead of hand-writing the boilerplate that every new manipulation task needs — scene config, robot articulation, observation/action managers, MDP terms, gym registration — you describe the task in a single **YAML file** and a code generator produces a working Isaac Lab extension you can teleoperate immediately.

## What it solves

Setting up a new IL task in Isaac Lab usually means copying an existing example, renaming dozens of symbols, threading the robot through scene/observation/action configs, and re-implementing the four mandatory `ManagerBasedRLMimicEnv` methods. Task Maker collapses that into a YAML edit and a single command.

## How it works

1. You write a YAML task definition specifying the **robot**, **scene**, **controller**, **end-effector links**, and optional teleop/sim settings.
2. `scripts/create_task.py` renders Jinja2 templates into a fully-formed task folder (`__init__.py` with `gym.register()`, `<task>_cfg.py`, and an `mdp/` package).
3. A shared `BaseILEnv` implements the IL-specific environment hooks once, so generated tasks only carry task-specific code.
4. You drop USD files for scene/objects/robot into the printed asset folders, `pip install -e` the extension, and launch teleoperation.

## Supported controllers

Eight controller types are supported out of the box, covering the common spectrum:

| Controller | Action space | Typical use |
|---|---|---|
| `pink_ik` | Task-space pose | Humanoids (G1, GR1T2) with XR hand tracking |
| `differential_ik` | Task-space pose/position | General single-arm manipulation (default) |
| `operational_space` | Task-space pose | Contact-rich tasks, impedance/force control |
| `rmpflow` | Task-space pose | Collision-aware reactive motion |
| `joint_position` | Joint angles | Trajectory replay, RL policies |
| `relative_joint_position` | Joint deltas | RL with bounded actions |
| `joint_velocity` | Joint velocities | Velocity servoing |
| `joint_effort` | Joint torques | Force control, research |

Teleop devices: keyboard, spacemouse, gamepad, OpenXR hand tracking, ManusVive.

## Included examples

Ready-to-generate task definitions live in `templates/task_definitions/`:

- **Humanoids** — `g1.yaml`, `gr1t.yaml` (Pink IK + XR hand tracking)
- **Franka Panda 7-DOF** — one YAML per controller (diff IK, operational space, RMPFlow, joint position/velocity/effort, relative joint position)
- **SO-100 5-DOF** — `so100.yaml`, `so100_joint_position.yaml`

## Project layout

```
scripts/                   # create_task.py, remove_task.py, list_envs.py
source/IsaacLabTaskMaker/
  isaaclab_task_maker/
    assets/                # USD scenes, objects, robots (per task)
    templates/             # Jinja2 templates + YAML task definitions
    tasks/manager_based/
      base_il_env/         # shared IL base — implements the 4 mimic hooks
      <task_name>/         # generated tasks
```

## Quickstart

```bash
# 1. Install into your Isaac Lab conda env
pip install -e source/IsaacLabTaskMaker

# 2. Generate a task from YAML
python scripts/create_task.py franka_panda.yaml

# 3. Teleoperate
cd <IsaacLab-directory>
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task IL-FrankaPanda-v0 --teleop_device keyboard
```

A one-line addition to Isaac Lab's `teleop_se3_agent.py` (`import isaaclab_task_maker`) registers the generated tasks with Gymnasium.

## Status

Active development on the `dev` branch. License: BSD-3-Clause.
