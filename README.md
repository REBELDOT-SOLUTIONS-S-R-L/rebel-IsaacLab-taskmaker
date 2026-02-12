# IsaacLab Task Maker

Config-only imitation learning task creation for [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

Scaffold new IL tasks from **YAML configs and templates** — a script generates the boilerplate (scene, robot, observations, actions), and you customize the resulting Python code for task-specific needs.

## Features

- **Config-only tasks** — define new tasks by copying an example config and swapping robot/scene/joints
- **Shared base environment** — `BaseILEnv` implements all 4 mandatory `ManagerBasedRLMimicEnv` methods once
- **Modular folder structure** — each task lives in its own folder under `tasks/manager_based/`
- **Built-in MDP helpers** — EEF pose, object observation, and joint state functions ready to use
- **XR teleoperation support** — OpenXR hand tracking and ManusVive out of the box

## Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) installed (conda or uv recommended)
- Python 3.10+

## Installation

1. **Clone this repository** (outside the Isaac Lab directory):

    ```bash
    git clone https://github.com/REBELDOT-SOLUTIONS-S-R-L/ROBOTICS-IsaacLab-Task-Maker.git
    cd ROBOTICS-IsaacLab-Task-Maker
    ```

2. **Install the extension** in your Isaac Lab conda environment:

    ```bash
    conda activate <your-isaaclab-env>      # e.g. conda activate leisaac
    pip install -e source/IsaacLabTaskMaker
    ```

3. **Add the import** to your teleoperation script (one-time setup):

    In `IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py`, add after the `import isaaclab_tasks` line:

    ```python
    import isaaclab_task_maker  # noqa: F401
    ```

4. **Verify** the installation:

    ```bash
    python scripts/list_envs.py
    ```

    You should see `IL-TM7-G1-v0` (or any registered tasks) in the output.

## Usage

### Running teleoperation

```bash
cd <IsaacLab-directory>
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task IL-TM7-G1-v0 \
    --enable_pinocchio \
    --teleop_device handtracking
```

### Creating a new task

Tasks are generated from **YAML definition files** using the `create_task.py` script.

#### 1. Write a YAML definition

Create a YAML file in `source/IsaacLabTaskMaker/isaaclab_task_maker/templates/task_definitions/`:

```yaml
task_name: my_task          # folder name (snake_case)
task_id: IL-MyTask-v0       # gymnasium environment ID

robot:
  import_path: isaaclab_assets.robots.fourier
  config_name: GR1T2_CFG
  prim_path: "/World/envs/env_.*/Robot"
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]

scene:
  usd_file: scene.usd
  scale: [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]

ik_controller:
  controller_type: pink_ik    # pink_ik | differential_ik | joint_position | ...
  base_link_name: base_link
  controlled_joint_names: [...]
  hand_joint_names: [...]
  num_hand_joints: 22
```

See `templates/task_definitions/gr1t.yaml` for a full example.

##### Custom robot from USD

Instead of importing a pre-defined `ArticulationCfg`, you can load a robot from a local USD file:

```yaml
robot:
  usd_file: my_robot.usd     # loaded from assets/robots/<task_name>/
  prim_path: "/World/envs/env_.*/Robot"
  scale: [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]
```

See `templates/task_definitions/example_custom_robot.yaml` for a full example.

#### 2. Generate the task

```bash
# Preview (no files written):
python scripts/create_task.py my_task.yaml --dry-run

# Generate:
python scripts/create_task.py my_task.yaml
```

This creates:

```
tasks/manager_based/my_task/
├── __init__.py           ← gym.register()
├── my_task_cfg.py        ← full task config (scene, robot, actions, observations, ...)
└── mdp/
    ├── __init__.py
    ├── observations.py   ← task-specific observation functions
    ├── events.py         ← task-specific event/reset functions
    └── terminations.py   ← task-specific termination conditions
```

Asset folders are also created under `assets/scenes/`, `assets/objects/`, and `assets/robots/`.

#### 3. Place USD files

Copy your scene, object, and (if using custom USD) robot USD files into the asset folders printed by the script.

#### 4. Install and run

```bash
pip install -e source/IsaacLabTaskMaker

cd <IsaacLab-directory>
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task IL-MyTask-v0
```

#### Supported controllers

| Controller | `controller_type` |
|---|---|
| Pink IK (with XR teleop) | `pink_ik` |
| Differential IK | `differential_ik` |
| Operational Space | `operational_space` |
| RMPFlow | `rmpflow` |
| Joint Position | `joint_position` |
| Relative Joint Position | `relative_joint_position` |
| Joint Velocity | `joint_velocity` |
| Joint Effort | `joint_effort` |

## Project Structure

```
source/IsaacLabTaskMaker/
├── config/extension.toml             ← extension metadata & dependencies
├── setup.py                          ← pip install configuration
└── isaaclab_task_maker/
    ├── __init__.py                    ← package entry point
    ├── assets/                        ← USD scenes, objects, robots
    │   ├── scenes/<task_name>/
    │   ├── objects/<task_name>/
    │   └── robots/<task_name>/
    ├── templates/
    │   ├── skeleton_task_cfg.py.template  ← code generation template
    │   └── task_definitions/             ← YAML task definitions
    │       ├── gr1t.yaml
    │       ├── example.yaml
    │       └── example_custom_robot.yaml
    └── tasks/
        └── manager_based/
            ├── base_il_env/           ← shared base (DO NOT MODIFY)
            │   ├── base_il_env.py
            │   ├── base_il_env_cfg.py
            │   └── mdp/
            └── <task_name>/           ← generated tasks
                ├── __init__.py
                ├── <task_name>_cfg.py
                └── mdp/
```

## License

BSD-3-Clause