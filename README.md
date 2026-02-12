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

1. Create a new folder under `tasks/manager_based/`:

    ```
    tasks/manager_based/my_task/
    ├── __init__.py       ← gym.register()
    └── my_task_cfg.py    ← your config (copy from tm7_g1)
    ```

2. Copy the example task config:

    ```bash
    cp source/IsaacLabTaskMaker/isaaclab_task_maker/tasks/manager_based/tm7_g1/tm7_g1_cfg.py \
       source/IsaacLabTaskMaker/isaaclab_task_maker/tasks/manager_based/my_task/my_task_cfg.py
    ```

3. In `my_task_cfg.py`:
    - Swap the **robot** `ArticulationCfg` import
    - Update **USD scene/object paths**
    - Set IK **joint names** and **EEF link names**
    - Configure `eef_names`, `eef_action_slices`, `eef_gripper_slices`

4. Create `__init__.py` with `gym.register()`:

    ```python
    import gymnasium as gym

    gym.register(
        id="IL-MyRobot-MyTask-v0",
        entry_point="isaaclab_task_maker.tasks.manager_based.base_il_env.base_il_env:BaseILEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.my_task_cfg:MyTaskCfg",
        },
    )
    ```

5. Reinstall and run:

    ```bash
    pip install -e source/IsaacLabTaskMaker
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task IL-MyRobot-MyTask-v0
    ```

## Project Structure

```
source/IsaacLabTaskMaker/
├── config/extension.toml           ← extension metadata & dependencies
├── setup.py                        ← pip install configuration
└── isaaclab_task_maker/
    ├── __init__.py                  ← package entry point
    ├── assets/                      ← USD scenes, objects, robots
    │   ├── scenes/
    │   ├── objects/
    │   └── robots/
    └── tasks/
        └── manager_based/
            ├── base_il_env/         ← shared base (DO NOT MODIFY)
            │   ├── base_il_env.py   ← BaseILEnv class
            │   ├── base_il_env_cfg.py
            │   └── mdp/
            │       └── observations.py
            └── tm7_g1/              ← example task
                ├── __init__.py
                └── tm7_g1_cfg.py
```

## License

BSD-3-Clause