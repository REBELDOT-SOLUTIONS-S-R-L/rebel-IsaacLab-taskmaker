---
description: Create a new IL task for IsaacLabILEnvs (config-only workflow)
---

# Create a New IL Task

This workflow creates a new imitation learning task in the IsaacLabILEnvs extension. New tasks are **config-only** — you never need to write Python logic, only swap robot/scene/joints.

## Steps

1. **Copy the example task config**

   Copy `example_task_cfg.py` to a new file:
   ```bash
   cp source/IsaacLabILEnvs/IsaacLabILEnvs/tasks/manager_based/isaaclabilenvs/example_task_cfg.py \
      source/IsaacLabILEnvs/IsaacLabILEnvs/tasks/manager_based/isaaclabilenvs/<new_task>_cfg.py
   ```

2. **Swap the robot**

   In your new config file, change the robot `ArticulationCfg` import at the top:
   ```python
   # FROM:
   from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG
   # TO (example):
   from isaaclab_assets.robots.unitree import G1_INSPIRE_HAND_CFG
   ```

   Then update the `robot` field in `ExampleSceneCfg` to use the new config:
   ```python
   robot: ArticulationCfg = G1_INSPIRE_HAND_CFG.replace(
       prim_path="/World/envs/env_.*/Robot",
       init_state=ArticulationCfg.InitialStateCfg(
           pos=(0, 0, 0.75),  # adjust for your robot
           rot=(0.7071, 0, 0, 0.7071),
           joint_pos={...},  # set initial joint positions
       ),
   )
   ```

3. **Swap the scene (optional)**

   - Change `usd_path` for objects (table, manipulated object, etc.)
   - Add or remove `RigidObjectCfg` / `AssetBaseCfg` entries
   - Point `usd_path` to your custom USD scene from Isaac Sim

4. **Update joint names**

   In `ExampleActionsCfg`, update:
   - `pink_controlled_joint_names` — the arm joints your robot's IK controls
   - `hand_joint_names` — the finger/hand joints
   - `target_eef_link_names` — the end-effector link names in the URDF
   - `FrameTask` names in the IK controller config
   - Retargeter class (e.g., `GR1T2RetargeterCfg` → your robot's retargeter)

5. **Update EEF configuration**

   In your main config class (e.g., `MyTaskCfg`), set:

   ```python
   # Names of end-effectors (must match observation term prefixes)
   eef_names = ["left", "right"]

   # Where pos/quat live in the action tensor for each arm
   eef_action_slices = {
       "left":  {"pos": (0, 3),  "quat": (3, 7)},
       "right": {"pos": (7, 10), "quat": (10, 14)},
   }

   # Where gripper commands live in the action tensor
   eef_gripper_slices = {
       "left":  (14, 25),
       "right": (25, 36),
   }
   ```

6. **Update observations**

   In `ExampleObservationsCfg.PolicyCfg`, change:
   - `link_name` params in `left_eef_pos`, `left_eef_quat`, etc. to your robot's EEF link
   - `joint_names` in `hand_joint_state` to your robot's hand joints
   - Ensure obs term names follow the `{eef_name}_eef_pos` / `{eef_name}_eef_quat` convention

7. **Register the new task**

   In `__init__.py`, add:
   ```python
   gym.register(
       id="IL-MyRobot-MyTask-v0",
       entry_point=f"{__name__}.base_il_env:BaseILEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.<new_task>_cfg:MyTaskCfg",
       },
   )
   ```

8. **Test**

   Run with your teleoperation script:
   ```bash
   python scripts/environments/teleoperation/teleop.py --task IL-MyRobot-MyTask-v0
   ```

## File Structure

```
tasks/manager_based/isaaclabilenvs/
├── __init__.py              ← gym.register() calls
├── base_il_env.py           ← shared class (DO NOT MODIFY)
├── base_il_env_cfg.py       ← base config  (DO NOT MODIFY)
├── example_task_cfg.py      ← GR1T2 example (copy this!)
├── <new_task>_cfg.py        ← YOUR new task config
├── agents/
│   └── ...
└── mdp/
    ├── __init__.py
    └── observations.py      ← EEF/object observation functions
```
