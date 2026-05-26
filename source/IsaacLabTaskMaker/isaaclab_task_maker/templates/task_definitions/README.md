# Task Definition Examples

This directory contains example YAML task definitions for generating Isaac Lab imitation learning tasks. Each file is a complete, runnable example that demonstrates a specific **robot + controller** combination.

## Quick Start

```bash
# Preview what will be generated (no files written):
python scripts/create_task.py <example>.yaml --dry-run

# Generate the task:
python scripts/create_task.py <example>.yaml

# Overwrite an existing task:
python scripts/create_task.py <example>.yaml --force
```

## Examples by Controller Type

| Controller | File | Robot | Description |
|---|---|---|---|
| **pink_ik** | `example.yaml` | Unitree G1 (humanoid) | Full-body IK with XR hand tracking, null-space posture control |
| **pink_ik** | `gr1t.yaml` | Fourier GR1T2 (humanoid) | Same as above, different humanoid — compare joint naming |
| **differential_ik** | `franka_panda.yaml` | Franka Panda (7-DOF) | Jacobian-based IK with DLS method, keyboard teleop |
| **differential_ik** | `so100.yaml` | SO-100 (5-DOF) | Diff IK on a low-cost arm, shows 5-DOF considerations |
| **operational_space** | `franka_operational_space.yaml` | Franka Panda (7-DOF) | Impedance control for contact-rich tasks |
| **rmpflow** | `franka_rmpflow.yaml` | Franka Panda (7-DOF) | Collision-aware reactive motion |
| **joint_position** | `franka_joint_position.yaml` | Franka Panda (7-DOF) | Direct joint angle commands |
| **joint_position** | `so100_joint_position.yaml` | SO-100 (5-DOF) | Joint-space control for low-cost arm |
| **relative_joint_position** | `franka_relative_joint_position.yaml` | Franka Panda (7-DOF) | Incremental joint commands (delta actions) |
| **joint_velocity** | `franka_joint_velocity.yaml` | Franka Panda (7-DOF) | Joint angular velocity commands |
| **joint_effort** | `franka_joint_effort.yaml` | Franka Panda (7-DOF) | Direct joint torque commands |

## Examples by Robot Type

### Humanoid Robots (dual-arm, dexterous hands)
- `example.yaml` — Unitree G1 + Pink IK
- `gr1t.yaml` — Fourier GR1T2 + Pink IK

### 7-DOF Arms (single-arm with gripper)
- `franka_panda.yaml` — Differential IK (most common)
- `franka_operational_space.yaml` — Operational Space Controller
- `franka_rmpflow.yaml` — RMPFlow
- `franka_joint_position.yaml` — Joint Position
- `franka_relative_joint_position.yaml` — Relative Joint Position
- `franka_joint_velocity.yaml` — Joint Velocity
- `franka_joint_effort.yaml` — Joint Effort

### 5-DOF Low-Cost Arms (SO-100 / SO-101 family)
- `so100.yaml` — Differential IK
- `so100_joint_position.yaml` — Joint Position

## Choosing a Controller

```
                        ┌─────────────────────────────┐
                        │  What type of robot?        │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    ▼              ▼                   ▼
              Humanoid        Single Arm           Low-Cost Arm
             (G1, GR1T)     (Franka, UR)         (SO-100/101)
                    │              │                   │
                    ▼              │                   │
               pink_ik            │                   │
          (XR hand tracking)      │                   │
                                  │                   │
                    ┌─────────────┼───────────────┐   │
                    │             │               │   │
                    ▼             ▼               ▼   ▼
             Need force     Need collision    Simple replay /
              control?      avoidance?        RL policy?
                    │             │               │
                    ▼             ▼               ▼
           operational_space  rmpflow      joint_position
                                           (or relative_joint_position
                                            for RL delta actions)
                    │
                    │  Default for most tasks:
                    └──────► differential_ik
```

### Controller Summary

| Controller | Action Space | Best For |
|---|---|---|
| `pink_ik` | Task-space (pose) | Humanoids with XR hand tracking |
| `differential_ik` | Task-space (pose/position) | General single-arm manipulation |
| `operational_space` | Task-space (pose) | Contact-rich tasks, force control |
| `rmpflow` | Task-space (pose) | Cluttered environments, collision avoidance |
| `joint_position` | Joint-space (angles) | Trajectory replay, RL policies |
| `relative_joint_position` | Joint-space (deltas) | RL policies with bounded actions |
| `joint_velocity` | Joint-space (velocities) | Velocity servoing, smooth motion |
| `joint_effort` | Joint-space (torques) | Advanced force control, research |

## YAML Structure Reference

Every task definition has these sections:

```yaml
# ─── REQUIRED ────────────────────────────────────────────────
task_name: my_task              # snake_case → becomes folder name + class prefix
task_id: IL-MyTask-v0           # Gymnasium environment ID for registration

robot:
  import_path: isaaclab_assets.robots.<package>   # Python import path
  config_name: ROBOT_CFG_NAME                     # ArticulationCfg variable
  prim_path: "/World/envs/env_.*/Robot"            # USD scene path pattern
  init_pos: [x, y, z]                             # meters
  init_rot: [w, x, y, z]                          # quaternion

ik_controller:
  controller_type: <controller>                    # see table above
  controlled_joint_names: [...]                    # joints the controller moves

eef:
  names: ["gripper"]                               # or ["left", "right"] for bimanual
  target_links:
    gripper: <link_name>                           # robot link for EEF tracking

# ─── OPTIONAL (with defaults) ────────────────────────────────
scene:                          # defaults to scene.usd at identity transform
  usd_file: scene.usd
  scale: [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]

scene_objects: []               # list of RigidObjectCfg entries
                                # optional per-object reset:
                                #   reset:
                                #     sampler: uniform | sobol  # default uniform
                                #     seed: 0                   # Sobol seed
                                #     pos_range: {x: [-0.03, 0.03], y: [-0.03, 0.03]}
                                #     rot_range: {rz: [-0.2, 0.2]}

resets:
  seed: 0                       # global default for reset.seed

observations:
  eef_link_names:               # links to observe (usually same as target_links)
    gripper: <link_name>

teleop:
  device: keyboard              # keyboard | spacemouse | gamepad | handtracking

sim:
  decimation: 6                 # sim steps per policy step
  episode_length_s: 20.0
  dt: 0.008333                  # physics timestep (1/120)
  render_interval: 2
  env_spacing: 2.5              # meters between parallel envs
```

### Controller-Specific Settings

**differential_ik** (add to `ik_controller`):
```yaml
ik_method: dls                  # dls | svd | pinv
command_type: pose              # pose (6-DOF) | position (3-DOF)
use_relative_mode: false
body_offset: [0.0, 0.0, 0.107] # offset to fingertip
```

**operational_space** (add to `ik_controller`):
```yaml
body_offset: [0.0, 0.0, 0.107]
nullspace_joint_pos_target: "zero"
```

**rmpflow** (add to `ik_controller`):
```yaml
body_offset: [0.0, 0.0, 0.107]
use_relative_mode: false
```

**pink_ik** (add to `ik_controller` and `eef`):
```yaml
# ik_controller:
hand_joint_names: [...]         # finger joints for XR hand tracking
num_hand_joints: 24             # total finger joint count
null_space_joints: [...]        # joints for posture stability

# eef:
frame_names:                    # URDF-prefixed link names (required by Pink)
  left: urdf_prefix_left_link
  right: urdf_prefix_right_link

# teleop:
device: handtracking
retargeter_import: isaaclab.devices.openxr.retargeters...
retargeter_class: RetargeterCfg
```

## Creating Your Own Task

1. **Copy the closest example** to a new YAML file
2. **Change `task_name` and `task_id`** to unique values
3. **Update the robot** section with your robot's config
4. **List the correct joint names** for your robot in `ik_controller`
5. **Set the EEF link** in `eef.target_links`
6. **Preview**: `python scripts/create_task.py your_task.yaml --dry-run`
7. **Generate**: `python scripts/create_task.py your_task.yaml`
8. **Place USD files** in the asset folders printed by the script
9. **Install**: `pip install -e source/IsaacLabTaskMaker`
10. **Run**: `./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task <task_id>`
