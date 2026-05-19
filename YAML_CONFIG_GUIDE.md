# YAML Config Guide

How to write a task-definition YAML for IsaacLabTaskMaker. A YAML file is the
**only input** the generator needs — running `python scripts/create_task.py
<your_file>.yaml` turns it into a complete, installable Isaac Lab extension.

The full set of working examples lives in
`source/IsaacLabTaskMaker/isaaclab_task_maker/templates/task_definitions/`.
Copy the one closest to your robot/controller and edit from there.

---

## 1. The minimum viable config

These four sections are required. Everything else has sensible defaults.

```yaml
task_name: my_task                    # snake_case → folder + class prefix
task_id: IL-MyTask-v0                 # Gymnasium env ID

robot:
  import_path: isaaclab_assets.robots.franka   # Python module
  config_name: FRANKA_PANDA_HIGH_PD_CFG        # ArticulationCfg variable

ik_controller:
  controller_type: differential_ik             # see §4
  controlled_joint_names: ["panda_joint1", ..., "panda_joint7"]

eef:
  names: ["gripper"]
  target_links:
    gripper: panda_hand                        # robot link name
```

Generate it:

```bash
python scripts/create_task.py my_task.yaml --dry-run   # preview
python scripts/create_task.py my_task.yaml             # write files
python scripts/create_task.py my_task.yaml --force     # overwrite existing
```

---

## 2. Section-by-section reference

### `task_name` and `task_id` (required)

| Field | Format | Example | Becomes |
|---|---|---|---|
| `task_name` | `snake_case` | `tm7_franka` | folder name, package name, class prefix `TM7Franka` |
| `task_id` | Gymnasium ID | `IL-TM7-Franka-v0` | what you pass to `--task` |

### `robot` (required)

```yaml
robot:
  import_path: isaaclab_assets.robots.franka     # Python import path
  config_name: FRANKA_PANDA_HIGH_PD_CFG          # ArticulationCfg variable in that module
  prim_path: "/World/envs/env_.*/Robot"          # default; rarely changed
  init_pos: [0.0, 0.5, 0.4]                      # meters, [x, y, z]
  init_rot: [0.7071, 0.0, 0.0, -0.7071]          # quaternion, [w, x, y, z]
  scale:    [1.0, 1.0, 1.0]                      # optional
```

Only `import_path` and `config_name` are mandatory; the rest fall back to defaults.

### `scene` (optional)

The world USD that gets loaded once per environment.

```yaml
scene:
  usd_file: scene.usd            # placed at assets/scenes/<usd_file>
  scale:    [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]
```

If omitted, the generator assumes `scene.usd` at the origin.

### `scene_objects` (optional)

Rigid bodies the robot interacts with. List as many as you need.

```yaml
scene_objects:
  - name: tm7_bowl                                 # becomes a Python attribute
    type: RigidObjectCfg                           # almost always this
    prim_path: "{ENV_REGEX_NS}/Scene/tm7_bowl"     # {ENV_REGEX_NS} = per-env prefix
    usd_file: tm7_bowl_collisions.usd              # placed at assets/objects/<usd_file>
    scale:    [0.35, 0.35, 0.35]
    init_pos: [0.0, 0.5, 0.185]
    init_rot: [0.70711, 0.0, 0.0, 0.70711]
```

`usd_file` defaults to `<name>.usd` if omitted.

### `ik_controller` (required)

Common fields for every controller:

```yaml
ik_controller:
  controller_type: <one of §4>
  base_link_name: panda_link0                  # root of the kinematic chain
  controlled_joint_names: [...]                # joints the controller drives
  hand_joint_names: [...]                      # finger joints (optional)
  num_hand_joints: 2                           # set to len(hand_joint_names)
```

Per-controller extras: see §4.

### `eef` (required)

End-effectors the task tracks. One entry per EEF.

```yaml
eef:
  names: ["gripper"]                          # logical labels, free-form
  target_links:
    gripper: panda_hand                       # robot link to track
  frame_names:
    gripper: panda_hand                       # for pink_ik, use URDF-prefixed name
```

Bimanual humanoids:

```yaml
eef:
  names: ["left", "right"]
  target_links:
    left_wrist:  left_wrist_yaw_link
    right_wrist: right_wrist_yaw_link
  frame_names:                                # Pink IK requires URDF-prefixed names
    left:  g1_29dof_rev_1_0_left_wrist_yaw_link
    right: g1_29dof_rev_1_0_right_wrist_yaw_link
```

### `observations` (optional)

Which links to expose in the observation dict. Usually mirrors `target_links`.

```yaml
observations:
  eef_link_names:
    gripper: panda_hand
```

### `teleop` (optional)

```yaml
teleop:
  device: keyboard          # keyboard | spacemouse | gamepad | handtracking | manusvive
```

Default: `handtracking` if `controller_type: pink_ik`, otherwise `keyboard`.

For `handtracking` with `pink_ik` you must also set the retargeter:

```yaml
teleop:
  device: handtracking
  retargeter_import: isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter
  retargeter_class:  UnitreeG1RetargeterCfg
```

### `sim` (optional)

```yaml
sim:
  decimation: 6                # sim steps per policy step
  episode_length_s: 20.0
  dt: 0.008333                 # 1/120 — physics timestep
  render_interval: 2
  env_spacing: 2.5             # meters between parallel envs
```

---

## 3. Picking a controller

```
              What kind of robot?
                     │
        ┌────────────┼────────────┐
        ▼            ▼             ▼
     Humanoid    Single arm    Low-cost arm
      (G1)       (Franka)        (SO-100)
        │            │             │
        ▼            │             │
     pink_ik         │             │
                     ▼             ▼
              ┌──────┴──────┐  joint_position
              ▼      ▼      ▼  (or relative_joint_position)
            force  collide  default
            ctrl?  avoid?   │
              │      │      ▼
              ▼      ▼   differential_ik
        operational rmpflow
          _space
```

| Controller | Action space | Use for |
|---|---|---|
| `pink_ik` | task-space pose | Humanoids, dual-arm, finger control |
| `differential_ik` | task-space pose/position | General single-arm — **default choice** |
| `operational_space` | task-space pose + impedance | Contact-rich, force-regulated tasks |
| `rmpflow` | task-space pose | Cluttered scenes, collision avoidance |
| `joint_position` | joint angles | Trajectory replay, RL policies |
| `relative_joint_position` | joint deltas | RL with bounded incremental actions |
| `joint_velocity` | joint velocities | Velocity servoing, smooth motion |
| `joint_effort` | joint torques | Force control, research |

---

## 4. Controller-specific fields

### `differential_ik`

```yaml
ik_controller:
  controller_type: differential_ik
  ik_method: dls                 # dls (recommended) | svd | pinv
  command_type: pose             # pose (6-DOF) | position (3-DOF)
  use_relative_mode: false       # false = absolute, true = delta
  body_offset: [0.0, 0.0, 0.107] # body frame → fingertip
```

### `operational_space`

```yaml
ik_controller:
  controller_type: operational_space
  body_offset: [0.0, 0.0, 0.107]
  nullspace_joint_pos_target: "zero"   # or an explicit joint vector
```

### `rmpflow`

```yaml
ik_controller:
  controller_type: rmpflow
  body_offset: [0.0, 0.0, 0.107]
  use_relative_mode: false
```

### `pink_ik`

```yaml
ik_controller:
  controller_type: pink_ik
  base_link_name: pelvis
  controlled_joint_names: [...]      # arm joints (regex patterns OK)
  hand_joint_names:      [...]       # finger joints from XR retargeting
  num_hand_joints: 24
  null_space_joints:     [...]       # e.g. waist joints, for posture
```

`eef.frame_names` must use the URDF-prefixed link names.

### `joint_position`, `joint_velocity`, `joint_effort`, `relative_joint_position`

Only the common fields apply — `controlled_joint_names` is what matters.

```yaml
ik_controller:
  controller_type: joint_position
  controlled_joint_names: ["panda_joint1", ..., "panda_joint7"]
```

---

## 5. After generating

The script prints exact paths, but in summary:

1. **Drop your USD files** at the printed paths
   (`<project>/source/<pkg>/<pkg>/assets/scenes/` and `.../assets/objects/`).
2. **Install** the extension: `pip install -e <project>/source/<pkg>`.
3. **Launch**:
   ```bash
   ./isaaclab.sh -p <project>/scripts/teleop.py --task <task_id>
   ```
   (the script is auto-copied if you pass `--isaaclab-path` to `create_task.py`).

---

## 6. Common mistakes

- **`task_name` not `snake_case`** — the class-name conversion expects underscores.
- **Joint names that don't match the URDF** — Pydantic doesn't catch this; the env will fail at load. Use regex (e.g. `.*_shoulder_pitch_joint`) when joint names share patterns.
- **`frame_names` vs `target_links` for Pink IK** — `frame_names` uses the URDF-prefixed name (what Pink sees), `target_links` uses the USD link name (what Isaac Sim sees). Mixing them causes silent IK failures.
- **Forgetting `body_offset`** — without it the EEF is tracked at the wrist, not the fingertip, and grasping won't line up.
- **Wrong quaternion order** — the format is `[w, x, y, z]`, not `[x, y, z, w]`.

---

## 7. Full schema (from `scripts/create_task.py`)

The Pydantic models in `create_task.py` are the source of truth. If a field
isn't documented here, it's defined there. Validation runs before any files
are written, so a bad YAML fails fast with a readable error.
