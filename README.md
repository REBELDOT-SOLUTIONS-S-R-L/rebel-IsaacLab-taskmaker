# IsaacLab Task Maker

Config-only imitation-learning task creation for [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

Define a task in **one YAML file**, run `create_task.py`, get back a complete
standalone Isaac Lab extension with the scene, controller, observations,
teleop bindings, cameras, and Mimic data-generation wiring already in place.
You spend your time filling in subtask logic and tuning thresholds — not
writing boilerplate.

```bash
python scripts/create_task.py templates/task_definitions/g1_lego.yaml
# → ./lego_g1/  (a full, installable IL extension)
```

---

## Features

- **One YAML → one installable extension.** Everything the generator needs is
  in a single, validated config; output is a self-contained project ready for
  `pip install -e`.
- **Eight controllers.** Pink IK (humanoid/dual-arm with finger control),
  differential IK, operational space, RMPFlow, and all four joint-space
  controllers — picked per task in the YAML.
- **`BaseILEnv` shared base class.** Implements the four mandatory
  `ManagerBasedRLMimicEnv` methods once; tasks never subclass it.
- **Auto-emitted action-tensor slicing.** `eef_action_slices` and
  `eef_gripper_slices` are derived from the controller type and EEF list
  (including prefix-based gripper indices for interleaved Inspire-style hands).
- **Cameras with XR support.** Declarative `cameras:` block plus a generated
  `attach_cameras(scene_cfg)` helper that gets re-applied after IsaacLab's XR
  pipeline strips cameras via `remove_camera_configs`.
- **Mimic data generation, end-to-end.** Optional `mimic.subtasks` block
  registers a sibling `<task_id>-Mimic` gym env and emits a
  `SubTaskConfig`-driven Mimic cfg. Pair it with `subtask_terms:` and the
  generator stubs out the MDP predicates with inferred type signatures.
- **No script forking.** Generated extensions install a `.pth` file that
  auto-registers gym envs on Python startup, so IsaacLab's stock scripts
  (`teleop_se3_agent.py`, `record_annotated_demos.py`, `random_agent.py`,
  `zero_agent.py`) find your task without any per-task patching.
- **XR teleop hooks baked into IsaacLab.** `--dataset_dir` / `--dataset_file`
  on the teleop script, an `xr_camera_reattach` cfg hook, and per-step
  `env._debug_subtask_heads` publishing for the annotated recorder — all in
  upstream-friendly hooks, no patching required.

---

## Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) installed (conda or uv).
- Python 3.10+.

## Installation

```bash
git clone https://github.com/REBELDOT-SOLUTIONS-S-R-L/IsaacLab-Task-Maker.git
cd IsaacLab-Task-Maker

conda activate <your-isaaclab-env>      # e.g. conda activate isaaclab
pip install -e source/IsaacLabTaskMaker
```

That's it — no script patching, no extra imports to add anywhere.

## Quick start

Generate the bundled `g1_lego` task as a working reference:

```bash
python scripts/create_task.py g1_lego.yaml --dry-run     # preview
python scripts/create_task.py g1_lego.yaml               # write files
```

The script prints the exact next steps. In short:

```bash
# 1. Drop your USD assets into the printed scene/objects/robots folders.
# 2. Install the generated extension (writes a .pth that auto-registers gyms).
pip install -e lego_g1/source/lego_g1

# 3. Run any stock IsaacLab script against the task.
cd <IsaacLab-dir>
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task IL-LEGO-G1-v0 --enable_pinocchio --teleop_device handtracking
./isaaclab.sh -p scripts/tools/record_annotated_demos.py --task IL-LEGO-G1-v0-Mimic
```

---

# YAML Config Guide

How to write a task-definition YAML. A YAML file is the **only input** the
generator needs.

The full set of working examples lives in
`source/IsaacLabTaskMaker/isaaclab_task_maker/templates/task_definitions/`.
Copy the one closest to your robot/controller and edit from there.

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
    use_default_sdf_collision: true                # use SDF instead of baked collision
    physics_material:                              # optional friction/restitution override
      static_friction:  1.2
      dynamic_friction: 1.0
      restitution:      0.0
```

- `usd_file` defaults to `<name>.usd` if omitted.
- `use_default_sdf_collision: true` (default) overrides the mesh collision
  approximation with SDF — good for arbitrary visual meshes used as dynamic
  bodies. Set `false` to keep the collision approximation baked into the USD.
- `physics_material` overrides PhysX friction/restitution for this object.
  Omit to inherit the simulation's default material.

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
  hand_joint_prefixes:                        # required for multi-EEF pink_ik
    left:  "L_"                               # with interleaved hand joints
    right: "R_"
```

`hand_joint_prefixes` tells the generator which entries of
`ik_controller.hand_joint_names` belong to each arm. Without it the generator
falls back to an even split, which only works when joints are pre-grouped by
arm in `hand_joint_names`.

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
  xr_anchor_pos: [-1.05, 0.0, -0.3]          # OpenXR tracking-space origin
  xr_anchor_rot: [1.0, 0.0, 0.0, 0.0]
```

`xr_anchor_pos` / `xr_anchor_rot` position the operator's tracking-space
origin in world coordinates so their hands land near the robot's wrists.
Defaults are identity — required only for robots that aren't at `(0, 0, 0)`.

### `sim` (optional)

```yaml
sim:
  decimation: 6                # sim steps per policy step
  episode_length_s: 20.0
  dt: 0.008333                 # 1/120 — physics timestep
  render_interval: 2
  env_spacing: 2.5             # meters between parallel envs
```

These propagate into the generated `__post_init__` and override the
`BaseILEnvCfg` defaults.

### `cameras` (optional)

Each entry becomes a `CameraCfg` attribute on the scene cfg, plus a call
inside a module-level `attach_cameras(scene_cfg)` helper. IsaacLab's teleop
script (via the `xr_camera_reattach` hook on `BaseILEnvCfg`) re-applies
`attach_cameras` after the XR pipeline strips cameras via
`remove_camera_configs`.

```yaml
cameras:
  - name: head_camera
    prim_path: "{ENV_REGEX_NS}/Robot/torso_link/d435_link/head_camera"
    height: 480
    width: 640
    data_types: ["rgb"]                       # rgb / distance_to_image_plane / ...
    focal_length: 12.0
    focus_distance: 400.0
    horizontal_aperture: 20.955
    clipping_range: [0.05, 10.0]
    offset_pos: [0.0, 0.0, 0.0]               # local-frame offset from parent prim
    offset_rot: [0.5, 0.5, -0.5, -0.5]
    convention: opengl                        # opengl / ros / world
```

### `subtask_terms` (optional)

Declarative binding from a `subtask_term_signal` name to the MDP function +
params that compute it. Multiple signals can share one function (e.g. a
single `grasp_brick_done` backs both `grasp_brick_left` and
`grasp_brick_right`), so the generated `mdp/observations.py` collapses N
near-duplicate stubs into one stub whose signature is the union of every
param used.

```yaml
subtask_terms:
  grasp_brick_left:
    func: grasp_brick_done                    # defaults to signal name if omitted
    params:
      eef_link: left_wrist_yaw_link
      object_name: brick_2x2_1
      dist_threshold: 0.25
      gripper_joint_pattern: "L_.*_proximal_joint"
      gripper_closed_threshold: 1.3
  grasp_brick_right:
    func: grasp_brick_done                    # ← same function, different params
    params:
      eef_link: right_wrist_yaw_link
      object_name: brick_2x2
      dist_threshold: 0.25
      gripper_joint_pattern: "R_.*_proximal_joint"
      gripper_closed_threshold: 1.3
```

Param types are inferred from the YAML value (`str` / `int` / `float` /
`tuple[float, ...]`) and used to type the generated stub's signature. The
`signal_name` kwarg is added automatically.

If `subtask_terms` is omitted but `mimic.subtasks` declares signals, the
generator falls back to one zero-returning stub per signal.

### `mimic` (optional)

When present, an additional `<task_id>-Mimic` gym env is registered alongside
the base task. Its cfg mixes the task cfg with
`isaaclab.envs.mimic_env_cfg.MimicEnvCfg`, exposing `datagen_config` and
`subtask_configs` for the IsaacLab Mimic pipeline.

```yaml
mimic:
  datagen:                                    # passed through to self.datagen_config.<k>
    name: demo_src_my_task_D0
    generation_guarantee: true
    generation_num_trials: 1000
    seed: 1
  subtasks:                                   # eef_name → ordered SubTaskConfig list
    left:
      - object_ref: brick_2x2_1
        subtask_term_signal: grasp_brick_left
        subtask_term_offset_range: [0, 0]
        selection_strategy: nearest_neighbor_object
        selection_strategy_kwargs: { nn_k: 3 }
        action_noise: 0.003
        num_interpolation_steps: 0
      # ... more subtasks
      - object_ref: brick_2x2_1
        subtask_term_signal: null             # final subtask: run until end-of-demo
        subtask_term_offset_range: [0, 0]
        selection_strategy: nearest_neighbor_object
        selection_strategy_kwargs: { nn_k: 3 }
        action_noise: 0.003
```

`mimic.subtasks[*].subtask_term_signal` names must match keys declared in
`subtask_terms:` (or, in the fallback layout, are themselves used as the
predicate function names). The final subtask of each arm uses
`subtask_term_signal: null` per IsaacLab convention.

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

`eef.frame_names` must use the URDF-prefixed link names. Multi-EEF pink_ik
tasks with `hand_joint_names` should set `eef.hand_joint_prefixes`.

### `joint_position`, `joint_velocity`, `joint_effort`, `relative_joint_position`

Only the common fields apply — `controlled_joint_names` is what matters.

```yaml
ik_controller:
  controller_type: joint_position
  controlled_joint_names: ["panda_joint1", ..., "panda_joint7"]
```

---

## After generating

1. **Drop your USD files** at the printed paths
   (`<project>/source/<pkg>/<pkg>/assets/scenes/` and `.../assets/objects/`).
2. **Install** the extension: `pip install -e <project>/source/<pkg>`. The
   install writes a `.pth` file that auto-imports the package on Python
   startup, so the gym envs register automatically.
3. **Run** any IsaacLab script directly:
   ```bash
   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task IL-MyTask-v0 --teleop_device handtracking \
       --dataset_dir ./datasets/my_task --dataset_file run0

   # Mimic recording (requires `mimic:` in the YAML)
   ./isaaclab.sh -p scripts/tools/record_annotated_demos.py --task IL-MyTask-v0-Mimic
   ```

## Common mistakes

- **`task_name` not `snake_case`** — the class-name conversion expects underscores.
- **Joint names that don't match the URDF** — Pydantic doesn't catch this; the env will fail at load. Use regex (e.g. `.*_shoulder_pitch_joint`) when joint names share patterns.
- **`frame_names` vs `target_links` for Pink IK** — `frame_names` uses the URDF-prefixed name (what Pink sees), `target_links` uses the USD link name (what Isaac Sim sees). Mixing them causes silent IK failures.
- **Forgetting `body_offset`** — without it the EEF is tracked at the wrist, not the fingertip, and grasping won't line up.
- **Wrong quaternion order** — the format is `[w, x, y, z]`, not `[x, y, z, w]`.
- **Multi-EEF pink_ik without `hand_joint_prefixes`** — the generator can't tell which finger joints belong to which arm and falls back to a contiguous even split. Set the prefixes explicitly for any interleaved hand layout (e.g. Inspire).
- **Subtask predicate names that don't match** — `mimic.subtasks[*].subtask_term_signal` must equal a key in `subtask_terms` (or, in the fallback layout, an emitted stub function name). Mismatches cause `KeyError` at recording time.

## Full schema

The Pydantic models in `scripts/create_task.py` are the source of truth. If a
field isn't documented here, it's defined there. Validation runs before any
files are written, so a bad YAML fails fast with a readable error.

---

## Generated project structure

```
<task_name>/                                   ← standalone, installable project
├── README.md
├── pyproject.toml
└── source/<package_name>/
    ├── config/extension.toml
    ├── setup.py                              ← writes the .pth on install
    ├── <package_name>_register.pth
    └── <package_name>/
        ├── __init__.py
        ├── assets/                           ← USD scene, object, robot files
        │   ├── scenes/
        │   ├── objects/
        │   └── robots/
        ├── base_il_env/                      ← copy of the shared base
        │   ├── base_il_env.py
        │   ├── base_il_env_cfg.py
        │   └── mdp/
        └── tasks/
            └── manager_based/
                └── <task_name>/
                    ├── __init__.py           ← gym.register() (+ Mimic variant)
                    ├── <task_name>_cfg.py    ← scene, actions, observations
                    ├── <task_name>_mimic_cfg.py   ← (when `mimic:` is set)
                    └── mdp/
                        ├── observations.py   ← subtask predicate stubs
                        ├── events.py
                        └── terminations.py
```

The taskmaker repo itself contains the **templates and generator** — generated
projects live wherever you point `--output-dir` at (defaults to the current
directory).

## License

BSD-3-Clause
