# IsaacLab Task Maker

Config-only imitation-learning task creation for [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

Define a task in **one YAML file**, run `create_task.py`, get back a complete
standalone Isaac Lab extension with the scene, controller, observations,
teleop bindings, cameras, and Mimic data-generation wiring already in place.
You spend your time filling in subtask logic and tuning thresholds — not
writing boilerplate.

```bash
python scripts/create_task.py templates/task_definitions/g1_lego.yaml
# → ./examples/lego_g1/  (a full, installable IL extension)
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
- **Declarative `DoneTerm`s.** Optional `terminations:` block wires extra
  `DoneTerm` entries (e.g. a Mimic-style `success` criterion) onto
  `TerminationsCfg` and stubs the matching predicates in `mdp/terminations.py`
  with inferred type signatures — same DRY collapsing as `subtask_terms`.
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
pip install -e examples/lego_g1/source/lego_g1

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

### Task identity (required)

```yaml
task_name: my_task                                # snake_case folder + Python class prefix
task_id: IL-MyTask-v0                             # Gymnasium env ID (used by --task <id>)
```

- **`task_name`** — folder name under `tasks/manager_based/` and the prefix of
  the generated cfg class (`my_task` → `MyTaskTaskCfg`).
- **`task_id`** — gym registration ID; when `mimic:` is present, a sibling
  `<task_id>-Mimic` env is registered too.

### `robot` (required)

```yaml
robot:
  name: my_robot                                  # scene attribute name (default: "robot")
  import_path: isaaclab_assets.robots.franka      # Python module with the ArticulationCfg
  config_name: FRANKA_PANDA_HIGH_PD_CFG           # ArticulationCfg variable in that module
  prim_path: "/World/envs/env_.*/Robot"           # USD path; rarely changed
  init_pos: [0.0, 0.5, 0.4]                       # meters, [x, y, z]
  init_rot: [0.7071, 0.0, 0.0, -0.7071]           # quaternion, [w, x, y, z]
  scale: [1.0, 1.0, 1.0]                          # optional, default identity
```

- **`name`** — becomes `cfg.scene.<name>` and the key for `env.scene["<name>"]`
  / `SceneEntityCfg("<name>")` throughout the generated code. Defaults to
  `"robot"` for backward compat; pick something self-identifying (e.g.
  `unitree_g1`, `franka`, `so100`) so observations and subtask configs read
  well.
- **`import_path`** — module to `import` for the articulation config.
- **`config_name`** — the `ArticulationCfg` variable inside that module.
- **`prim_path`** — USD path pattern where the robot is spawned (`env_.*` is
  replaced per parallel env). Default `/World/envs/env_.*/Robot`.
- **`init_pos`** — initial world position in meters.
- **`init_rot`** — initial world orientation as a `[w, x, y, z]` quaternion.
- **`scale`** — uniform scale, default identity.

Only `import_path` and `config_name` are mandatory; everything else has a default.

### `scene` (optional)

The world USD loaded once per environment.

```yaml
scene:
  usd_file: scene.usd                             # filename in assets/scenes/
  scale: [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]
```

- **`usd_file`** — scene USD filename; resolved against `assets/scenes/`.
- **`scale`**, **`init_pos`**, **`init_rot`** — uniform scale and world-frame
  transform applied to the scene root.

If omitted, the generator assumes `scene.usd` at the origin.

### `scene_objects` (optional)

Rigid bodies the robot interacts with. Each entry becomes a Python attribute
on the generated `SceneCfg`.

```yaml
scene_objects:
  - name: object_1                                # Python attribute / scene key
    type: RigidObjectCfg                          # or "AssetBaseCfg" for static deco
    prim_path: "{ENV_REGEX_NS}/object_1"          # per-env USD path
    usd_file: parts/object.usd                    # filename in assets/objects/
    scale: [1.0, 1.0, 1.0]
    init_pos: [0.0, -0.15, 0.85]
    init_rot: [1.0, 0.0, 0.0, 0.0]
    use_default_sdf_collision: true               # SDF-approximate the mesh
    physics_material:                             # optional PhysX override
      static_friction: 1.0
      dynamic_friction: 1.0
      restitution: 0.0
  - name: object_2
    type: RigidObjectCfg
    prim_path: "{ENV_REGEX_NS}/object_2"
    usd_file: parts/object.usd
    scale: [1.0, 1.0, 1.0]
    init_pos: [0.0, 0.15, 0.85]
    init_rot: [1.0, 0.0, 0.0, 0.0]
```

- **`name`** — unique identifier; the generator emits `scene_cfg.<name>` and
  this is what `env.scene["<name>"]` returns.
- **`type`** — `RigidObjectCfg` for physics-tracked / resettable bodies,
  `AssetBaseCfg` for static decoration.
- **`prim_path`** — USD path for the spawned object. `{ENV_REGEX_NS}` is
  substituted with the per-env prefix.
- **`usd_file`** — file under `assets/objects/`; subdirectories are allowed.
  Defaults to `<name>.usd` if omitted.
- **`scale` / `init_pos` / `init_rot`** — per-object transform.
- **`use_default_sdf_collision`** — when `true` (default), the spawner replaces
  baked-in collision approximations with an SDF mesh. Set `false` to keep
  whatever collision shape is in the USD.
- **`physics_material`** — optional per-object friction / restitution override
  (`static_friction`, `dynamic_friction`, `restitution`). Omit the block to
  inherit the simulation defaults.

### `ik_controller` (required)

Common fields shared by every controller; per-controller extras are listed in §4.

```yaml
ik_controller:
  controller_type: differential_ik                # see §4 for the full list
  base_link_name: panda_link0                     # root of the kinematic chain
  controlled_joint_names:                         # joints the controller drives
    - "panda_joint.*"
  hand_joint_names:                               # finger joints (optional)
    - "panda_finger_joint.*"
  num_hand_joints: 2                              # len(hand_joint_names)
```

- **`controller_type`** — picks the action / controller layout (see §4 for
  the full menu).
- **`base_link_name`** — root link of the kinematic chain (required for
  task-space controllers like Pink IK; ignored by joint-space controllers).
- **`controlled_joint_names`** — joints the controller can move. Regexes are
  allowed; `.*` matches all joints matching the pattern in one entry.
- **`hand_joint_names`** *(optional)* — finger joints driven separately from
  the arm. Order matters: it defines the column layout of the hand portion
  of the action tensor.
- **`num_hand_joints`** *(optional)* — total finger DOFs. Set to
  `len(hand_joint_names)`.

### `eef` (required)

End-effectors the task tracks. Logical names appear in observations,
subtask configs, and Mimic configs.

Single-arm:

```yaml
eef:
  names: ["gripper"]                              # logical EEF identifiers
  target_links:
    gripper: panda_hand                           # robot link to track
  frame_names:                                    # for pink_ik, URDF-prefixed name
    gripper: panda_hand
```

Bimanual:

```yaml
eef:
  names: ["left", "right"]
  target_links:
    left:  left_wrist_yaw_link                    # USD body name
    right: right_wrist_yaw_link
  frame_names:                                    # URDF-prefixed names for Pink IK
    left:  prefix_left_wrist_yaw_link
    right: prefix_right_wrist_yaw_link
  hand_joint_prefixes:                            # per-arm `hand_joint_names` prefix
    left:  "L_"
    right: "R_"
```

- **`names`** — logical EEF identifiers used by observations, subtasks, and
  Mimic configs.
- **`target_links`** — maps each EEF to the **USD body name** the controller
  drives.
- **`frame_names`** — maps each EEF to the **URDF link name** Pink IK uses
  internally. URDF conversion often prefixes link names, so this is usually
  different from `target_links`.
- **`hand_joint_prefixes`** *(required for multi-EEF pink_ik with interleaved
  hand joints)* — per-arm prefix of `hand_joint_names` entries that belong
  to each EEF. Without it the generator falls back to an even split, which
  only works when joints are pre-grouped by arm in `hand_joint_names`.

### `observations` (optional)

Which robot links to expose in the observation dict. Usually mirrors
`eef.target_links`.

```yaml
observations:
  eef_link_names:
    gripper: panda_hand                           # robot link whose pose is observed
```

- **`eef_link_names`** — robot links whose pose is included in the
  observation tensor (one ObsTerm per EEF, named `<eef>_eef_pos` /
  `<eef>_eef_quat`). Must match the values in `eef.target_links`.

### `cameras` (optional)

Each entry becomes a `CameraCfg` on the scene cfg and is re-attached by
`attach_cameras(scene_cfg)` after IsaacLab's XR pipeline strips cameras via
`remove_camera_configs`. The `xr_camera_reattach` hook on `BaseILEnvCfg`
calls that helper automatically.

```yaml
cameras:
  - name: front_camera                            # scene_cfg.<name>
    prim_path: "{ENV_REGEX_NS}/Robot/torso/front_camera"
    height: 480
    width: 640
    data_types: ["rgb"]                           # add "distance_to_image_plane" for depth
    focal_length: 12.0                            # mm
    focus_distance: 400.0                         # cm
    horizontal_aperture: 20.955                   # mm sensor width
    clipping_range: [0.05, 10.0]                  # meters
    offset_pos: [0.0, 0.0, 0.0]                   # parent-relative, meters
    offset_rot: [0.5, 0.5, -0.5, -0.5]            # parent-relative quaternion
    convention: opengl                            # "opengl" | "ros" | "world"
```

- **`name`** — scene attribute the camera is exposed under.
- **`prim_path`** — USD prim the camera is mounted on (usually a robot link).
- **`height` / `width`** — image resolution in pixels.
- **`data_types`** — channels the camera renders; e.g. `["rgb"]`,
  `["distance_to_image_plane"]`, or both.
- **`focal_length` / `focus_distance` / `horizontal_aperture`** — pinhole
  optics parameters (focal length and aperture in mm; focus distance in cm).
- **`clipping_range`** — near / far clip planes in meters.
- **`offset_pos` / `offset_rot`** — local-frame offset from the parent prim.
- **`convention`** — axis convention for the camera frame.

### `teleop` (optional)

```yaml
teleop:
  device: keyboard                                # keyboard | spacemouse | gamepad | handtracking | manusvive
```

Default: `handtracking` if `controller_type: pink_ik`, otherwise `keyboard`.

For `handtracking` with `pink_ik` you must also set the retargeter:

```yaml
teleop:
  device: handtracking
  retargeter_import: isaaclab.devices.openxr.retargeters.<your_retargeter_module>
  retargeter_class: YourRetargeterCfg
  xr_anchor_pos: [-1.0, 0.0, -0.3]                # XR tracking-space origin in world
  xr_anchor_rot: [1.0, 0.0, 0.0, 0.0]
```

- **`device`** — default teleop device. Can be overridden at the script CLI.
- **`retargeter_import` / `retargeter_class`** *(required for hand tracking)* —
  retargeter that maps XR hand poses onto the robot's joints / EEFs.
- **`xr_anchor_pos` / `xr_anchor_rot`** *(handtracking only)* — anchor that
  places the operator's tracking-space origin in world coordinates so their
  hands land near the robot's wrists. Defaults are identity — required only
  for robots that aren't at `(0, 0, 0)`.

### `sim` (optional, has defaults)

```yaml
sim:
  decimation: 6                                   # sim steps per policy step
  episode_length_s: 20.0                          # max episode duration, seconds
  dt: 0.008333                                    # physics timestep (1/120)
  render_interval: 2                              # render every N sim steps
  env_spacing: 2.5                                # meters between parallel envs
```

- **`decimation`** — number of physics steps per policy step. Higher values
  smooth control at the cost of latency.
- **`episode_length_s`** — maximum episode duration in seconds.
- **`dt`** — physics timestep in seconds (`1/120 ≈ 0.008333`).
- **`render_interval`** — render every N sim steps. `1` matches `dt`; larger
  values speed up headless training.
- **`env_spacing`** — distance between parallel envs in meters.

All fields are optional. Defaults are sensible for IL workflows; tighten
`dt` or lower `decimation` if the policy needs higher control bandwidth.

### `subtask_terms` (optional)

Declarative binding from a `subtask_term_signal` name to the MDP predicate
function plus parameters that compute it. Multiple signals can share one
function (DRY) — the generator collapses N near-duplicate stubs into one
whose signature is the union of every param used.

```yaml
subtask_terms:
  pick_object_1_done:
    func: pick_done                               # MDP function (defaults to signal name)
    params:                                       # kwargs forwarded to the function
      eef_link: left_wrist_yaw_link
      object_name: object_1
      dist_threshold: 0.05
      gripper_joint_pattern: "L_.*_proximal_joint"
      gripper_closed_threshold: 1.0
  pick_object_2_done:
    func: pick_done                               # ← same function, different params
    params:
      eef_link: right_wrist_yaw_link
      object_name: object_2
      dist_threshold: 0.05
      gripper_joint_pattern: "R_.*_proximal_joint"
      gripper_closed_threshold: 1.0
```

- **`func`** — name of the MDP function in the generated `mdp/observations.py`.
  Multiple signals can share one function (DRY); the `signal_name` kwarg is
  added automatically. Omit `func` to get one stub per signal.
- **`params`** — kwargs passed to that function. Parameter types are inferred
  from the YAML value (`str`, `int`, `float`, `tuple[float, ...]`) and used
  to type the generated stub's signature.

If `subtask_terms` is omitted but `mimic.subtasks` declares signals, the
generator falls back to one zero-returning stub per signal.

### `terminations` (optional)

Extra `DoneTerm` entries on the generated `TerminationsCfg`. The YAML key
becomes the cfg attribute name; `func` resolves to a function emitted into
`mdp/terminations.py`. As with `subtask_terms`, multiple terms can share one
function — the generator emits a single stub whose signature carries the
union of every param used.

```yaml
terminations:
  success:
    func: bricks_released_at_targets                # MDP function name
    time_out: false                                 # optional; default false
    params:                                         # kwargs forwarded to the function
      left_object_name: red_brick
      right_object_name: blue_brick
      left_target_pos: [-0.4, 0.05, 0.83]
      right_target_pos: [-0.4, -0.05, 0.83]
      target_dist_threshold: 0.29
      left_gripper_joint_pattern: "L_(index_proximal_joint|middle_proximal_joint|thumb_proximal_pitch_joint)"
      right_gripper_joint_pattern: "R_(index_proximal_joint|middle_proximal_joint|thumb_proximal_pitch_joint)"
      gripper_open_threshold: 0.15
```

- **`func`** — name of the MDP function emitted into `mdp/terminations.py`.
  The generated stub returns `torch.zeros(env.num_envs, dtype=torch.bool, ...)` —
  fill in the body to compute the actual predicate.
- **`params`** — kwargs passed to the function. Parameter types are inferred
  from the YAML value (`str`, `int`, `float`, `tuple[float, ...]`) and used
  to type the generated stub's signature.
- **`time_out`** — defaults to `false`. Set `true` only for "ran out of time"
  terminations; the conventional success signal used by Mimic data
  generation keeps the default.
- **Reserved names** — the attribute name `time_out` is taken by the
  built-in `DoneTerm`. Pick anything else (`success`, `object_dropped`, …).

### `mimic` (optional)

When present, a sibling `<task_id>-Mimic` env is registered with a cfg class
that mixes `MimicEnvCfg` into the base task cfg.

```yaml
mimic:
  datagen:                                        # passthrough to DataGenConfig
    name: demo_src_my_task_D0
    generation_guarantee: true
    generation_num_trials: 1000
    seed: 1
    # ... any DataGenConfig field is accepted
  subtasks:                                       # eef_name → ordered SubTaskConfig list
    left:
      - object_ref: object_1                      # scene object the subtask is "about"
        subtask_term_signal: pick_object_1_done   # must match a key in subtask_terms
        subtask_term_offset_range: [0, 0]
        selection_strategy: nearest_neighbor_object
        selection_strategy_kwargs:
          nn_k: 3
        action_noise: 0.003
        num_interpolation_steps: 0
      # ... more subtasks; the final entry uses subtask_term_signal: null
      # to run "until end of demo".
      - object_ref: object_1
        subtask_term_signal: null
        subtask_term_offset_range: [0, 0]
        selection_strategy: nearest_neighbor_object
        selection_strategy_kwargs:
          nn_k: 3
        action_noise: 0.003
    right:
      # ...mirror structure for the right EEF, referencing object_2
```

- **`datagen`** — free-form passthrough; every key is forwarded as
  `self.datagen_config.<key> = <value>`. See `DataGenConfig` in
  `isaaclab/envs/mimic_env_cfg.py` for the full schema.
- **`subtasks`** — mapping of `eef_name` (must appear in `eef.names`) to an
  ordered list of `SubTaskConfig` entries. Each entry's keys are forwarded
  as kwargs to `SubTaskConfig(...)`; tuple-typed fields are written as YAML
  lists and coerced to tuples by `create_task.py`. Set
  `subtask_term_signal: null` on the final entry of each arm to denote "run
  until end of demo" per IsaacLab convention.

`mimic.subtasks[*].subtask_term_signal` names must match keys declared in
`subtask_terms:` (or, in the fallback layout, are themselves used as the
predicate function names).

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
