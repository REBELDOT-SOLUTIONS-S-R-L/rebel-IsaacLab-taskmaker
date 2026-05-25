# IL Task: LegoG1

IsaacLab IL extension for the **IL-LEGO-G1-v0** task: a Unitree G1 humanoid
with Inspire FTP hands picks up two Lego bricks and places them on per-arm
targets.

Generated from
`source/IsaacLabTaskMaker/.../templates/task_definitions/g1_lego.yaml` via
`scripts/create_task.py`, then hand-tuned (subtask thresholds, gripper joint
pattern, etc.).

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

## YAML schema (`g1_lego.yaml`)

This task is defined by a single YAML file consumed by `scripts/create_task.py`.
The sections below mirror the file top-to-bottom — required ones are flagged
explicitly; the rest are optional with sensible defaults.

### Task identity (required)

```yaml
task_name: lego_g1                # snake_case folder + Python class prefix
task_id: IL-LEGO-G1-v0            # Gymnasium env ID (used by --task <id>)
```

- **`task_name`** — folder name under `tasks/manager_based/` and the prefix of
  the generated cfg class (`lego_g1` → `LegoG1TaskCfg`).
- **`task_id`** — gym registration ID; when `mimic:` is present, a sibling
  `<task_id>-Mimic` env is registered too.

### `robot` (required)

```yaml
robot:
  name: unitree_g1                                # scene attribute name (default: "robot")
  import_path: isaaclab_assets.robots.unitree     # Python module with the ArticulationCfg
  config_name: G1_INSPIRE_FTP_CFG                 # ArticulationCfg variable in that module
  prim_path: "/World/envs/env_.*/Robot"           # USD path; rarely changed
  init_pos: [-1.05, 0, 0.8]                       # meters, [x, y, z]
  init_rot: [1.0, 0.0, 0.0, 0.0]                  # quaternion, [w, x, y, z]
```

- **`name`** — becomes `cfg.scene.<name>` and the key for `env.scene["<name>"]`
  / `SceneEntityCfg("<name>")` everywhere in the generated code. Defaults to
  `"robot"`. Set to something descriptive (e.g. `unitree_g1`, `franka`) so the
  asset is self-identifying.
- **`import_path`** — module to `import` for the articulation config.
- **`config_name`** — the `ArticulationCfg` variable inside that module that
  defines the robot.
- **`prim_path`** — USD path pattern where the robot is spawned (`env_.*` is
  replaced per parallel env). Default `/World/envs/env_.*/Robot`.
- **`init_pos`** — initial world position in meters.
- **`init_rot`** — initial world orientation as a `[w, x, y, z]` quaternion.

Only `import_path` and `config_name` are mandatory; everything else has a
default.

### `scene` (required)

```yaml
scene:
  usd_file: lego.usd                # filename in assets/scenes/
  scale: [1.0, 1.0, 1.0]
  init_pos: [0.0, 0.0, 0.0]
  init_rot: [1.0, 0.0, 0.0, 0.0]
```

- **`usd_file`** — scene USD filename; resolved against `assets/scenes/`.
- **`scale`**, **`init_pos`**, **`init_rot`** — uniform scale and world-frame
  transform applied to the scene root.

### `scene_objects` (optional)

Rigid bodies the robot interacts with. Each entry becomes a Python attribute
on the generated `SceneCfg`.

```yaml
scene_objects:
  - name: blue_brick                              # Python attribute / scene key
    type: RigidObjectCfg                          # or "AssetBaseCfg" for static deco
    prim_path: "{ENV_REGEX_NS}/blue_brick"        # per-env USD path
    usd_file: lego/brick_2x2.usd                  # filename in assets/objects/
    scale: [0.0005, 0.0005, 0.0005]
    init_pos: [-0.5, 0.15, 0.83]
    init_rot: [1.0, 0.0, 0.0, 0.0]
    use_default_sdf_collision: true               # SDF-approximate the mesh
    physics_material:                             # optional PhysX override
      static_friction: 1.2
      dynamic_friction: 1.0
      restitution: 0.0
```

- **`name`** — unique identifier; the generator emits `scene_cfg.<name>` and
  this is what `env.scene["<name>"]` will return.
- **`type`** — `RigidObjectCfg` for physics-tracked / resettable bodies,
  `AssetBaseCfg` for static decoration.
- **`prim_path`** — USD path for the spawned object. `{ENV_REGEX_NS}` is
  substituted with the per-env prefix.
- **`usd_file`** — file under `assets/objects/`; subdirectories are allowed
  (e.g. `lego/brick_2x2.usd`).
- **`scale` / `init_pos` / `init_rot`** — per-object transform.
- **`use_default_sdf_collision`** — when `true` (default), the spawner replaces
  baked-in collision approximations with an SDF mesh. Set `false` to keep
  whatever collision shape is in the USD.
- **`physics_material`** — optional per-object friction / restitution override.
  Omit the block to inherit the simulation defaults.

### `ik_controller` (required)

```yaml
ik_controller:
  controller_type: pink_ik                        # task-space IK via the Pink library
  base_link_name: pelvis                          # root of the kinematic chain
  controlled_joint_names:                         # joints the IK solver may move
    - ".*_shoulder_pitch_joint"
    - ".*_elbow_joint"
    # ...
  hand_joint_names:                               # finger joints (mapped from XR)
    - "L_index_proximal_joint"
    # ...
  num_hand_joints: 24                             # total count across both hands
  null_space_joints:                              # optional posture-control joints
    - "waist_yaw_joint"
    - "waist_pitch_joint"
    - "waist_roll_joint"
```

- **`controller_type`** — picks the action / controller layout. Currently:
  `pink_ik`, `differential_ik`, `operational_space`, `rmp_flow`, and four
  joint-space variants.
- **`base_link_name`** *(pink_ik)* — root link of the kinematic chain.
- **`controlled_joint_names`** *(pink_ik)* — joints the IK solver can move
  (arms / wrists). Regexes are allowed; `.*` matches both arms in one entry.
- **`hand_joint_names`** *(pink_ik)* — finger joints driven by XR hand tracking.
  Order matters: it defines the column layout of the hand portion of the
  action tensor.
- **`num_hand_joints`** *(pink_ik)* — total finger DOFs across all hands.
- **`null_space_joints`** *(optional)* — extra joints (e.g. waist) for posture
  control. Lets the torso stay stable while the arms move.

### `eef` (required for multi-EEF tasks)

```yaml
eef:
  names: ["left", "right"]                        # logical EEF identifiers
  target_links:                                   # logical → robot link name
    left_wrist: left_wrist_yaw_link
    right_wrist: right_wrist_yaw_link
  frame_names:                                    # logical → URDF frame name
    left: g1_29dof_rev_1_0_left_wrist_yaw_link
    right: g1_29dof_rev_1_0_right_wrist_yaw_link
  hand_joint_prefixes:                            # per-arm `hand_joint_names` prefix
    left: "L_"
    right: "R_"
```

- **`names`** — logical EEF names referenced from observations, subtasks, and
  Mimic configs.
- **`target_links`** — maps each EEF to the **USD body name** Pink IK should
  drive.
- **`frame_names`** — maps each EEF to the **URDF link name** Pink uses
  internally (URDF often prefixes the link name during conversion, so this is
  usually different from `target_links`).
- **`hand_joint_prefixes`** — per-arm prefix of `hand_joint_names` entries that
  belong to that EEF. Required when the same controller drives multiple arms
  with interleaved finger joints (Inspire-style hands prefix every joint with
  `L_` / `R_`). The generator uses these to slice gripper indices for
  `BaseILEnv`.

### `observations` (required)

```yaml
observations:
  eef_link_names:
    left: left_wrist_yaw_link
    right: right_wrist_yaw_link
```

- **`eef_link_names`** — robot links whose pose is included in the observation
  tensor (one ObsTerm per EEF, named `<eef>_eef_pos` / `<eef>_eef_quat`).
  Must match the values in `eef.target_links`.

### `cameras` (optional)

Each entry becomes a `CameraCfg` on the scene cfg and is re-attached after
IsaacLab's XR pipeline strips cameras during `remove_camera_configs`.

```yaml
cameras:
  - name: head_camera                             # scene_cfg.<name>
    prim_path: "{ENV_REGEX_NS}/Robot/torso_link/d435_link/head_camera"
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
  device: handtracking                            # keyboard | spacemouse | gamepad | handtracking | manusvive
  retargeter_import: isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter
  retargeter_class: UnitreeG1RetargeterCfg
  xr_anchor_pos: [-1.05, 0.0, -0.3]               # XR tracking-space origin in world
  xr_anchor_rot: [1.0, 0.0, 0.0, 0.0]
```

- **`device`** — default teleop device. Can be overridden at the script CLI.
- **`retargeter_import` / `retargeter_class`** *(required for hand tracking)* —
  retargeter that maps XR hand poses onto the robot's joints / EEFs.
- **`xr_anchor_pos` / `xr_anchor_rot`** *(handtracking only)* — anchor that
  places the operator's tracking-space origin in world coordinates so their
  hands land near the robot's wrists. The default identity only works for
  robots at the world origin; offset it for any displaced base (G1 sits at
  `-1.05, 0, 0.8`).

### `sim` (optional, has defaults)

```yaml
sim:
  decimation: 6                                   # sim steps per policy step
  episode_length_s: 20.0                          # max episode duration, seconds
  dt: 0.008333                                    # physics timestep (1/120)
  render_interval: 2                              # render every N sim steps
  env_spacing: 2.5                                # meters between parallel envs
```

All fields are optional. Defaults are sensible for IL workflows; tighten `dt`
or lower `decimation` if the policy needs higher control bandwidth.

### `subtask_terms` (optional)

Declarative binding from a `subtask_term_signal` name to the MDP predicate
function plus parameters that compute it. The generator stubs each function
in `mdp/observations.py` with an inferred type signature; you fill in the
body.

```yaml
subtask_terms:
  grasp_brick_left:
    func: grasp_brick_done                        # MDP function name; defaults to signal name
    params:                                       # kwargs forwarded to the function
      eef_link: left_wrist_yaw_link
      object_name: red_brick
      dist_threshold: 0.25
      gripper_joint_pattern: "L_.*_proximal_joint"
      gripper_closed_threshold: 1.3
```

- **`func`** — name of the MDP function in the generated `mdp/observations.py`.
  Multiple signals can share one function (DRY); the signal name is added as
  a kwarg automatically. Omit `func` to get one stub per signal.
- **`params`** — kwargs passed to that function. Parameter types are inferred
  from the YAML value (`str`, `int`, `float`, `tuple[float, ...]`) and used
  to type the generated stub's signature.

### `mimic` (optional — Mimic data-generation pipeline)

When present, a sibling `<task_id>-Mimic` env is registered with a cfg class
that mixes `MimicEnvCfg` into the base task cfg.

```yaml
mimic:
  datagen:                                        # passthrough to DataGenConfig
    name: demo_src_lego_g1_D0
    generation_guarantee: true
    generation_num_trials: 1000
    seed: 1
    # ... any DataGenConfig field is accepted
  subtasks:                                       # eef_name → ordered SubTaskConfig list
    right:
      - object_ref: blue_brick                    # scene object the subtask is "about"
        subtask_term_signal: grasp_brick_right    # must match a key in subtask_terms
        first_subtask_start_offset_range: [0, 0]
        subtask_term_offset_range: [0, 0]
        selection_strategy: nearest_neighbor_object
        selection_strategy_kwargs:
          nn_k: 3
        action_noise: 0.003
        num_interpolation_steps: 0
        num_fixed_steps: 0
        apply_noise_during_interpolation: false
      # ...more subtasks; the last one uses subtask_term_signal: null to run
      # "until end of demo".
    left:
      # ...mirror structure
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
└── tasks/           # Task-specific configurations and MDP functions
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
