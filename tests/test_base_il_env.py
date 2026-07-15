# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BaseILEnv tensor <-> action conversions (goal §6).

Uses fake ``isaaclab`` modules (see conftest) so no Isaac Sim is required. The
fake ``PoseUtils`` are pass-through bijections, so a "pose" here is simply
``{"pos": <tensor>, "rot": <tensor>}`` and round trips are exact.
"""

from __future__ import annotations

import types

import pytest
import torch


def _pose(pos, rot):
    return {"pos": torch.tensor(pos), "rot": torch.tensor(rot)}


# ---------------------------------------------------------------------------
# get_robot_eef_pose
# ---------------------------------------------------------------------------
def test_get_robot_eef_pose_reads_obs_keys(make_env):
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    env = make_env(
        eef_names=["left"],
        action_slices={"left": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"left": (7, 8)},
        obs_policy={"left_eef_pos": pos, "left_eef_quat": quat},
    )
    pose = env.get_robot_eef_pose("left")
    assert torch.equal(pose["pos"], pos)
    assert torch.equal(pose["rot"], quat)


def test_get_robot_eef_pose_honors_env_ids(make_env):
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    env = make_env(
        eef_names=["left"],
        action_slices={"left": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"left": (7, 8)},
        obs_policy={"left_eef_pos": pos, "left_eef_quat": quat},
    )
    pose = env.get_robot_eef_pose("left", env_ids=[1])
    assert torch.equal(pose["pos"], pos[[1]])


# ---------------------------------------------------------------------------
# target_eef_pose_to_action — slice placement
# ---------------------------------------------------------------------------
def test_target_pose_to_action_tuple_gripper(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": (7, 8)},
    )
    target = {"ee": _pose([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0, 7.0]])}
    gripper = {"ee": torch.tensor([[9.0]])}
    action = env.target_eef_pose_to_action(target, gripper)
    assert action.shape == (1, 8)
    assert torch.equal(action[..., 0:3], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(action[..., 3:7], torch.tensor([[4.0, 5.0, 6.0, 7.0]]))
    assert torch.equal(action[..., 7:8], torch.tensor([[9.0]]))


def test_target_pose_to_action_list_index_gripper(make_env):
    # Interleaved gripper columns (e.g. Inspire hand): indices 7 and 9.
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": [7, 9]},
    )
    target = {"ee": _pose([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0, 7.0]])}
    gripper = {"ee": torch.tensor([[8.0, 9.0]])}
    action = env.target_eef_pose_to_action(target, gripper)
    assert action.shape == (1, 10)  # sized from max index (9) + 1
    assert action[0, 7].item() == 8.0
    assert action[0, 9].item() == 9.0
    assert action[0, 8].item() == 0.0  # untouched column stays zero


def test_target_pose_to_action_dual_eef(make_env):
    env = make_env(
        eef_names=["left", "right"],
        action_slices={
            "left": {"pos": (0, 3), "quat": (3, 7)},
            "right": {"pos": (7, 10), "quat": (10, 14)},
        },
        gripper_slices={"left": (14, 15), "right": (15, 16)},
    )
    target = {
        "left": _pose([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]),
        "right": _pose([[2.0, 2.0, 2.0]], [[0.0, 0.0, 1.0, 0.0]]),
    }
    gripper = {"left": torch.tensor([[0.1]]), "right": torch.tensor([[0.2]])}
    action = env.target_eef_pose_to_action(target, gripper)
    assert action.shape == (1, 16)
    assert torch.equal(action[..., 7:10], torch.tensor([[2.0, 2.0, 2.0]]))
    assert action[0, 15].item() == pytest.approx(0.2)


def test_target_pose_to_action_applies_noise(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": (7, 8)},
    )
    target = {"ee": _pose([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0, 7.0]])}
    gripper = {"ee": torch.tensor([[9.0]])}

    torch.manual_seed(0)
    clean = env.target_eef_pose_to_action(target, gripper)
    torch.manual_seed(0)
    noisy = env.target_eef_pose_to_action(target, gripper, action_noise_dict={"ee": 0.5})

    # Noise perturbs the pos/quat slices but leaves the gripper column untouched.
    assert not torch.equal(noisy[..., 0:7], clean[..., 0:7])
    assert torch.equal(noisy[..., 7:8], clean[..., 7:8])


def test_empty_eef_names_raises(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    with pytest.raises(ValueError) as exc:
        env.target_eef_pose_to_action({}, {})
    assert "eef_names is empty" in str(exc.value)


# ---------------------------------------------------------------------------
# action_to_target_eef_pose — inverse
# ---------------------------------------------------------------------------
def test_action_to_target_pose_round_trip(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": (7, 8)},
    )
    target = {"ee": _pose([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0, 7.0]])}
    gripper = {"ee": torch.tensor([[9.0]])}
    action = env.target_eef_pose_to_action(target, gripper)
    recovered = env.action_to_target_eef_pose(action)
    assert torch.equal(recovered["ee"]["pos"], target["ee"]["pos"])
    assert torch.equal(recovered["ee"]["rot"], target["ee"]["rot"])


# ---------------------------------------------------------------------------
# actions_to_gripper_actions — 2D and 3D
# ---------------------------------------------------------------------------
def test_actions_to_gripper_actions_2d(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": (7, 8)},
    )
    actions = torch.arange(8.0).reshape(1, 8)  # (N, D)
    grippers = env.actions_to_gripper_actions(actions)
    assert grippers["ee"].shape == (1, 1)
    assert grippers["ee"][0, 0].item() == 7.0


def test_actions_to_gripper_actions_3d(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": (7, 8)},
    )
    actions = torch.arange(2 * 3 * 8.0).reshape(2, 3, 8)  # (N, T, D)
    grippers = env.actions_to_gripper_actions(actions)
    assert grippers["ee"].shape == (2, 3, 1)


def test_actions_to_gripper_actions_list_index(make_env):
    env = make_env(
        eef_names=["ee"],
        action_slices={"ee": {"pos": (0, 3), "quat": (3, 7)}},
        gripper_slices={"ee": [7, 9]},
    )
    actions = torch.arange(10.0).reshape(1, 10)
    grippers = env.actions_to_gripper_actions(actions)
    assert grippers["ee"].shape == (1, 2)
    assert grippers["ee"][0].tolist() == [7.0, 9.0]


# ---------------------------------------------------------------------------
# get_object_poses
# ---------------------------------------------------------------------------
def test_get_object_poses_reads_scene_state(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    root_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0]]
    )
    env.scene = types.SimpleNamespace(
        get_state=lambda is_relative: {"rigid_object": {"cube": {"root_pose": root_pose}}}
    )
    poses = env.get_object_poses()
    assert set(poses) == {"cube"}
    assert torch.equal(poses["cube"]["pos"], root_pose[:, :3])
    assert torch.equal(poses["cube"]["rot"], root_pose[:, 3:7])


def test_get_object_poses_honors_env_ids(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    root_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0]]
    )
    env.scene = types.SimpleNamespace(
        get_state=lambda is_relative: {"rigid_object": {"cube": {"root_pose": root_pose}}}
    )
    poses = env.get_object_poses(env_ids=[1])
    assert torch.equal(poses["cube"]["pos"], root_pose[[1], :3])


# ---------------------------------------------------------------------------
# subtask signal methods
# ---------------------------------------------------------------------------
def test_get_subtask_start_signals_default_empty(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    assert env.get_subtask_start_signals() == {}


def test_get_subtask_term_predicates_reads_obs_buf(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    grasp = torch.tensor([[1.0], [0.0]])
    env.obs_buf["subtask_terms"] = {"grasp": grasp}
    predicates = env.get_subtask_term_predicates()
    assert torch.equal(predicates["grasp"], grasp)


def test_get_subtask_term_predicates_honors_env_ids(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    grasp = torch.tensor([[1.0], [0.0]])
    env.obs_buf["subtask_terms"] = {"grasp": grasp}
    predicates = env.get_subtask_term_predicates(env_ids=[0])
    assert torch.equal(predicates["grasp"], grasp[[0]])


def test_get_subtask_term_predicates_missing_key_returns_empty(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    assert env.get_subtask_term_predicates() == {}


def test_get_subtask_term_signals_delegates_to_predicates(make_env):
    env = make_env(eef_names=[], action_slices={}, gripper_slices={})
    grasp = torch.tensor([[1.0], [0.0]])
    env.obs_buf["subtask_terms"] = {"grasp": grasp}
    assert torch.equal(env.get_subtask_term_signals()["grasp"], grasp)
