# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small pure helpers: teleop sensitivity, literal rendering, jinja filters."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _teleop_sensitivity
# ---------------------------------------------------------------------------
def test_teleop_sensitivity_override_wins(create_task):
    # An explicit YAML value is returned verbatim, ignoring device defaults.
    assert create_task._teleop_sensitivity(0.42, "keyboard", axis="pos") == 0.42
    assert create_task._teleop_sensitivity(0.42, "gamepad", axis="rot") == 0.42


@pytest.mark.parametrize(
    ("device", "axis", "expected"),
    [
        ("keyboard", "pos", 0.05),
        ("keyboard", "rot", 0.05),
        ("spacemouse", "pos", 0.05),
        ("gamepad", "pos", 10.0),
        ("gamepad", "rot", 10.0),
    ],
)
def test_teleop_sensitivity_device_defaults(create_task, device, axis, expected):
    assert create_task._teleop_sensitivity(None, device, axis=axis) == expected


def test_teleop_sensitivity_unknown_device_falls_back_to_keyboard(create_task):
    # Non-Se3 devices (handtracking, ...) have no defaults; keyboard is used so
    # the template always substitutes a valid float.
    assert create_task._teleop_sensitivity(None, "handtracking", axis="pos") == 0.05
    assert create_task._teleop_sensitivity(None, "handtracking", axis="rot") == 0.05


# ---------------------------------------------------------------------------
# _py_value
# ---------------------------------------------------------------------------
def test_py_value_numeric_list_becomes_tuple_literal(create_task):
    assert create_task._py_value([0.1, 0.2]) == repr((0.1, 0.2))
    assert create_task._py_value([1, 2, 3]) == repr((1, 2, 3))


def test_py_value_empty_list_stays_list(create_task):
    # Empty lists don't trip the numeric-upgrade branch.
    assert create_task._py_value([]) == repr([])


def test_py_value_bool_list_not_upgraded(create_task):
    # A list of bools must not be rendered as a numeric tuple.
    assert create_task._py_value([True, False]) == repr([True, False])


def test_py_value_mixed_list_stays_list(create_task):
    assert create_task._py_value([1, "x"]) == repr([1, "x"])


def test_py_value_scalar_and_dict_passthrough(create_task):
    assert create_task._py_value("hello") == repr("hello")
    assert create_task._py_value(None) == repr(None)
    assert create_task._py_value({"x": (0.0, 1.0)}) == repr({"x": (0.0, 1.0)})


# ---------------------------------------------------------------------------
# _pyrepr / _pylist / _pydict jinja filters
# ---------------------------------------------------------------------------
def test_pyrepr_filter(create_task):
    assert create_task._pyrepr([1, 2]) == "[1, 2]"
    assert create_task._pyrepr((1, 2)) == "(1, 2)"
    assert create_task._pyrepr(None) == "None"
    assert create_task._pyrepr(True) == "True"


def test_pyrepr_registered_as_jinja_filter(create_task):
    # Confirms the filter is wired into the template environment.
    env = create_task._create_jinja_env()
    assert "pyrepr" in env.filters
    rendered = env.from_string("{{ value | pyrepr }}").render(value=(1.0, 2.0))
    assert rendered == "(1.0, 2.0)"


def test_pylist_and_pydict_filters(create_task):
    assert create_task._pylist(["a", "b"]) == '["a", "b"]'
    assert create_task._pydict({"left": "L_", "right": "R_"}) == '{"left": "L_", "right": "R_"}'
