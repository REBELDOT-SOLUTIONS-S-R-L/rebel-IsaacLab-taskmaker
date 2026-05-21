# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for the lego_g1 task.

This module re-exports everything from the base IL MDP and adds
task-specific observations, events, and terminations.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Base IL observations (EEF helpers, joint state, etc.)
from lego_g1.base_il_env.mdp import *  # noqa: F401, F403

# Task-specific overrides
from .observations import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
