# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific event functions for tm7_gr1t.

Add custom event/reset functions here. They will be available as `mdp.<func_name>`
in your task config's EventCfg.

Base events (reset_scene_to_default, etc.) are already available from isaaclab.envs.mdp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Example: uncomment and customize
# def my_custom_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
#     """Custom reset logic for specific objects."""
#     pass
