# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="IL-TM7-GR1T-v0",
    entry_point="isaaclab_task_maker.tasks.manager_based.base_il_env.base_il_env:BaseILEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tm7_gr1t_cfg:Tm7Gr1tTaskCfg",
    },
)
