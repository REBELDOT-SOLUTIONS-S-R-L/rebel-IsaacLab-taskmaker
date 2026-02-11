# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register the TM7 G1 IL task.
##
gym.register(
    id="IL-TM7-G1-v0",
    entry_point="IsaacLabILEnvs.tasks.manager_based.base_il_env.base_il_env:BaseILEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tm7_g1_cfg:TM7G1TaskCfg",
    },
)
