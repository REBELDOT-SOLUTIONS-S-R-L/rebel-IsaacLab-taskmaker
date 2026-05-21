# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="IL-LEGO-G1-v0",
    entry_point="lego_g1.base_il_env.base_il_env:BaseILEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lego_g1_cfg:LegoG1TaskCfg",
    },
)

gym.register(
    id="IL-LEGO-G1-v0-Mimic",
    entry_point="lego_g1.base_il_env.base_il_env:BaseILEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lego_g1_mimic_cfg:LegoG1MimicEnvCfg",
    },
)
