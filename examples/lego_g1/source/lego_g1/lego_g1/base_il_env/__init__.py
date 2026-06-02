# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base IL environment — shared base classes and MDP helpers.

This package provides BaseILEnv and BaseILEnvCfg that all IL tasks inherit from.
Do NOT register gym environments here — each task folder has its own __init__.py.
"""