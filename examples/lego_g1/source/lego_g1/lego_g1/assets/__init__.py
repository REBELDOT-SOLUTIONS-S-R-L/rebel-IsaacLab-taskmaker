# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset path constants for lego_g1.

Usage:
    from lego_g1.assets import SCENES_DIR, OBJECTS_DIR, ROBOTS_DIR

    SCENE_USD_PATH = os.path.join(SCENES_DIR, "scene.usd")
"""

import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(ASSETS_DIR, "scenes")
OBJECTS_DIR = os.path.join(ASSETS_DIR, "objects")
ROBOTS_DIR = os.path.join(ASSETS_DIR, "robots")
