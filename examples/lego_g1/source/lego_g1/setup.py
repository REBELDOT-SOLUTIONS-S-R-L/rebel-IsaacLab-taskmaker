# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'lego_g1' python package."""

import os
import sysconfig

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "psutil",
]

# Install a .pth file so the extension's gym environments are auto-registered
# at Python startup (works with stock IsaacLab scripts without modification).
PTH_FILENAME = "lego_g1_register.pth"
_pth_dir = sysconfig.get_path("purelib")
_pth_path = os.path.join(_pth_dir, PTH_FILENAME)
try:
    with open(_pth_path, "w") as _f:
        _f.write("import lego_g1\n")
except (PermissionError, OSError):
    pass

setup(
    name="lego_g1",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
