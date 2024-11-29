#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from setuptools import setup
import os
import sys
import shutil


def get_venv_path():
    if hasattr(sys, "real_prefix"):
        return sys.prefix
    elif sys.base_prefix != sys.prefix:
        return sys.prefix
    else:
        return None


venv_path = get_venv_path()
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
dest_path = f"{venv_path}/lib/{python_version}/site-packages/ctools"

setup_dir = os.path.dirname(os.path.abspath(__file__))
src_path = f"{setup_dir}/src/ctools"

if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
shutil.copytree(src_path, dest_path)
setup(
    name="mlir_loop",
)
