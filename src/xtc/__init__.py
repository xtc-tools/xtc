#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import importlib.metadata
import os
import sys

__version__ = importlib.metadata.version("xtc")
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../sampler"))
)
