#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from .IREEConfig import IREEConfig
from .IREEBackend import IREEBackend
from .IREEBackend import IREEBackend as Backend

__all__ = [
    "Backend",
    "IREEBackend",
    "IREEConfig",
]
