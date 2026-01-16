#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import logging

from xtc.runtimes.host.runtime import runtime_funcs, resolve_runtime, RuntimeType

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

# GPU Runtime


def type() -> RuntimeType:
    return RuntimeType.GPU


def __getattr__(x: str):
    if x in runtime_funcs:
        entries = resolve_runtime(RuntimeType.GPU)
        assert entries is not None
        return entries[x]
    raise AttributeError(f"undefined runtime function: {x}")
