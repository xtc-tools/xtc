#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
__all__ = ["make_loop_name", "basename"]

_LOOP_SEP = "/"


def make_loop_name(root: str, name: str) -> str:
    return f"{root}{_LOOP_SEP}{name}"


def basename(loop_name: str) -> str:
    return loop_name.split(_LOOP_SEP)[-1]
