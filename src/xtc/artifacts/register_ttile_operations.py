#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xtc.schedules.ttile.prob_sizes import (
    ddsizes_Yolo,
    ddsizes_MobilNet,
    ddsizes_RN18,
    ddsizes_matmul,
)

from .operations import register_operation

__all__ = []


def _register_conv2d_ops():
    map = dict(
        h="y",
        w="x",
        r="h",
        s="w",
    )
    for group in [ddsizes_Yolo, ddsizes_MobilNet, ddsizes_RN18]:
        for name, params in group.items():
            register_operation(
                "conv2d",
                name,
                {k: params[map.get(k, k)] for k in ["n", "h", "w", "f", "r", "s", "c"]},
                {"SH": params["stry"], "SW": params["strx"]},
            )


def _register_matmul_ops():
    for group in [ddsizes_matmul]:
        for name, params in group.items():
            register_operation(
                "matmul",
                name,
                {k: params[k] for k in ["i", "j", "k"]},
            )


def _register_operations():
    _register_conv2d_ops()
    _register_matmul_ops()
