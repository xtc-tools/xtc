#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from .operations import register_operation
from .subgraph_sizes import *

__all__ = []


def _register_conv2d_ops():
    for group in [squeezenet_convs, alexnet_convs]:
        for name, params in group.items():
            register_operation(
                "conv2d",
                name,
                {k: params[k] for k in ["n", "h", "w", "f", "r", "s", "c"]},
                {"SH": params["stry"], "SW": params["strx"]},
            )


def _register_matmul_ops():
    for group in [alexnet_matmuls]:
        for name, params in group.items():
            register_operation(
                "matmul",
                name,
                {k: params[k] for k in ["i", "j", "k"]},
            )


def _register_operations():
    _register_conv2d_ops()
    _register_matmul_ops()
