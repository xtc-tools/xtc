#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

from .context import XTCGraphContext
from .data import XTCTensorType
from .expr import (
    XTCExpr,
    XTCTensorExpr,
    XTCMatmulExpr,
    XTCReluExpr,
    XTCConv2DExpr,
    XTCPad2DExpr,
    XTCReshapeExpr,
)

__all__ = [
    "matmul",
    "conv2d",
    "pad2d",
    "relu",
    "reshape",
    "tensor",
    "inputs",
    "outputs",
    "type",
]


def matmul(a: XTCExpr, b: XTCExpr, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCMatmulExpr(a, b, **attrs))


def conv2d(inp: XTCExpr, weight: XTCExpr, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCConv2DExpr(inp, weight, **attrs))


def pad2d(inp: XTCExpr, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCPad2DExpr(inp, **attrs))


def relu(inp: XTCExpr, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCReluExpr(inp, **attrs))


def reshape(inp: XTCExpr, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCReshapeExpr(inp, **attrs))


def tensor(*args: Any, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCTensorExpr(*args, **attrs))


def outputs(*outs: XTCExpr) -> None:
    XTCGraphContext.outputs(*outs)


def inputs(*inps: XTCExpr) -> None:
    XTCGraphContext.inputs(*inps)


def type(*args: Any, **attrs: Any) -> XTCTensorType:
    return XTCTensorType(*args, **attrs)
