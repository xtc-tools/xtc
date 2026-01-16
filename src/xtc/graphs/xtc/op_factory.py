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
    XTCUnpad2DExpr,
    XTCPadExpr,
    XTCReshapeExpr,
    XTCTransposeExpr,
)

__all__ = [
    "matmul",
    "conv2d",
    "pad2d",
    "unpad2d",
    "pad",
    "relu",
    "reshape",
    "transpose",
    "tensor",
    "inputs",
    "outputs",
    "type",
]


def matmul(a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCMatmulExpr(a, b, **attrs),
        name=name,
    )


def conv2d(
    inp: XTCExpr, weight: XTCExpr, name: str | None = None, **attrs: Any
) -> XTCExpr:
    return XTCGraphContext.append(
        XTCConv2DExpr(inp, weight, **attrs),
        name=name,
    )


def pad2d(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCPad2DExpr(inp, **attrs),
        name=name,
    )


def unpad2d(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCUnpad2DExpr(inp, **attrs),
        name=name,
    )


def pad(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCPadExpr(inp, **attrs),
        name=name,
    )


def relu(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCReluExpr(inp, **attrs),
        name=name,
    )


def reshape(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCReshapeExpr(inp, **attrs),
        name=name,
    )


def transpose(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCTransposeExpr(inp, **attrs),
        name=name,
    )


def tensor(*args: Any, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCTensorExpr(*args, **attrs),
        name=name,
    )


def outputs(*outs: XTCExpr) -> None:
    XTCGraphContext.outputs(*outs)


def inputs(*inps: XTCExpr) -> None:
    XTCGraphContext.inputs(*inps)


def type(*args: Any, **attrs: Any) -> XTCTensorType:
    return XTCTensorType(*args, **attrs)
