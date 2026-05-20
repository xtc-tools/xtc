#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# pyright: reportFunctionMemberAccess=false
from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch.library import custom_op

from xtc.integration.pytorch.eager import matmul_cpu

# ---------------------------------------------------------------------------
# Matmul custom op (xtc::matmul)
# ---------------------------------------------------------------------------


@custom_op("xtc::matmul", mutates_args=())
def matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    raise RuntimeError("xtc::matmul: no kernel registered for this device")


@matmul.register_fake
def matmul_meta(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    torch._check(x.dim() >= 1)
    torch._check(w.dim() == 2)
    torch._check(x.shape[-1] == w.shape[1])
    if b is not None:
        torch._check(b.dim() == 1)
        torch._check(w.shape[0] == b.shape[0])
        torch._check(x.device == b.device)
    torch._check(x.device == w.device)
    return x.new_empty(*x.shape[:-1], w.shape[0])


def _matmul_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor
) -> None:
    x, w, b = inputs
    ctx.save_for_backward(x, w)
    ctx.has_bias = b is not None


def _matmul_backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x, w = ctx.saved_tensors
    k = w.shape[1]
    leading = x.shape[:-1]
    batch = prod(leading) if leading else 1
    x2d = x.reshape(batch, k)
    go2d = grad_output.reshape(batch, w.shape[0])
    grad_x = (go2d @ w).reshape(x.shape)
    grad_w = go2d.mT @ x2d
    grad_b = go2d.sum(dim=0) if ctx.has_bias else None
    return grad_x, grad_w, grad_b


# ---------------------------------------------------------------------------
# Register ops
# ---------------------------------------------------------------------------


def register_ops() -> None:
    # Register eager kernel
    matmul.register_kernel("cpu")(matmul_cpu)
    # Register autograd
    matmul.register_autograd(_matmul_backward, setup_context=_matmul_setup_context)
