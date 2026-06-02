#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from collections.abc import Callable

import torch

from xtc.integration.pytorch.compile import compile_matmul_kernel

_matmul_kernel_cache: dict[
    tuple[
        tuple[int, ...],
        tuple[int, int],
        tuple[int, ...] | None,
        torch.dtype,
        torch.device,
    ],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
] = {}


def clear_kernel_cache() -> None:
    """Clear eager kernel caches for all XTC PyTorch custom ops."""
    _matmul_kernel_cache.clear()


def matmul_cpu(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    x_shape = tuple(x.shape)
    w_shape = (int(w.shape[0]), int(w.shape[1]))
    b_shape = tuple(b.shape) if b is not None else None
    key = (x_shape, w_shape, b_shape, x.dtype, x.device)
    if key not in _matmul_kernel_cache:
        _matmul_kernel_cache[key] = compile_matmul_kernel(x_shape, w_shape, x.dtype)
    return _matmul_kernel_cache[key](x, w, b)
