#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import fresh_cache

from xtc.integration.pytorch import (
    XtcIntegration,
    clear_kernel_cache,
    register_torch_xtc_extensions,
)


@pytest.mark.skipif(sys.platform != "linux", reason="inductor-cpp XTC integration (linux)")
def test_inductor_cpp_matmul_matches_eager() -> None:
    try:
        import mlir  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"MLIR backend required: {exc}")

    clear_kernel_cache()
    register_torch_xtc_extensions(xtc_integration=XtcIntegration.INDUCTOR_CPP)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(16, 8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.w)

    m = M()
    x = torch.randn(4, 8)
    y_ref = m(x)
    with fresh_cache():
        y = torch.compile(m, backend="inductor")(x)
    assert torch.allclose(y, y_ref), f"max err={(y - y_ref).abs().max().item()}"
