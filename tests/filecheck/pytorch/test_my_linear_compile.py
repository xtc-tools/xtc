# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from xtc.integration.pytorch import (
    XtcIntegration,
    clear_kernel_cache,
    register_torch_xtc_extensions,
)


def main() -> None:
    clear_kernel_cache()
    register_torch_xtc_extensions(xtc_integration=XtcIntegration.EAGER)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(16, 8))
            self.b = nn.Parameter(torch.randn(16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.w, self.b)

    m = M()
    x = torch.randn(4, 8)
    y_ref = m(x)
    compiled = torch.compile(m)
    y = compiled(x)
    max_err = (y - y_ref).abs().max().item()
    print(f"MAX_ERR: {max_err}")
    if not torch.allclose(y, y_ref):
        sys.exit(1)
    print("OK: torch.compile + XTC linear matches eager reference")


if __name__ == "__main__":
    main()

# CHECK:       MAX_ERR: 0.0
# CHECK-NEXT:  OK: torch.compile + XTC linear matches eager reference
