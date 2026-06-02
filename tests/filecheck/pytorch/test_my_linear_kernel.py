# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from xtc.integration.pytorch import clear_kernel_cache, compile_matmul_kernel


def main() -> None:
    clear_kernel_cache()
    with tempfile.TemporaryDirectory(prefix="xtc_pytorch_") as tmp:
        cache_dir = Path(tmp)
        kernel = compile_matmul_kernel(
            (4, 8),
            (16, 8),
            torch.float32,
            cache_dir=cache_dir,
        )
        print("KERNEL: compiled")

        x = torch.randn(4, 8)
        w = torch.randn(16, 8)
        b = torch.randn(16)
        y = kernel(x, w, b)
        y_ref = F.linear(x, w, b)
        max_err = (y - y_ref).abs().max().item()
        print(f"MAX_ERR: {max_err}")
        if not torch.allclose(y, y_ref):
            sys.exit(1)
        print("OK: XTC linear kernel matches F.linear")


if __name__ == "__main__":
    main()

# CHECK: KERNEL: compiled
# CHECK-NEXT: MAX_ERR: 0
# CHECK-NEXT: OK: XTC linear kernel matches F.linear
