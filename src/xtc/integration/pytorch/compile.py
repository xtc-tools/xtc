#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch
import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.itf.comp.module import Module
from xtc.itf.schd.scheduler import Scheduler
from xtc.utils.cfunc import CFunc
from xtc.utils.ext_tools import get_shlib_extension
from xtc.utils.loader import LibLoader

# ---------------------------------------------------------------------------
# Shared operator compilation utilities
# ---------------------------------------------------------------------------


def torch_dtype_to_xtc(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"unsupported dtype for XTC operator: {dtype}")


def default_kernel_cache_dir() -> Path:
    return Path(os.environ.get("XTC_PYTORCH_CACHE", ".xtc_kernel_cache"))


def resolve_cache_dir(cache_dir: Path | None) -> Path:
    resolved = cache_dir if cache_dir is not None else default_kernel_cache_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def contiguous_strides_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major contiguous strides for ``shape`` (last dim stride 1)."""
    if not shape:
        return ()
    rev: list[int] = [1]
    for dim in reversed(shape[1:]):
        rev.append(rev[-1] * dim)
    return tuple(reversed(rev))


def validate_contiguous_strides(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    *,
    what: str,
) -> None:
    """Raise ``NotImplementedError`` when ``stride`` is not contiguous for ``shape``."""
    expected = contiguous_strides_for_shape(shape)
    if stride != expected:
        raise NotImplementedError(
            f"XTC operator requires contiguous {what} "
            f"(shape={shape}, stride={stride}, expected stride={expected})"
        )


@dataclass(frozen=True)
class KernelArtifacts:
    """Paths produced when compiling a specialized XTC operator kernel."""

    export_name: str
    cache_dir: Path
    module_path: Path
    export_dir: Path
    header_path: Path
    xtc_lib_path: Path
    payload_name: str
    shim_lib_path: Path
    shim_lib_name: str


# ---------------------------------------------------------------------------
# Matmul operator compilation
# ---------------------------------------------------------------------------


def validate_matmul_xtc_support(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    *,
    b_shape: tuple[int, ...] | None = None,
) -> None:
    """Raise ``NotImplementedError`` when XTC cannot specialize this matmul."""
    if device.type != "cpu":
        raise NotImplementedError(
            f"xtc::matmul: XTC matmul is only supported on CPU (got {device})"
        )
    try:
        torch_dtype_to_xtc(dtype)
    except ValueError as exc:
        raise NotImplementedError(
            f"xtc::matmul: XTC matmul does not support dtype {dtype}"
        ) from exc
    if len(x_shape) < 1:
        raise NotImplementedError(
            f"xtc::matmul: XTC matmul requires x with rank >= 1 (got shape {x_shape})"
        )
    if len(w_shape) != 2:
        raise NotImplementedError(
            f"xtc::matmul: XTC matmul requires 2D weight (got shape {w_shape})"
        )
    j, k_w = w_shape
    *leading, k = x_shape
    if k != k_w:
        raise NotImplementedError(
            f"xtc::matmul: in_features mismatch (x k={k}, w k={k_w})"
        )
    for name, dims in (("x", x_shape), ("w", w_shape)):
        if any(d <= 0 for d in dims):
            raise NotImplementedError(
                f"xtc::matmul: XTC matmul requires positive {name} dimensions "
                f"(got {dims})"
            )
    if b_shape is not None:
        if len(b_shape) != 1 or b_shape[0] != j:
            raise NotImplementedError(
                f"xtc::matmul: bias must be 1D with size {j} (got {b_shape})"
            )
        if b_shape[0] <= 0:
            raise NotImplementedError(
                f"xtc::matmul: bias must have positive size (got {b_shape})"
            )
    _ = leading


def validate_matmul_inductor_no_bias(has_bias: bool) -> None:
    """Raise ``NotImplementedError`` when the Inductor C++ shim receives a bias."""
    if has_bias:
        raise NotImplementedError(
            "xtc::matmul: XTC Inductor C++ integration does not support bias yet"
        )


def validate_matmul_inductor_layout(
    x_shape: tuple[int, ...],
    x_stride: tuple[int, ...],
    w_shape: tuple[int, int],
    w_stride: tuple[int, ...],
) -> None:
    """Require row-major contiguous tensor layouts for the Inductor C++ shim."""
    validate_contiguous_strides(x_shape, x_stride, what="x")
    validate_contiguous_strides(w_shape, w_stride, what="weight")


def matmul_cache_key(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[int, int, int, str]:
    *leading, k = x_shape
    j, k_w = w_shape
    if k != k_w:
        raise ValueError(f"in_features mismatch: x has {k}, w has {k_w}")
    i = prod(leading) if leading else 1
    return (i, j, k, torch_dtype_to_xtc(dtype))


def matmul_export_name(i: int, j: int, k: int, xtc_dtype: str) -> str:
    return f"matmul_{i}_{j}_{k}_{xtc_dtype}"


def validate_matmul_inductor_specialized_shapes(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    out_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int]:
    """Require 2D ``(i, k)``, ``(j, k)``, and ``(i, j)`` tensors for the AOTI C++ shim.

    The shim calls a specialized XTC matmul with fixed ``(i, j, k)``; shapes are
    checked here at Inductor lowering time instead of in generated C++.
    """
    i, j, k, _ = matmul_cache_key(x_shape, w_shape, dtype)
    expected_x = (i, k)
    expected_w = (j, k)
    expected_out = (i, j)

    if x_shape != expected_x:
        raise NotImplementedError(
            "xtc::matmul: XTC Inductor C++ integration requires 2D activations "
            f"with shape {expected_x} (got x shape {x_shape})"
        )
    if w_shape != expected_w:
        raise NotImplementedError(
            "xtc::matmul: XTC Inductor C++ integration requires weight shape "
            f"{expected_w} (got {w_shape})"
        )
    if out_shape is not None and out_shape != expected_out:
        raise NotImplementedError(
            "xtc::matmul: XTC Inductor C++ integration requires output shape "
            f"{expected_out} (got {out_shape})"
        )
    return i, j, k


def schedule_matmul(sch: Scheduler) -> None:
    sch.strip_mine("i", {"i1": 2})
    sch.strip_mine("j", {"j1": 16})
    sch.interchange(["k", "i", "j", "i1", "j1"])
    sch.vectorize(["j1"])
    sch.unroll({"i1": 2})


@dataclass(frozen=True)
class MatmulKernelArtifacts(KernelArtifacts):
    """Paths for a specialized matmul used by Inductor C++ integration."""

    i: int
    j: int
    k: int
    xtc_dtype: str


LinearKernelArtifacts = MatmulKernelArtifacts


def build_matmul_module(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    cache_dir: Path | None = None,
    dump_file: str | None = None,
) -> tuple[Module, MatmulKernelArtifacts]:
    i, j, k, xtc_dtype = matmul_cache_key(x_shape, w_shape, dtype)
    *leading, _ = x_shape
    leading_shape = tuple(leading)

    a = O.tensor((i, k), xtc_dtype, name="A")
    # PyTorch weights are (j, k); layout marks row-major storage for transpose_b.
    b = O.tensor((k, j), xtc_dtype, name="B", layout=[1, 0])

    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")

    resolved_cache = resolve_cache_dir(cache_dir)
    export_name = matmul_export_name(i, j, k, xtc_dtype)
    if dump_file is None:
        dump_file = str(resolved_cache / export_name)

    impl = Backend(gb.graph)
    sch = impl.get_scheduler()
    schedule_matmul(sch)
    sched = sch.schedule()

    comp = impl.get_compiler(shared_lib=True, dump_file=dump_file)
    module = comp.compile(sched)

    module_path = Path(module.file_name).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(
            f"XTC shared library not found after compile: {module_path}"
        )

    export_dir = resolved_cache / f"export_{export_name}"
    header_path = export_dir / "include" / f"{export_name}.h"
    ext = get_shlib_extension()
    xtc_lib_path = export_dir / "lib" / f"lib{export_name}.{ext}"
    shim_lib_name = f"xtc_matmul_shim_{i}_{j}_{k}_{xtc_dtype}"
    shim_lib_path = resolved_cache / f"lib{shim_lib_name}.{ext}"

    artifacts = MatmulKernelArtifacts(
        i=i,
        j=j,
        k=k,
        xtc_dtype=xtc_dtype,
        export_name=export_name,
        cache_dir=resolved_cache,
        module_path=module_path,
        export_dir=export_dir,
        header_path=header_path,
        xtc_lib_path=xtc_lib_path,
        payload_name=module.payload_name,
        shim_lib_path=shim_lib_path,
        shim_lib_name=shim_lib_name,
    )
    _ = leading_shape  # used by compile_matmul_kernel closure
    return module, artifacts


def compile_matmul_at_lowering(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    inductor_cpp: bool = False,
    cache_dir: Path | None = None,
    force: bool = False,
    x_stride: tuple[int, ...] | None = None,
    w_stride: tuple[int, ...] | None = None,
    out_shape: tuple[int, ...] | None = None,
    has_bias: bool = False,
) -> MatmulKernelArtifacts:
    """Compile the XTC matmul during Inductor lowering (export AOTI shim if needed)."""
    validate_matmul_xtc_support(
        x_shape,
        w_shape,
        dtype,
        torch.device("cpu"),
    )
    if inductor_cpp:
        validate_matmul_inductor_no_bias(has_bias)
        validate_matmul_inductor_specialized_shapes(
            x_shape, w_shape, dtype, out_shape=out_shape
        )
        if x_stride is None or w_stride is None:
            raise NotImplementedError(
                "xtc::matmul: XTC Inductor integration requires static "
                "strides at compile time"
            )
        validate_matmul_inductor_layout(
            x_shape,
            x_stride,
            w_shape,
            w_stride,
        )
        from xtc.integration.pytorch.inductor_cpp import (
            ensure_inductor_matmul_artifacts,
        )

        return ensure_inductor_matmul_artifacts(
            x_shape, w_shape, dtype, cache_dir=cache_dir, force=force
        )
    _, artifacts = build_matmul_module(x_shape, w_shape, dtype, cache_dir=cache_dir)
    return artifacts


def export_matmul_kernel(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> MatmulKernelArtifacts:
    """Compile (if needed), export C++ link tree, and build the Inductor AOTI shim."""
    module, artifacts = build_matmul_module(
        x_shape, w_shape, dtype, cache_dir=cache_dir
    )

    if force or not artifacts.header_path.is_file():
        artifacts.export_dir.mkdir(parents=True, exist_ok=True)
        module.export(artifacts.export_dir, name=artifacts.export_name)

    if force or not artifacts.shim_lib_path.is_file():
        from xtc.integration.pytorch.inductor_cpp import build_matmul_aoti_shim

        build_matmul_aoti_shim(artifacts, force=force)

    return artifacts


def compile_matmul_kernel(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    cache_dir: Path | None = None,
    dump_file: str | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Compile ``x @ w.mT`` (+ optional bias) with XTC and return a callable kernel."""
    i, j, k, xtc_dtype = matmul_cache_key(x_shape, w_shape, dtype)
    *leading, _ = x_shape

    module, artifacts = build_matmul_module(
        x_shape, w_shape, dtype, cache_dir=cache_dir, dump_file=dump_file
    )

    loader = LibLoader(str(artifacts.module_path))
    func = getattr(loader.lib, module.payload_name)
    func.packed = not getattr(module, "_bare_ptr", True)
    cfunc = CFunc(func)

    leading_shape = tuple(leading)

    def kernel(
        x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
    ) -> torch.Tensor:
        x2d = x.reshape(-1, k).contiguous()
        w2d = w.contiguous()
        y = torch.empty((*leading_shape, j), dtype=dtype, device=x.device)
        y2d = y.reshape(i, j).contiguous()
        cfunc(x2d.numpy(), w2d.numpy(), y2d.numpy())
        if b is not None:
            y = y + b
        return y

    kernel.loader = loader  # type: ignore[attr-defined]
    kernel.cfunc = cfunc  # type: ignore[attr-defined]
    return kernel
