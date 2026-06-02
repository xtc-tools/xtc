#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# pyright: reportFunctionMemberAccess=false
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from sympy import Expr
from torch._inductor import ir
from torch._inductor.ir import ExternKernelAlloc, ExternKernelOut, TensorBox
from torch._inductor.lowering import lowerings, register_lowering
from torch._inductor.virtualized import V

from xtc.integration.pytorch.compile import (
    compile_matmul_at_lowering,
    validate_matmul_xtc_support,
)

# ---------------------------------------------------------------------------
# Shared Inductor lowering helpers (reused by future XTC custom ops)
# ---------------------------------------------------------------------------


def _static_dim_or_nyi(expr: Expr | int, *, what: str) -> int:
    try:
        return V.graph.sizevars.size_hint_or_throw(expr)
    except Exception as exc:
        raise NotImplementedError(
            f"XTC operator requires static {what} at compile time"
        ) from exc


def _static_strides_or_nyi(
    tensor: Any, shape: tuple[int, ...], *, what: str
) -> tuple[int, ...]:
    stride = tensor.maybe_get_stride()
    if stride is None:
        raise NotImplementedError(
            f"XTC operator requires static {what} strides at compile time"
        )
    if len(stride) != len(shape):
        raise NotImplementedError(
            f"XTC operator {what} stride rank mismatch "
            f"(shape={shape}, stride len={len(stride)})"
        )
    return tuple(_static_dim_or_nyi(s, what=f"{what} stride") for s in stride)


def _register_extern_lowering(
    op_overload: Any,
    inputs: list[Any],
    kwargs: dict[str, Any],
    *,
    output_size: list[Any],
    device: torch.device,
    dtype: torch.dtype,
    cpp_kernel_name: str | None = None,
) -> Any:
    """Build an Inductor ExternKernel node for a registered XTC custom op."""
    output_layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=output_size,
        stride=ir.FlexibleLayout.contiguous_strides(cast(Sequence[int], output_size)),
    )

    extern_cls = ExternKernelOut if V.graph.cpp_wrapper else ExternKernelAlloc
    extern_kwargs: dict[str, Any] = {
        "layout": output_layout,
        "inputs": inputs,
        "kwargs": kwargs,
        "op_overload": op_overload,
    }
    if extern_cls is ExternKernelOut:
        extern_kwargs["output_view"] = None
        extern_kwargs["cpp_kernel_name"] = cpp_kernel_name

    extern = extern_cls(**extern_kwargs)
    return TensorBox.create(extern)


# ---------------------------------------------------------------------------
# Matmul Inductor lowering (xtc::matmul)
# ---------------------------------------------------------------------------


def _register_matmul_inductor_lowering() -> None:
    @register_lowering(torch.ops.xtc.matmul.default, type_promotion_kind=None)
    def matmul_lowering(x: Any, w: Any, b: Any | None = None) -> Any:
        x.realize()
        w.realize()
        if b is not None:
            b.realize()

        if V.graph.cpp_wrapper and b is not None:
            raise NotImplementedError(
                "xtc::matmul: XTC Inductor C++ integration does not support bias yet"
            )

        x_sizes = list(x.get_size())
        k = x_sizes[-1]
        x_prefix = x_sizes[:-1]
        j, k_w = w.get_size()
        V.graph.sizevars.check_equals(k, k_w)

        device = x.get_device()
        if device is None:
            raise NotImplementedError(
                "xtc::matmul: XTC matmul requires a concrete device at compile time"
            )
        if device.type != "cpu":
            raise NotImplementedError(
                f"xtc::matmul: XTC matmul is only supported on CPU (got {device})"
            )

        dtype = x.get_dtype()
        try:
            x_static = tuple(_static_dim_or_nyi(s, what="x shape") for s in x_sizes)
            j_static = _static_dim_or_nyi(j, what="weight out_features")
            k_static = _static_dim_or_nyi(k_w, what="weight in_features")
        except NotImplementedError:
            raise
        except Exception as exc:
            raise NotImplementedError(
                "xtc::matmul: XTC matmul requires fully static shapes at compile time"
            ) from exc

        w_static = (j_static, k_static)
        b_static: tuple[int, ...] | None = None
        if b is not None:
            b_sizes = list(b.get_size())
            try:
                b_static = tuple(
                    _static_dim_or_nyi(s, what="bias shape") for s in b_sizes
                )
            except NotImplementedError:
                raise
            b_device = b.get_device()
            if b_device is not None and b_device.type != "cpu":
                raise NotImplementedError(
                    f"xtc::matmul: XTC matmul bias must be on CPU (got {b_device})"
                )
            if b.get_dtype() != dtype:
                raise NotImplementedError(
                    "xtc::matmul: XTC matmul requires bias dtype to match activations"
                )

        validate_matmul_xtc_support(x_static, w_static, dtype, device, b_shape=b_static)

        x_stride = _static_strides_or_nyi(x, x_static, what="x")
        w_stride = _static_strides_or_nyi(w, w_static, what="weight")

        output_size = x_prefix + [j]
        out_static: tuple[int, ...] | None = None
        if V.graph.cpp_wrapper:
            try:
                out_static = tuple(
                    _static_dim_or_nyi(s, what="output shape") for s in output_size
                )
            except NotImplementedError:
                raise
            except Exception as exc:
                raise NotImplementedError(
                    "xtc::matmul: XTC matmul requires fully static output shape "
                    "at compile time"
                ) from exc

        compile_matmul_at_lowering(
            x_static,
            w_static,
            dtype,
            inductor_cpp=V.graph.cpp_wrapper,
            x_stride=x_stride if V.graph.cpp_wrapper else None,
            w_stride=w_stride if V.graph.cpp_wrapper else None,
            out_shape=out_static,
            has_bias=b is not None,
        )

        inputs: list[Any] = [x, w]
        kwargs: dict[str, Any] = {}
        if b is not None:
            inputs.append(b)
        else:
            kwargs["b"] = None

        return _register_extern_lowering(
            torch.ops.xtc.matmul.default,
            inputs,
            kwargs,
            output_size=output_size,
            device=device,
            dtype=dtype,
            cpp_kernel_name="xtc::matmul_cpp" if V.graph.cpp_wrapper else None,
        )


def register_inductor_lowerings() -> None:
    _register_matmul_inductor_lowering()

    assert torch.ops.xtc.matmul.default in lowerings, (
        "matmul Inductor lowering was not registered"
    )
