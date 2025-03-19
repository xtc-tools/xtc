#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
import numpy as np
from typing import Any, Type

__all__ = [
    "TVMOperation",
    "TVMOperator",
    "TVMOperators",
]

import tvm
import tvm.te as te


def get_tvm_native_target_options() -> str:
    """
    Returm the tvm target options to pass to llvm.
    """
    from cpuinfo import get_cpu_info

    info = get_cpu_info()
    arch = info["arch_string_raw"]
    flags = info["flags"]
    cpu, attrs, triple = "", "", ""
    if arch == "x86_64":
        triple = "x86_64-linux-gnu"
        if "avx512f" in flags:
            cpu = "skylake-avx512"
        elif "avx2" in flags:
            cpu = "core-avx2"
    elif arch == "aarch64":
        triple = "aarch64-linux-gnu"
        if "asimd" in flags:
            cpu = "cortex-a72"
            attrs = "+neon"
    target_options = []
    if triple:
        target_options.append(f"-mtriple={triple}")
    if cpu:
        target_options.append(f"-mcpu={cpu}")
    if attrs:
        target_options.append(f"-mattrs={attrs}")
    return " ".join(target_options)


def tvm_build_crt(
    sch: Any, operation: Any, target: str, name: str | None = None
) -> Any:
    # We use system-lib with crt runtime such that DSO loading works
    # The generated .so can then be used:
    # - for static compilation as soon as the tvm runtime is provided
    # - for dynamic loading from python
    # Recent version of tvm (i.e. 0.19) have a Runtime object
    # Older version (i.e. 0.16) support passing runtime options in target
    try:
        from tvm.relay.backend import Runtime

        runtime_kwargs = {
            "runtime": Runtime("crt", {"system-lib": True}),
        }
    except:
        runtime_kwargs = {}
        target = f"{target} --system-lib --runtime=c"
    return tvm.build(sch, operation, target=target, name=name, **runtime_kwargs)


class TVMOperation:
    def __init__(
        self,
        operator: Type["TVMOperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
        target: str | None = None,
    ) -> None:
        self.operator = operator(args, attrs, name=name)
        self.args = args
        self.attrs = attrs
        if target is not None:
            self.target_options = ""
            self.target = target
        else:
            self.target_options = get_tvm_native_target_options()
            self.target = "llvm"
        self.tgt = tvm.target.Target(target=f"{self.target} {self.target_options}")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.name = self.operator.name if name is None else name

    def generate(self) -> Any:
        return self.operator.generate_op()

    def schedule(self, operation: Any, schedule: str = "") -> Any:
        sch = te.create_schedule(operation[-1].op)
        if schedule:
            exec(schedule, {"sch": sch, "obj": operation}, {})
        return sch

    def build(self, operation: Any, sch: Any, func_name: str | None = None) -> Any:
        if func_name is None:
            func_name = self.name
        return tvm_build_crt(
            sch,
            operation,
            name=func_name,
            target=self.tgt,
        )

    def lower(self, operation: Any, sch: Any) -> str:
        return tvm.lower(sch, operation, simple_mode=True)

    def np_inputs_spec(self) -> list[dict[str, Any]]:
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.inputs_dims(), self.operator.inputs_types()
            )
        ]
        return inputs_spec

    def np_outputs_spec(self) -> list[dict[str, Any]]:
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.outputs_dims(), self.operator.outputs_types()
            )
        ]
        return outputs_spec

    def reference_impl(self, *args: Any) -> None:
        self.operator.reference_impl(*args)


class TVMOperator(ABC):
    DEFAULT_NAME = "undef"

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

    @abstractmethod
    def generate_op(self): ...
    @abstractmethod
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def inputs_types(self) -> tuple[str, ...]: ...
    @abstractmethod
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def outputs_types(self) -> tuple[str, ...]: ...
    @abstractmethod
    def reference_impl(self, *args: Any) -> None: ...


class TVMOperatorMatmul(TVMOperator):
    DEFAULT_NAME = "matmul"

    @override
    def generate_op(self) -> Any:
        i, j, k, dtype = self.args
        A = te.placeholder((i, k), name="A")
        B = te.placeholder((k, j), name="B")
        k = te.reduce_axis((0, k), "k")
        O = te.compute(
            (i, j),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            attrs={"layout_free_placeholders": [B]},
            name=self.name,
        )
        return A, B, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k, dtype = self.args
        return (i, k), (k, j)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        i, j, k, dtype = self.args
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k, dtype = self.args
        return ((i, j),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        i, j, k, dtype = self.args
        return (dtype,)

    @override
    def reference_impl(self, *args: Any) -> None:
        np.matmul(args[0], args[1], out=args[2])


class TVMOperatorRelu(TVMOperator):
    DEFAULT_NAME = "relu"
    DEFAULT_THRESHOLD = 0

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        attrs = {"threshold": self.DEFAULT_THRESHOLD, **attrs}
        super().__init__(args, attrs, name)

    @override
    def generate_op(self) -> Any:
        i, dtype = self.args
        A = te.placeholder((i,), name="A")
        O = te.compute(
            (i,),
            lambda i,: tvm.tir.max(self.attrs["threshold"], A[i]),
            name=self.name,
        )
        return A, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, dtype = self.args
        return ((i,),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        i, dtype = self.args
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, dtype = self.args
        return ((i,),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        i, dtype = self.args
        return (dtype,)

    @override
    def reference_impl(self, *args: Any) -> None:
        np.maximum(args[0], self.attrs["threshold"], out=args[1])


class TVMOperators:
    matmul = TVMOperatorMatmul
    relu = TVMOperatorRelu
