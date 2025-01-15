#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import numpy as np
from typing import Any

__all__ = [
    "Operation",
    "Operators",
    "Operator",
    "OperatorMatmul",
]

import tvm
import tvm.te as te
import tvm.relay as relay


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


def tvm_build_crt(sch, operation, target, name=None):
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


class Operation:
    def __init__(self, operator, args, name=None, target=None):
        self.operator = operator
        self.args = args
        if target is not None:
            self.target_options = ""
            self.target = target
        else:
            self.target_options = get_tvm_native_target_options()
            self.target = "llvm"
        self.tgt = tvm.target.Target(target=f"{self.target} {self.target_options}")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.name = self.operator.name if name is None else name

    def generate(self) -> tuple:
        return self.operator.generate_op(*self.args)

    def schedule(self, operation: tuple, schedule: str = "") -> Any:
        sch = te.create_schedule(operation[-1].op)
        if schedule:
            exec(schedule, {"sch": sch, "obj": operation}, {})
        return sch

    def build(self, operation: tuple, sch: Any) -> Any:
        return tvm_build_crt(
            sch,
            operation,
            name=self.name,
            target=self.tgt,
        )

    def lower(self, operation: tuple, sch: Any) -> str:
        return tvm.lower(sch, operation, simple_mode=True)

    def np_inputs_spec(self):
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.inputs_dims(*self.args),
                self.operator.inputs_types(*self.args),
            )
        ]
        return inputs_spec

    def np_outputs_spec(self):
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.outputs_dims(*self.args),
                self.operator.outputs_types(*self.args),
            )
        ]
        return outputs_spec

    def reference_impl(self, *args):
        self.operator.reference_impl(*args)


class Operator:
    name = "undef"

    @staticmethod
    def generate_op(i, j, k, dtype):
        raise Exception("unimplemneted")


class OperatorMatmul(Operator):
    name = "matmul"

    @staticmethod
    def generate_op(i, j, k, dtype, name=None):
        name = name if name is not None else OperatorMatmul.name
        A = te.placeholder((i, k), name="A")
        B = te.placeholder((k, j), name="B")

        k = te.reduce_axis((0, k), "k")
        O = te.compute(
            (i, j),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            attrs={"layout_free_placeholders": [B]},
            name=name,
        )
        return A, B, O

    @staticmethod
    def inputs_dims(i, j, k, dtype):
        return (i, k), (k, j)

    @staticmethod
    def inputs_types(i, j, k, dtype):
        return dtype, dtype

    @staticmethod
    def outputs_dims(i, j, k, dtype):
        return ((i, j),)

    @staticmethod
    def outputs_types(i, j, k, dtype):
        return (dtype,)

    @staticmethod
    def reference_impl(*args):
        np.matmul(args[0], args[1], out=args[2])


class Operators:
    matmul = OperatorMatmul
