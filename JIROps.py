#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import numpy as np


class Operation:
    def __init__(self, operator: "Operator", args: tuple) -> None:
        self.operator = operator
        self.args = args
        self.args_names = self.operator.args_names()

    def generate(self) -> tuple[str, str, str]:
        return self.operator.generate_op(*self.args)


class Operator:
    name = "undef"

    @classmethod
    def generate_op(cls, i, j, k, dtype):
        raise Exception("unimplemneted")


class OperatorMatmul(Operator):
    name = "matmul"
    op_function = """
__attribute__((always_inline)) void %OP_ENTRY(%DTYPE *out0, %DTYPE *inp0, %DTYPE *inp1) {
    *out0 += (*inp0) * (*inp1);
}
__attribute__((always_inline)) void %OP_ENTRY_0(%DTYPE *out0) {
    *out0 = 0;
}
"""
    jir_function = """
function %ENTRY
  dimensions
    I, J, K
  buffers
    A: <I, K> f32
    B: <K, J> f32
    O: <I, J> f32
  {
    I0: for i in I (O)
      J0: for j in J (O)
        %OP_ENTRY_0(O)
    I: for i in I (*)
      J: for j in J (*)
        K: for k in K (*)
            %OP_ENTRY(O, A, B)
  }
"""

    @classmethod
    def generate_op(cls, i, j, k, dtype) -> tuple:
        entry = f"{cls.name}_{i}x{j}x{k}x{dtype}"
        op_entry = f"op_{entry}"
        op_entry_0 = f"op0_{entry}"
        op_dtype = {"float32": "float", "float64": "double"}[dtype]
        source_op = cls.op_function
        source_op = source_op.replace("%DTYPE", op_dtype)
        source_op = source_op.replace("%OP_ENTRY", op_entry)
        source_op = source_op.replace("%OP_ENTRY_0", op_entry_0)

        jir_function = cls.jir_function
        jir_function = jir_function.replace("%ENTRY", entry)
        jir_function = jir_function.replace("%OP_ENTRY", op_entry)
        jir_function = jir_function.replace("%OP_ENTRY_0", op_entry_0)
        return (source_op, jir_function, entry)

    @classmethod
    def args_names(cls) -> tuple:
        return ("I", "J", "K", "DTYPE")

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
