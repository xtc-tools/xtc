#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import numpy as np
import utils


class Operation:
    def __init__(self, operator: "Operator", args: tuple, name=None) -> None:
        self.operator = operator
        self.args = args
        self.dim_names = self.operator.dim_names()
        self.axes_names = self.operator.axes_names()
        self.args_names = self.operator.args_names()
        self.name = self.operator.name if name is None else name

    def generate(self) -> tuple[str, str, str]:
        return self.operator.generate_op(*self.args, name=self.name)


class Operator:
    name = "undef"

    @classmethod
    def generate_op(cls, i, j, k, dtype, name=None):
        raise Exception("unimplemneted")


class OperatorMatmul(Operator):
    name = "matmul"
    source_op = """
__attribute__((always_inline)) void {{op_name}}({{ctype}} *out0, {{ctype}} *inp0, {{ctype}} *inp1) {
    *out0 += (*inp0) * (*inp1);
}
__attribute__((always_inline)) void {{op_name_0}}({{ctype}} *out0) {
    *out0 = 0;
}
"""
    jir_function = """
function {{name}}
  dimensions
    I, J, K
  buffers
    A: <I, K> {{ftype}}
    B: <K, J> {{ftype}}
    O: <I, J> {{ftype}}
  {
    I0: for i in I (O)
      J0: for j in J (O)
        {{op_name_0}}(O)
    II: for i in I (*)
      JJ: for j in J (*)
        KK: for k in K (*)
            {{op_name}}(O, A, B)
  }
"""
    _re_replace = utils.Replace(["name", "ctype", "ftype", "op_name_0", "op_name"])

    @classmethod
    def generate_op(cls, i, j, k, dtype, name=None) -> tuple:
        name = name if name is not None else cls.name
        replaces = {
            "name": name,
            "ctype": {"float32": "float", "float64": "double"}[dtype],
            "ftype": {"float32": "f32", "float64": "f64"}[dtype],
            "op_name": f"op_{name}",
            "op_name_0": f"op0_{name}",
        }
        source_op = cls._re_replace.replace(cls.source_op, **replaces)
        jir_function = cls._re_replace.replace(cls.jir_function, **replaces)
        return (source_op, jir_function)

    @classmethod
    def args_names(cls) -> tuple:
        return ("I", "J", "K", "DTYPE")

    @classmethod
    def dim_names(cls) -> tuple:
        return ("I", "J", "K")

    @classmethod
    def axes_names(cls) -> tuple:
        return ("II", "JJ", "KK")

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
