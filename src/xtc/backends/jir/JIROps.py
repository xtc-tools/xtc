#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any, Type
import numpy as np
import xtc.utils as utils

__all__ = [
    "JIROperation",
    "JIROperator",
    "JIROperators",
]


class JIROperation:
    def __init__(
        self,
        operator: Type["JIROperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
    ) -> None:
        self.operator = operator(args, attrs, name=name)
        self.args = args
        self.dim_names = self.operator.dim_names()
        self.axes_names = self.operator.axes_names()
        self.args_names = self.operator.args_names()
        self.name = self.operator.name if name is None else name

    def generate(self) -> Any:
        return self.operator.generate_op()

    def np_inputs_spec(self):
        operator = self.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(operator.inputs_dims(), operator.inputs_types())
        ]

    def np_outputs_spec(self):
        operator = self.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(operator.outputs_dims(), operator.outputs_types())
        ]

    def reference_impl(self, *operands: Any) -> None:
        self.operator.reference_impl(*operands)


class JIROperator(ABC):
    DEFAULT_NAME = "undef"

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

    @abstractmethod
    def generate_op(self) -> Any: ...
    @abstractmethod
    def args_names(self) -> tuple[str, ...]: ...
    @abstractmethod
    def dim_names(self) -> tuple[str, ...]: ...
    @abstractmethod
    def axes_names(self) -> tuple[str, ...]: ...
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


class JIROperatorMatmul(JIROperator):
    DEFAULT_NAME = "matmul"
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

    @override
    def generate_op(self) -> Any:
        i, j, k, dtype = self.args
        replaces = {
            "name": self.name,
            "ctype": {"float32": "float", "float64": "double"}[dtype],
            "ftype": {"float32": "f32", "float64": "f64"}[dtype],
            "op_name": f"op_{self.name}",
            "op_name_0": f"op0_{self.name}",
        }
        source_op = self._re_replace.replace(self.source_op, **replaces)
        jir_function = self._re_replace.replace(self.jir_function, **replaces)
        return (source_op, jir_function)

    @override
    def args_names(self) -> tuple[str, ...]:
        return ("I", "J", "K", "DTYPE")

    @override
    def dim_names(self) -> tuple[str, ...]:
        return ("I", "J", "K")

    @override
    def axes_names(self) -> tuple[str, ...]:
        return ("II", "JJ", "KK")

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


class JIROperators:
    matmul = JIROperatorMatmul
