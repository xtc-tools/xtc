#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, TypeAlias
import threading

from xtc.itf.data import Tensor, TensorType

from .data import XTCTensor, XTCTensorType, ShapeType, DataType
from .operators import (
    XTCOperator,
    XTCOperTensor,
    XTCOperMatmul,
    XTCOperRelu,
    XTCOperConv2D,
    XTCOperPad2D,
    XTCOperUnpad2D,
    XTCOperPad,
    XTCOperReshape,
    XTCOperTranspose,
)

__all__ = [
    "XTCExpr",
    "XTCValueExpr",
    "XTCTensorExpr",
    "XTCOpExpr",
]

ArgumentsType: TypeAlias = tuple["XTCExpr", ...]


class XTCExprCounter:
    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    def get_idx(self) -> int:
        with self._lock:
            idx = self._count
            self._count += 1
        return idx

    @property
    def count(self):
        return self._count


class XTCExpr(ABC):
    _counter = XTCExprCounter()
    _idx_map: dict[int, "XTCExpr"] = {}

    def __init__(self) -> None:
        self._idx = self._counter.get_idx()
        self._idx_map[self._idx] = self

    def get_expr(self, idx: int) -> "XTCExpr":
        return self._idx_map[idx]

    def _str_indent(self, indent: int = 0) -> str:
        ind = " " * indent
        sep = ""
        str_expr = ""
        for line in str(self).splitlines():
            str_expr += f"{sep}{ind}{line}"
            sep = "\n"
        return str_expr

    @abstractmethod
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]: ...

    @abstractmethod
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]: ...

    @property
    @abstractmethod
    def op_name(self) -> str: ...

    @property
    @abstractmethod
    def args(self) -> ArgumentsType: ...

    @property
    def uid(self) -> str:
        return f"%{self._idx}"

    @override
    def __str__(self) -> str:
        return f"{self.uid} = ?"


class XTCValueExpr(XTCExpr):
    @property
    @abstractmethod
    def value(self) -> Any: ...

    @property
    @abstractmethod
    def type(self) -> Any: ...

    @property
    @override
    def args(self) -> ArgumentsType:
        return ()


class XTCTensorExpr(XTCValueExpr):
    def __init__(
        self,
        tensor: XTCTensorType | XTCTensor | ShapeType | None = None,
        shape: ShapeType | DataType = None,
        dtype: DataType = None,
    ) -> None:
        super().__init__()
        if tensor is None:
            assert shape is None or isinstance(shape, tuple)
            assert dtype is None or isinstance(dtype, str)
            type = XTCTensorType(shape=shape, dtype=dtype)
            value = XTCTensor(type=type)
        elif isinstance(tensor, XTCTensorType):
            assert shape is None and dtype is None
            value = XTCTensor(type=tensor)
        elif isinstance(tensor, XTCTensor):
            assert shape is None and dtype is None
            value = tensor
        elif isinstance(tensor, tuple):
            if shape is not None:
                assert isinstance(shape, str)
                assert dtype is None
                type = XTCTensorType(shape=tensor, dtype=shape)
            else:
                type = XTCTensorType(shape=tensor, dtype=dtype)
            value = XTCTensor(type=type)
        self._value = value
        self._op = XTCOperTensor()

    @property
    @override
    def value(self) -> XTCTensor:
        assert isinstance(self._value, XTCTensor)
        return self._value

    @property
    @override
    def type(self) -> XTCTensorType:
        assert isinstance(self._value, XTCTensor)
        return self._value.type

    @property
    @override
    def op_name(self) -> str:
        return self._op.name

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == 0
        assert self.type.is_constant(), (
            f"Tensor type not constant in tensor initializer"
        )
        return [self.type]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        assert len(inputs) == 0
        assert self.type.is_constant(), (
            f"Tensor type not constant in tensor initializer"
        )
        return [self.value]

    @override
    def __str__(self) -> str:
        args = ", ".join([arg.uid for arg in self.args])
        return f"{self.uid} = {self.op_name}({args})"


class XTCOpExpr(XTCExpr):
    def __init__(self, op: XTCOperator, args: ArgumentsType) -> None:
        super().__init__()
        self._op = op
        self._args = args

    @property
    @override
    def op_name(self) -> str:
        return self._op.name

    @property
    @override
    def args(self) -> ArgumentsType:
        return self._args

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == len(self.args), (
            f"len of inputs types mismatch : {len(inputs_types)} == {len(self.args)}"
        )
        return self._op.forward_types(inputs_types)

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        assert len(inputs) == len(self.args), (
            f"len of inputs mismatch : {len(inputs)} == {len(self.args)}"
        )
        return self._op.forward(inputs)

    @override
    def __str__(self) -> str:
        params = [arg.uid for arg in self.args]
        params += [f"{attr}={value}" for attr, value in self._op.attrs.__dict__.items()]
        args = ", ".join(params)
        return f"{self.uid} = {self.op_name}({args})"


class XTCMatmulExpr(XTCOpExpr):
    def __init__(self, x: XTCExpr, y: XTCExpr, **attrs: Any) -> None:
        super().__init__(
            XTCOperMatmul(**attrs),
            (
                x,
                y,
            ),
        )


class XTCReluExpr(XTCOpExpr):
    def __init__(self, inp: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperRelu(**attrs), (inp,))


class XTCConv2DExpr(XTCOpExpr):
    def __init__(self, inp: XTCExpr, weight: XTCExpr, **attrs: Any) -> None:
        super().__init__(
            XTCOperConv2D(**attrs),
            (
                inp,
                weight,
            ),
        )


class XTCPad2DExpr(XTCOpExpr):
    def __init__(self, inp: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperPad2D(**attrs), (inp,))


class XTCUnpad2DExpr(XTCOpExpr):
    def __init__(self, inp: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperUnpad2D(**attrs), (inp,))


class XTCPadExpr(XTCOpExpr):
    def __init__(self, inp: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperPad(**attrs), (inp,))


class XTCReshapeExpr(XTCOpExpr):
    def __init__(self, x: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperReshape(**attrs), (x,))


class XTCTransposeExpr(XTCOpExpr):
    def __init__(self, x: XTCExpr, **attrs: Any) -> None:
        super().__init__(XTCOperTranspose(**attrs), (x,))
