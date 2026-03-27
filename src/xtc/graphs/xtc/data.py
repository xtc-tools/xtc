#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override, Self
from typing import cast, Any, Iterable
import functools
import operator
import numpy as np
import numpy.typing

from xtc.itf.data import (
    Tensor,
    TensorType,
    ShapeType,
    DataType,
    ConstantTensorType,
    ConstantShapeType,
    ConstantDataType,
)

from xtc.itf.runtime.accelerator import AcceleratorDevice


__all__ = [
    "XTCTensorType",
    "XTCTensor",
]


class XTCTensorType(TensorType):
    def __init__(
        self,
        shape: ShapeType = None,
        dtype: DataType = None,
        device: AcceleratorDevice | None = None,
        const: bool = False,
        layout: list[int] | None = None,
    ):
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._const = const
        self._layout = layout

    @property
    @override
    def shape(self) -> ShapeType:
        return self._shape

    @property
    @override
    def dtype(self) -> DataType:
        return self._dtype

    @property
    @override
    def device(self) -> AcceleratorDevice | None:
        return self._device

    @property
    @override
    def const(self) -> bool:
        return self._const

    @property
    @override
    def layout(self) -> list[int] | None:
        return self._layout

    @property
    @override
    def ndim(self) -> int:
        return 0 if self._shape is None else len(self._shape)

    @property
    def size(self) -> int:
        if self._shape is None:
            return 0
        assert self.is_constant_shape(), "TODO: symboling shapes not implemented"
        return functools.reduce(operator.mul, self._shape, 1)

    @property
    def itemsize(self) -> int:
        assert self.is_constant_dtype(), "TODO: symboling dtype not implemented"
        return np.dtype(self.dtype).itemsize

    @property
    def nbytes(self) -> int:
        assert self.is_constant(), "TODO: symboling shape/dtype not implemented"
        return self.size * self.itemsize

    def is_constant_shape(self) -> bool:
        return self._shape is not None and all(
            [isinstance(dim, int) for dim in self._shape]
        )

    def is_constant_dtype(self) -> bool:
        return self.dtype is not None

    def is_constant(self) -> bool:
        return self.is_constant_shape() and self.is_constant_dtype()

    @property
    def constant_shape(self) -> tuple[int, ...]:
        assert self._shape is not None and self.is_constant_shape(), (
            "TODO: symboling shapes not implemented"
        )
        return (*[cast(int, dim) for dim in self._shape],)

    @property
    def constant_dtype(self) -> str:
        assert self.is_constant_dtype(), "TODO: symboling shapes not implemented"
        return cast(str, self.dtype)

    @property
    def constant(self) -> "XTCConstantTensorType":
        return XTCConstantTensorType(
            shape=self.constant_shape,
            dtype=self.constant_dtype,
            device=self.device,
            const=self.const,
            layout=self.layout,
        )

    @override
    def __repr__(self) -> str:
        if self._shape is None:
            dims = "??"
        else:
            dims = "x".join([str(d if d else "?") for d in self._shape])
        dtype = self._dtype if self._dtype else "?"
        const = ", const" if self.const else ""
        layout = ""
        if self.layout is not None:
            assert self._shape is not None
            layout += (
                ", <(" + ",".join([str(i) for i in range(len(self._shape))]) + ")->("
            )
            layout += ",".join([str(i) for i in self.layout]) + ")>"
        return f"{dims}x{dtype}{const}{layout}"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, XTCTensorType):
            return NotImplemented
        return (
            self.dtype == other.dtype
            and self.shape == other.shape
            and self.const == other.const
            and self.layout == other.layout
        )

    @override
    def to_dict(self) -> dict[str, Any]:
        tensor_dict: dict[str, Any] = {}
        if self.shape:
            tensor_dict["shape"] = list(cast(Iterable[int], self.shape))
        if self.dtype:
            tensor_dict["dtype"] = self.dtype
        return tensor_dict

    @override
    @classmethod
    def from_dict(cls, tensor_dict: dict[str, Any]) -> Self:
        return cls(
            shape=tuple(tensor_dict["shape"]) if "shape" in tensor_dict else None,
            dtype=tensor_dict["dtype"] if "dtype" in tensor_dict else None,
        )


class XTCConstantTensorType(XTCTensorType, ConstantTensorType):
    def __init__(
        self,
        shape: ConstantShapeType,
        dtype: ConstantDataType,
        device: AcceleratorDevice | None = None,
        const: bool = False,
        layout: list[int] | None = None,
    ):
        self._shape: ConstantShapeType = shape
        self._dtype: ConstantDataType = dtype
        self._device = device
        self._const = const
        self._layout = layout

    @property
    @override
    def shape(self) -> ConstantShapeType:
        return self._shape

    @property
    @override
    def dtype(self) -> ConstantDataType:
        return self._dtype


class XTCTensor(Tensor):
    def __init__(
        self, data: Any | None = None, type: XTCTensorType | None = None
    ) -> None:
        self._data = np.array(data) if data is not None else None
        if self._data is not None:
            self._type = XTCTensorType(self._data.shape, str(self._data.dtype))
        else:
            self._type = XTCTensorType() if type is None else type

    @property
    @override
    def type(self) -> XTCTensorType:
        return self._type

    @property
    @override
    def data(self) -> Any | None:
        return self._data

    @override
    def numpy(self) -> numpy.typing.NDArray:
        assert self._data is not None
        return np.array(self._data)

    @override
    def __repr__(self) -> str:
        if self._data is None:
            return f"Tensor(type={self._type}, data=None)"
        else:
            data = self._data.reshape((-1,))
            if len(data) > 8:
                data_str = (
                    f"{' '.join([f'{d:g}' for d in data[:4]])}..."
                    f"{' '.join([f'{d:g}' for d in data[-4:]])}"
                )
            else:
                data_str = f"{' '.join([f'{d:g}' for d in data])}"
            return f"Tensor(type={self._type}, data={data_str})"
