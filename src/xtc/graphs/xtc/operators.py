#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import TypeAlias, cast, Any
from types import SimpleNamespace as NS
import functools
import operator
import numpy as np

from xtc.itf.operator import Operator
from xtc.itf.data import Tensor, TensorType

from .data import XTCTensor, XTCTensorType

__all__ = [
    "XTCOperator",
]


XTCOperatorAttr: TypeAlias = Any
XTCOperatorAttrs: TypeAlias = NS
XTCOperPaddingAttr: TypeAlias = (
    int | tuple[int] | tuple[int, int] | tuple[int, int, int, int]
)
XTCOperStrideAttr: TypeAlias = int | tuple[int] | tuple[int, int]


class XTCOperator(Operator):
    def __init__(self, name: str, **attrs: XTCOperatorAttr) -> None:
        self._name = name
        self._attrs = NS(**attrs)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    def attrs(self) -> XTCOperatorAttrs:
        return self._attrs

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        return inputs_types

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return inputs


class XTCOperTensor(XTCOperator):
    def __init__(self) -> None:
        super().__init__("tensor")


class XTCOperMatmul(XTCOperator):
    def __init__(self) -> None:
        super().__init__("matmul")

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # assume (IK, KJ) inputs and IJ output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) == 2
        assert len(inputs_types[1].shape) == 2
        i, k = cast(XTCTensorType, inputs_types[0]).constant_shape
        bk, j = cast(XTCTensorType, inputs_types[1]).constant_shape
        assert k == bk
        return [
            XTCTensorType(
                shape=(i, j),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        matmul = XTCTensor(np.matmul(inputs[0].numpy(), inputs[1].numpy()))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert matmul.type == expected_type, (
            f"output type mismatch expect: {matmul.type} != {expected_type}"
        )
        return [matmul]


class XTCOperRelu(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("relu", **attrs)
        self._threshold = 0 if "threshold" not in attrs else self.attrs.threshold

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        relu = XTCTensor(np.maximum(inputs[0].numpy(), self._threshold))
        return [relu]


class XTCOperConv2D(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("conv2d", **attrs)
        if "stride" not in attrs:
            self._stride = (1, 1)
        else:
            stride = self.attrs.stride
            if isinstance(stride, int):
                self._stride = (stride, stride)
            else:
                assert isinstance(stride, tuple), (
                    f"padding for pad2d of wrong type, expect int or tuple: {stride}"
                )
                if len(stride) == 1:
                    self._stride = tuple([stride[0]] * 4)
                else:
                    assert len(stride) == 2, (
                        f"stride for conv2d of wrong size, expected 1 or 2: {stride}"
                    )
                    self._stride = stride

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # TODO: assume (HWC, RSCF) inputs and HWF output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) >= 3
        assert len(inputs_types[1].shape) == 4
        inp_shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        weight_shape = cast(XTCTensorType, inputs_types[1]).constant_shape
        h, w, c = inp_shape[-3:]
        r, s, wc, f = weight_shape
        assert c == wc
        sh, sw = self._stride
        oh, ow = (h - r) // sh + 1, (w - s) // sw + 1
        return [
            XTCTensorType(
                shape=tuple([*inputs_types[0].shape[:-3], oh, ow, f]),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        # Note that the input is supposed to be already padded
        inp, weight = [inp.numpy() for inp in inputs]
        inp_shape = inp.shape
        h, w, c = inp_shape[-3:]
        r, s, _, f = weight.shape
        sh, sw = self._stride
        oh, ow = (h - r) // sh + 1, (w - s) // sw + 1
        out_shape = (*inp_shape[:-3], oh, ow, f)
        out = np.zeros(shape=out_shape, dtype=inp.dtype).reshape((-1, oh, ow, f))
        inp = inp.reshape(-1, h, w, c)
        for vb, voh, vow, vf in np.ndindex(out.shape):
            view = inp[vb, voh * sh : voh * sh + r, vow * sw : vow * sw + s, 0:c]
            elts = view * weight[:, :, :, vf]
            out[vb, voh, vow, vf] = np.sum(elts)
        conv = XTCTensor(out.reshape(out_shape))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert conv.type == expected_type, (
            f"output type mismatch expect: {conv.type} != {expected_type}"
        )
        return [conv]


class XTCOperPad2D(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("pad2d", **attrs)
        if "padding" not in attrs:
            self._padding = (0, 0, 0, 0)
        else:
            padding = self.attrs.padding
            if isinstance(padding, int):
                self._padding = (padding, padding, padding, padding)
            else:
                assert isinstance(padding, tuple), (
                    f"padding for pad2d of wrong type, expect int or tuple: {padding}"
                )
                if len(padding) == 1:
                    self._padding = (padding[0], padding[0], padding[0], padding[0])
                elif len(padding) == 2:
                    self._padding = (padding[0], padding[0], padding[1], padding[1])
                else:
                    assert len(padding) == 4, (
                        f"padding for pad2d of wrong size, expected 1, 2 or 4: {padding}"
                    )
                    self._padding = padding

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        # TODO: assume HWC input
        assert len(inputs_types) == 1
        assert inputs_types[0].shape is not None
        assert len(inputs_types[0].shape) >= 2
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        return [
            XTCTensorType(
                shape=tuple(
                    [
                        *inputs_types[0].shape[:-3],
                        shape[-3] + self._padding[0] + self._padding[1],
                        shape[-2] + self._padding[2] + self._padding[3],
                        inputs_types[0].shape[-1],
                    ]
                ),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        inp_shape = cast(XTCTensorType, inputs[0].type).constant_shape
        pad_2d = [
            (self._padding[0], self._padding[1]),
            (self._padding[2], self._padding[3]),
        ]
        pads = [(0, 0) for _ in range(len(inp_shape) - 3)] + pad_2d + [(0, 0)]
        padded = XTCTensor(data=np.pad(inputs[0].numpy(), pads))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert padded.type == expected_type, (
            f"output type mismatch expect: {padded.type} != {expected_type}"
        )
        return [padded]


class XTCOperReshape(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("reshape", **attrs)
        if "shape" not in attrs:
            self._shape = (-1,)
        else:
            self._shape = self.attrs.shape
            assert all([x is not None for x in self._shape])
            assert len([x for x in self._shape if x == -1]) <= 1

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        size = cast(XTCTensorType, inputs_types[0]).size
        fixed_size = functools.reduce(
            operator.mul, [x for x in self._shape if x != -1], 1
        )
        out_shape = tuple([x if x != -1 else size // fixed_size for x in self._shape])
        return [
            XTCTensorType(
                shape=out_shape,
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        reshaped = XTCTensor(inputs[0].numpy().reshape(self._shape))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert reshaped.type == expected_type, (
            f"output type mismatch expect: {reshaped.type} != {expected_type}"
        )
        return [reshaped]
