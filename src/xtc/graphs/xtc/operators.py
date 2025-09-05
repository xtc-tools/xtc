#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import TypeAlias, cast, Any
from types import SimpleNamespace as NS
from collections.abc import Sequence, Mapping
import functools
import operator
import numpy as np

from xtc.itf.operator import Operator
from xtc.itf.data import Tensor, TensorType

from .data import XTCTensor, XTCTensorType
from .operation import XTCOperation

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
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        return [cast(XTCTensorType, inp_type) for inp_type in inputs_types]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return [cast(XTCTensor, inp) for inp in inputs]

    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        return self._get_operation(
            inps_types,
            outs_types,
        )

    def _get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
        dims: Mapping[str, int | str] = {},
        kinds: Sequence[str] = (),
        inps_maps: Sequence[Sequence[str]] = (),
        outs_maps: Sequence[Sequence[str]] = (),
    ) -> XTCOperation:
        # TODO: no symbolic shape
        inputs_types = [inp.constant for inp in inps_types]
        outputs_types = [out.constant for out in outs_types]
        return XTCOperation(
            name=self.name,
            attrs=self.attrs.__dict__,
            inputs_types=tuple(inputs_types),
            outputs_types=tuple(outputs_types),
            dims=dims,
            kinds=kinds,
            inps_maps=inps_maps,
            outs_maps=outs_maps,
        )


class XTCOperTensor(XTCOperator):
    def __init__(self) -> None:
        super().__init__("tensor")

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inputs_types = [inp.constant for inp in inps_types]
        outputs_types = [out.constant for out in outs_types]
        return XTCOperation(
            name=self.name,
            attrs=self.attrs.__dict__,
            inputs_types=tuple(inputs_types),
            outputs_types=tuple(outputs_types),
            dims={"i": outputs_types[0].size},
            kinds=("P",),
            inps_maps=(),
            outs_maps=(("i",)),
        )


class XTCOperMatmul(XTCOperator):
    def __init__(self) -> None:
        super().__init__("matmul")

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp0_shape = inps_types[0].constant_shape
        inp1_shape = inps_types[1].constant_shape
        i, k = inp0_shape
        bk, j = inp1_shape
        assert k == bk
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"i": i, "j": j, "k": k},
            kinds=("P", "P", "R"),
            inps_maps=(
                ("i", "k"),
                ("k", "j"),
            ),
            outs_maps=(("i", "j"),),
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        # assume (IK, KJ) inputs and IJ output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) == 2
        assert len(inputs_types[1].shape) == 2
        i, k = cast(XTCTensorType, inputs_types[0]).constant_shape
        bk, j = cast(XTCTensorType, inputs_types[1]).constant_shape
        assert k == bk, (
            f"incompatible dimension k for matmul inputs shapes: ({i}, {k}) ({bk}, {j})"
        )
        return [
            XTCTensorType(
                shape=(i, j),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
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
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        i = functools.reduce(operator.mul, inp_shape, 1)
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"i": i},
            kinds=("P",),
            inps_maps=(("i",)),
            outs_maps=(("i",)),
        )

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
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
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        assert len(inp_shape) >= 3
        weight_shape = inps_types[1].constant_shape
        assert len(weight_shape) == 4
        b = functools.reduce(operator.mul, inp_shape[:-3], 1)
        h, w, c = inp_shape[-3:]
        r, s, wc, f = weight_shape
        assert c == wc
        sh, sw = self._stride
        oh, ow = (h - r) // sh + 1, (w - s) // sw + 1
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"b": b, "h": oh, "w": ow, "f": f, "r": r, "s": s, "c": c},
            kinds=("P", "P", "P", "P", "R", "R", "R"),
            inps_maps=(
                ("b", f"h*{sh}+r", f"w*{sw}+s", "c"),
                ("r", "s", "c", "f"),
            ),
            outs_maps=(("b", "h", "w", "f"),),
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        # TODO: assume (NHWC, RSCF) inputs and HWF output
        assert len(inputs_types) == 2
        assert inputs_types[0].shape is not None
        assert inputs_types[1].shape is not None
        assert len(inputs_types[0].shape) >= 3
        assert len(inputs_types[1].shape) == 4
        inp_shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        weight_shape = cast(XTCTensorType, inputs_types[1]).constant_shape
        h, w, c = inp_shape[-3:]
        r, s, wc, f = weight_shape
        assert c == wc, f"unexpected with shapes: {inp_shape = }, {weight_shape = }"
        sh, sw = self._stride
        oh, ow = (h - r) // sh + 1, (w - s) // sw + 1
        return [
            XTCTensorType(
                shape=tuple([*inputs_types[0].shape[:-3], oh, ow, f]),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
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
        padding = attrs.get("padding", (0, 0, 0, 0))
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        else:
            assert isinstance(padding, tuple), (
                f"padding for pad2d of wrong type, expect int or tuple: {padding}"
            )
            if len(padding) == 1:
                padding = (padding[0], padding[0], padding[0], padding[0])
            elif len(padding) == 2:
                padding = (padding[0], padding[0], padding[1], padding[1])
            else:
                assert len(padding) == 4, (
                    f"padding for pad2d of wrong size, expected 1, 2 or 4: {padding}"
                )
        super().__init__("pad2d", padding=padding)

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        assert len(inp_shape) >= 3
        padding = self.attrs.padding
        b = functools.reduce(operator.mul, inp_shape[:-3], 1)
        oh, ow, c = (
            inp_shape[-3] + padding[0] + padding[1],
            inp_shape[-2] + padding[2] + padding[3],
            inp_shape[-1],
        )
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"b": b, "h": oh, "w": ow, "c": c},
            kinds=("P", "P", "P", "P"),
            inps_maps=(("b", f"h-{padding[0]}", f"w-{padding[2]}", "c"),),
            outs_maps=(("b", "h", "w", "c"),),
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        # TODO: assume NHWC input
        assert len(inputs_types) == 1
        assert inputs_types[0].shape is not None
        assert len(inputs_types[0].shape) >= 3
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        padding = self.attrs.padding
        return [
            XTCTensorType(
                shape=tuple(
                    [
                        *shape[:-3],
                        shape[-3] + padding[0] + padding[1],
                        shape[-2] + padding[2] + padding[3],
                        shape[-1],
                    ]
                ),
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        shape = cast(XTCTensorType, inputs[0].type).constant_shape
        padding = self.attrs.padding
        pad_2d = [(padding[0], padding[1]), (padding[2], padding[3])]
        pads = [(0, 0) for _ in range(len(shape) - 3)] + pad_2d + [(0, 0)]
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
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        i = functools.reduce(operator.mul, inp_shape, 1)
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"i": i},
            kinds=("P",),
            inps_maps=(("i",)),
            outs_maps=(("i",)),
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
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
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        reshaped = XTCTensor(inputs[0].numpy().reshape(self._shape))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert reshaped.type == expected_type, (
            f"output type mismatch expect: {reshaped.type} != {expected_type}"
        )
        return [reshaped]


class XTCOperTranspose(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        axes = attrs.get("axes", ())
        super().__init__("transpose", axes=axes)

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert inputs_types[0].shape is not None
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        if self.attrs.axes == ():
            out_shape = shape[::-1]
        else:
            assert len(self.attrs.axes) == len(shape)
            out_shape = tuple([shape[n] for n in self.attrs.axes])
        return [
            XTCTensorType(
                shape=out_shape,
                dtype=inputs_types[0].dtype,
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        axes = self.attrs.axes if self.attrs.axes != () else None
        transposed = XTCTensor(inputs[0].numpy().transpose(axes))
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert transposed.type == expected_type, (
            f"output type mismatch expect: {transposed.type} != {expected_type}"
        )
        return [transposed]

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        i = functools.reduce(operator.mul, inp_shape, 1)
        return self._get_operation(
            inps_types,
            outs_types,
            dims={"i": i},
            kinds=("P",),
            inps_maps=(("i",)),
            outs_maps=(("i",)),  # TODO: invalid
        )
