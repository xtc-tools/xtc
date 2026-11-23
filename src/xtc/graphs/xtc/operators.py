#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Callable, TypeAlias, cast, Any
from types import SimpleNamespace as NS
from collections.abc import Sequence, Mapping
import functools
import operator
import string
import numpy as np
import hashlib

from xtc.itf.data.tensor import DataType
from xtc.utils.math import mulall
from xtc.itf.operator import Operator
from xtc.itf.data import Tensor, TensorType
from xtc.itf.runtime.accelerator import AcceleratorDevice

from .data import XTCTensor, XTCTensorType
from .operation import XTCOperation

__all__ = [
    "XTCOperator",
]


XTCOperatorAttr: TypeAlias = Any
XTCOperatorAttrs: TypeAlias = NS

# Positional names for operators with one dim per tensor dim: i, j, k, ...
_AXES = string.ascii_lowercase[8:]
XTCOperPaddingAttr: TypeAlias = (
    int | tuple[int] | tuple[int, int] | tuple[int, int, int, int]
)
XTCOperStrideAttr: TypeAlias = int | tuple[int] | tuple[int, int]


class XTCOperator(Operator):
    def __init__(
        self,
        name: str,
        device: AcceleratorDevice | None = None,
        **attrs: XTCOperatorAttr,
    ) -> None:
        self._name = name
        self._device = device
        self._attrs = NS(**attrs)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    def attrs(self) -> XTCOperatorAttrs:
        return self._attrs

    @property
    def device(self) -> AcceleratorDevice | None:
        return self._device

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        if self.device is None:
            return [cast(XTCTensorType, inp_type) for inp_type in inputs_types]
        res_types = []
        for inp_type in inputs_types:
            inp_tensor_type = cast(XTCTensorType, inp_type)
            res_types.append(
                XTCTensorType(
                    shape=inp_tensor_type.shape,
                    dtype=inp_tensor_type.dtype,
                    device=inp_tensor_type.device,
                )
            )
        return res_types

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return [cast(XTCTensor, inp) for inp in inputs]

    def _forward_helper(
        self,
        inputs: Sequence[Tensor],
        ref_func: Callable,
        acc_dtype: DataType,
        out_dtype: DataType,
    ) -> Sequence[XTCTensor]:
        assert len(inputs) != 0
        np_input = [inp.numpy() for inp in inputs]
        res_type = str(np_input[0].dtype)
        if acc_dtype is not None:
            np_input = [inp.astype(acc_dtype) for inp in np_input]
        elif out_dtype is not None:
            np_input = [inp.astype(out_dtype) for inp in np_input]
        result = ref_func(*np_input)
        if out_dtype is not None:
            res_type = out_dtype
        if isinstance(result, (tuple, list)) and out_dtype is not None:
            return [XTCTensor(res.astype(res_type)) for res in result]
        result = np.asarray(result)
        return [XTCTensor(result.astype(res_type))]

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

    @override
    def to_dict(self) -> dict[str, Any]:
        def listify(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: listify(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return [listify(v) for v in obj]
            else:
                return obj

        op_dict: dict[str, Any] = {"name": self._name}
        op_dict["attrs"] = {k: listify(v) for k, v in self._attrs.__dict__.items()}
        return op_dict

    @classmethod
    def version_string(cls) -> str:
        names = sorted(f"{c.__module__}.{c.__qualname__}" for c in cls.__subclasses__())
        data = "|".join(names).encode()
        return hashlib.sha256(data).hexdigest()[:16]


class XTCOperTensor(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("tensor", **attrs)

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
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("matmul", **attrs)
        self._acc_dtype = attrs.get("acc_dtype", None)
        self._out_dtype = attrs.get("out_dtype", None)
        assert isinstance(self._acc_dtype, DataType)
        assert isinstance(self._out_dtype, DataType)

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp0_shape = inps_types[0].constant_shape
        inp1_shape = inps_types[1].constant_shape
        i, k = (inp0_shape[0], mulall(list(inp0_shape[1:])))
        bk, j = (mulall(list(inp1_shape[:-1])), inp1_shape[-1])
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
        assert len(inputs_types[0].shape) >= 2
        assert len(inputs_types[1].shape) >= 2
        inp0_shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        inp1_shape = cast(XTCTensorType, inputs_types[1]).constant_shape
        i, k = (inp0_shape[0], mulall(list(inp0_shape[1:])))
        bk, j = (mulall(list(inp1_shape[:-1])), inp1_shape[-1])
        assert k == bk, (
            f"incompatible dimension k for matmul inputs shapes: ({i}, {k}) ({bk}, {j})"
        )
        if self._out_dtype is not None:
            out_type = self._out_dtype
        else:
            out_type = inputs_types[0].dtype
        return [
            XTCTensorType(shape=(i, j), dtype=out_type, device=self.device),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        def compute_matmul(A: np.ndarray, B: np.ndarray):
            A = A.reshape((A.shape[0], -1))
            B = B.reshape((-1, B.shape[-1]))
            # Note, use np.dot instead of np.matmul which may be buggy on Mac accelerators
            return np.dot(A, B)

        matmul = self._forward_helper(
            inputs, compute_matmul, self._acc_dtype, self._out_dtype
        )[0]
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
        # One parallel dim per input dim, named i, j, k, ... as in the backends
        inp_shape = inps_types[0].constant_shape
        assert len(inp_shape) <= len(_AXES), f"unsupported relu rank: {inp_shape}"
        dims_names = tuple(_AXES[: len(inp_shape)])
        return self._get_operation(
            inps_types,
            outs_types,
            dims=dict(zip(dims_names, inp_shape)),
            kinds=tuple("P" for _ in dims_names),
            inps_maps=(dims_names,),
            outs_maps=(dims_names,),
        )

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        relu = XTCTensor(np.maximum(inputs[0].numpy(), self._threshold))
        return [relu]


class _OperElementwiseBinaryImpl:
    @staticmethod
    def get_operation_variable(
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> tuple[
        dict[str, int],
        tuple[str, ...],
        tuple[tuple[str, ...], ...],
        tuple[tuple[str, ...], ...],
    ]:
        # One parallel dim per output dim. Inputs are aligned on trailing
        # dims (numpy broadcasting), broadcast unit dims are accessed at "0".
        out_shape = outs_types[0].constant_shape
        dims_names = [f"d{n}" for n in range(len(out_shape))]

        def inp_map(inp_shape: Sequence[int]) -> tuple[str, ...]:
            offset = len(out_shape) - len(inp_shape)
            return tuple(
                "0"
                if size == 1 and out_shape[offset + pos] != 1
                else dims_names[offset + pos]
                for pos, size in enumerate(inp_shape)
            )

        return (
            dict(zip(dims_names, out_shape)),
            tuple("P" for _ in out_shape),
            tuple(inp_map(inp.constant_shape) for inp in inps_types),
            (tuple(dims_names),),
        )

    @staticmethod
    def forward_types(
        name: str,
        inputs_types: Sequence[TensorType],
        device: AcceleratorDevice | None,
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == 2
        inp0, inp1 = [cast(XTCTensorType, inp_type) for inp_type in inputs_types]
        assert inp0.dtype == inp1.dtype, (
            f"incompatible dtypes for {name} inputs: {inp0.dtype} {inp1.dtype}"
        )
        try:
            shape = np.broadcast_shapes(inp0.constant_shape, inp1.constant_shape)
        except ValueError:
            raise ValueError(
                f"incompatible shapes for {name} inputs: "
                f"{inp0.constant_shape} {inp1.constant_shape}"
            ) from None
        return [XTCTensorType(shape=shape, dtype=inp0.dtype, device=device)]


class XTCOperAdd(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("add", **attrs)

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        dims, kinds, inps_maps, outs_maps = (
            _OperElementwiseBinaryImpl.get_operation_variable(inps_types, outs_types)
        )
        return self._get_operation(
            inps_types,
            outs_types,
            dims=dims,
            kinds=kinds,
            inps_maps=inps_maps,
            outs_maps=outs_maps,
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        return _OperElementwiseBinaryImpl.forward_types(
            self.name, inputs_types, self.device
        )

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return [XTCTensor(np.add(inputs[0].numpy(), inputs[1].numpy()))]


class XTCOperMul(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        super().__init__("mul", **attrs)

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        dims, kinds, inps_maps, outs_maps = (
            _OperElementwiseBinaryImpl.get_operation_variable(inps_types, outs_types)
        )
        return self._get_operation(
            inps_types,
            outs_types,
            dims=dims,
            kinds=kinds,
            inps_maps=inps_maps,
            outs_maps=outs_maps,
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        return _OperElementwiseBinaryImpl.forward_types(
            self.name, inputs_types, self.device
        )

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return [XTCTensor(np.multiply(inputs[0].numpy(), inputs[1].numpy()))]


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
        self._acc_dtype = attrs.get("acc_dtype", None)
        self._out_dtype = attrs.get("out_dtype", None)
        assert isinstance(self._acc_dtype, DataType)
        assert isinstance(self._out_dtype, DataType)

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
        if self._out_dtype is not None:
            out_type = self._out_dtype
        else:
            out_type = inputs_types[0].dtype
        return [
            XTCTensorType(
                shape=tuple([*inputs_types[0].shape[:-3], oh, ow, f]),
                dtype=out_type,
                device=self.device,
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        # Note that the input is supposed to be already padded
        def compute_conv2d(inp: np.ndarray, weight: np.ndarray):
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
            return out.reshape(out_shape)

        conv = self._forward_helper(
            inputs, compute_conv2d, self._acc_dtype, self._out_dtype
        )[0]
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert conv.type == expected_type, (
            f"output type mismatch expect: {conv.type} != {expected_type}"
        )
        return [conv]


class _OperPadImpl:
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        padding = attrs.get("padding", 0)
        constant_value = attrs.get("constant_value", 0)
        device = attrs.get("device", None)
        self.padding = padding
        self.constant_value = constant_value
        self.device = device

    def get_operation_variable(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> tuple[
        dict[str, int], tuple[str, ...], tuple[tuple[str, ...]], tuple[tuple[str, ...]]
    ]:
        inp_shape = inps_types[0].constant_shape
        size_inp_shape = len(inp_shape)
        assert size_inp_shape >= 2
        padding = self.padding
        dims_names = [f"d{i}" for i in range(len(inp_shape))]
        dims_values = list(inp_shape)
        dims_inp = dims_names[:]
        if isinstance(padding, dict):
            for axis, (val1, val2) in padding.items():
                dims_values[axis] += val1 + val2
                dims_inp[axis] += f"-{val1}"
        else:
            pad = sum(padding)
            dims_values = [value + pad for value in dims_values]
            dims_inp = [inp + f"-{padding[0]}" for inp in dims_inp]
        return (
            {name: value for name, value in zip(dims_names, dims_values)},
            tuple("P" for _ in inp_shape),
            (tuple(dims_inp),),
            (tuple(dims_names),),
        )

    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == 1
        assert inputs_types[0].shape is not None
        size_input_type_0 = len(inputs_types[0].shape)
        assert size_input_type_0 >= 2
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        padding = self.padding
        dims_types = list(shape)
        if isinstance(padding, dict):
            for axis, pad in padding.items():
                assert -size_input_type_0 <= axis and axis < size_input_type_0, (
                    f"axis: {axis} is out of bound should be between {-size_input_type_0} and {size_input_type_0 - 1}"
                )
                dims_types[axis] += sum(pad)
        else:
            pad = sum(padding)
            dims_types = [value + pad for value in dims_types]
        return [
            XTCTensorType(
                shape=tuple(dims_types), dtype=inputs_types[0].dtype, device=self.device
            ),
        ]

    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        shape = cast(XTCTensorType, inputs[0].type).constant_shape
        padding = self.padding
        data_input = inputs[0].numpy()
        constant_value = self.constant_value
        if isinstance(padding, dict):
            pads = [(0, 0) for _ in range(len(shape))]
            for axis, pad in padding.items():
                pads[axis] = pad
        else:
            pads = [padding for _ in range(len(shape))]
        padded = XTCTensor(
            data=np.pad(
                data_input, pads, mode="constant", constant_values=constant_value
            )
        )
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert padded.type == expected_type, (
            f"output type mismatch expect: {padded.type} != {expected_type}"
        )
        return [padded]


class XTCOperPad(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        padding = attrs.get("padding", 0)
        constant_value = attrs.get("constant_value", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        elif (
            isinstance(padding, (tuple, list))
            and all(isinstance(pad, int) for pad in padding)
            and len(padding) == 2
        ):
            pass
        elif isinstance(padding, (tuple, list)):
            len_pad = len(padding)
            padding = {
                -(len_pad - i): (value, value) if isinstance(value, int) else value
                for i, value in enumerate(padding)
            }
        else:
            assert isinstance(padding, dict), (
                "padding is either a list of tuple/int or int or a dict"
            )
            for key, value in padding.items():
                if isinstance(value, (int, float)):
                    padding[key] = (value, value)
        assert isinstance(constant_value, (int, float)), (
            f"constant_value need to be a number"
        )
        if isinstance(padding, dict):
            padding = {k: v for k, v in padding.items() if v != (0, 0)}
        super().__init__(
            "pad",
            padding=padding,
            constant_value=constant_value,
            device=attrs.get("device", None),
        )
        self.impl = _OperPadImpl(
            padding=padding, constant_value=constant_value, device=self.device
        )

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        dims, kinds, inps_maps, outs_maps = self.impl.get_operation_variable(
            inps_types, outs_types
        )
        return self._get_operation(
            inps_types,
            outs_types,
            dims=dims,
            kinds=kinds,
            inps_maps=inps_maps,
            outs_maps=outs_maps,
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        return self.impl.forward_types(inputs_types)

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return self.impl.forward(inputs)


class XTCOperPad2D(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        padding = attrs.get("padding", (0, 0, 0, 0))
        constant_value = attrs.get("constant_value", 0)
        # Axes -3 and -2 are [..., a3, a2, .]
        axes = attrs.get("axes", (-3, -2))
        if isinstance(padding, int):
            padding = {axes[0]: (padding, padding), axes[1]: (padding, padding)}
        else:
            assert isinstance(padding, (tuple, dict)), (
                f"padding for pad2d of wrong type, expect int or tuple or dict: {padding}"
            )
            if not isinstance(padding, dict) and len(padding) == 1:
                padding = {
                    axes[0]: (padding[0], padding[0]),
                    axes[1]: (padding[0], padding[0]),
                }
            elif isinstance(padding, dict) and len(padding) == 2:
                padding = {
                    i: (pad, pad) if isinstance(pad, int) else pad
                    for i, pad in padding.items()
                }
                pass
            elif all(isinstance(pad, int) for pad in padding) and len(padding) == 2:
                padding = {
                    axes[0]: (padding[0], padding[1]),
                    axes[1]: (padding[0], padding[1]),
                }
            elif (
                all(isinstance(pad, tuple) and len(pad) == 2 for pad in padding)
                and len(padding) == 2
            ):
                padding = {
                    axes[0]: padding[0],
                    axes[1]: padding[1],
                }
            else:
                assert len(padding) == 4, (
                    f"padding for pad2d of wrong size, expected 1, 2 or 4: {padding}"
                )
                padding = {
                    axes[0]: (padding[0], padding[1]),
                    axes[1]: (padding[2], padding[3]),
                }
        padding = {k: v for k, v in padding.items() if v != (0, 0)}
        assert len(axes) == 2, f"axes for pad2d of wrong size, expected 2: {axes}"
        assert axes[0] != axes[1], f"axes need 2 different dimension to pad: {axes}"
        assert isinstance(constant_value, (int, float)), (
            f"constant_value need to be a number"
        )
        super().__init__(
            "pad2d",
            padding=padding,
            constant_value=constant_value,
            device=attrs.get("device", None),
        )
        self.impl = _OperPadImpl(
            padding=padding, constant_value=constant_value, device=self.device
        )

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        dims, kinds, inps_maps, outs_maps = self.impl.get_operation_variable(
            inps_types, outs_types
        )
        return self._get_operation(
            inps_types,
            outs_types,
            dims=dims,
            kinds=kinds,
            inps_maps=inps_maps,
            outs_maps=outs_maps,
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        return self.impl.forward_types(inputs_types)

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        return self.impl.forward(inputs)


class XTCOperUnpad(XTCOperator):
    def __init__(self, **attrs: XTCOperatorAttr) -> None:
        padding = attrs.get("padding", 0)
        constant_value = attrs.get("constant_value", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        elif (
            isinstance(padding, (tuple, list))
            and all(isinstance(pad, int) for pad in padding)
            and len(padding) == 2
        ):
            pass
        elif isinstance(padding, (tuple, list)):
            len_pad = len(padding)
            padding = {
                -(len_pad - i): (value, value) if isinstance(value, int) else value
                for i, value in enumerate(padding)
            }
        else:
            assert isinstance(padding, dict), (
                "padding is either a list of tuple/int or int or a dict"
            )
            for key, value in padding.items():
                if isinstance(value, (int, float)):
                    padding[key] = (value, value)
        assert isinstance(constant_value, (int, float)), (
            f"constant_value need to be a number"
        )
        if isinstance(padding, dict):
            padding = {k: v for k, v in padding.items() if v != (0, 0)}
        super().__init__("unpad", padding=padding)

    @override
    def get_operation(
        self,
        inps_types: Sequence[XTCTensorType],
        outs_types: Sequence[XTCTensorType],
    ) -> XTCOperation:
        inp_shape = inps_types[0].constant_shape
        size_inp_shape = len(inp_shape)
        assert size_inp_shape >= 2
        padding = self.attrs.padding
        dims_names = [f"d{i}" for i in range(len(inp_shape))]
        dims_values = list(inp_shape)
        dims_inp = dims_names[:]
        if isinstance(padding, dict):
            for axis, (val1, val2) in padding.items():
                dims_values[axis] -= val1 + val2
                dims_inp[axis] += f"+{val1}"
        else:
            pad = sum(padding)
            dims_values = [value - pad for value in dims_values]
            dims_inp = [inp + f"+{padding[0]}" for inp in dims_inp]
        return self._get_operation(
            inps_types,
            outs_types,
            dims={name: value for name, value in zip(dims_names, dims_values)},
            kinds=tuple("P" for _ in inp_shape),
            inps_maps=(tuple(dims_inp),),
            outs_maps=(tuple(dims_names),),
        )

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == 1
        assert inputs_types[0].shape is not None
        size_input_type_0 = len(inputs_types[0].shape)
        assert size_input_type_0 >= 2
        shape = cast(XTCTensorType, inputs_types[0]).constant_shape
        padding = self.attrs.padding
        dims_types = list(shape)
        if isinstance(padding, dict):
            for axis, pad in padding.items():
                assert -size_input_type_0 <= axis and axis < size_input_type_0, (
                    f"axis: {axis} is out of bound should be between {-size_input_type_0} and {size_input_type_0 - 1}"
                )
                dims_types[axis] -= sum(pad)
        else:
            pad = sum(padding)
            dims_types = [value - pad for value in dims_types]
        return [
            XTCTensorType(
                shape=tuple(dims_types), dtype=inputs_types[0].dtype, device=self.device
            ),
        ]

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        padding = self.attrs.padding
        data_input = inputs[0].numpy()
        data_shape = data_input.shape
        if isinstance(padding, dict):
            slices = [slice(None)] * data_input.ndim
            for axis, pad in padding.items():
                slices[axis] = slice(pad[0], data_shape[axis] - pad[1])
        else:
            slices = [
                slice(padding[0], data_shape[i] - padding[1])
                for i in range(len(data_shape))
            ]

        unpadded = XTCTensor(data=data_input[tuple(slices)])
        expected_type = self.forward_types([inp.type for inp in inputs])[0]
        assert unpadded.type == expected_type, (
            f"output type mismatch expect: {unpadded.type} != {expected_type}"
        )
        return [unpadded]


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
                shape=out_shape, dtype=inputs_types[0].dtype, device=self.device
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
                shape=out_shape, dtype=inputs_types[0].dtype, device=self.device
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
