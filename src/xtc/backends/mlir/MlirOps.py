#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, Type, TypeAlias, cast

from xdsl.dialects import linalg, arith, builtin, memref
from xdsl.dialects.builtin import (
    MemRefType,
    f32,
    f64,
    i64,
    UnitAttr,
    StringAttr,
    AffineMapAttr,
)
from xdsl.ir import Block, BlockArgument, Region
from xdsl.ir.affine import AffineMap
from xdsl.irdl import irdl_op_definition
from xdsl.builder import ImplicitBuilder

from xtc.itf.graph import Operation
from xtc.utils.math import mulall


__all__ = [
    "MlirOperation",
    "MlirOperator",
    "MlirOperators",
]

OpAttrs: TypeAlias = dict[str, Any]


class MlirOperation:
    def __init__(
        self,
        operator: Type["MlirOperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
    ) -> None:
        self.operator = operator(args, attrs, name=name)
        self.args = args
        self.attrs = attrs
        self.name = self.operator.name if name is None else name

    def generate(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        return self.operator.generate_op(block, args)

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

    @classmethod
    def from_operation(cls, xtc_op: Operation, name: str | None) -> "MlirOperation":
        dims = xtc_op.dims.values()
        dtype = xtc_op.inputs_types[0].dtype  # TODO: currently get dtype from 1st arg
        args = tuple([*dims, dtype])
        attrs = xtc_op.attrs
        return MlirOperation(
            MlirOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
        )


class MlirOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

    @abstractmethod
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]: ...
    @abstractmethod
    def dims(self, kind: str = "") -> tuple[str, ...]: ...
    @abstractmethod
    def dims_sizes(self) -> dict[str, int]: ...
    @abstractmethod
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def inputs_types(self) -> tuple[str, ...]: ...
    @abstractmethod
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def outputs_types(self) -> tuple[str, ...]: ...

    def _dims(self, kind: str = "") -> tuple[str, ...]:
        if kind == "":
            return tuple(self.AXES)
        return tuple([a for a, k in zip(self.AXES, self.KINDS) if k == kind])


class MlirOperatorMatmul(MlirOperator):
    DEFAULT_NAME = "matmul"
    AXES = "ijk"
    KINDS = "PPR"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, j, k, _ = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Ki, Kj, Kk, dtype = self.args
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [
                MemRefType(elt_type, shape) for shape in [[Ki, Kk], [Kk, Kj], [Ki, Kj]]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            fill = linalg.FillOp(
                res=(),
                inputs=(cst0.results[0],),
                outputs=(args[2],),
            )
            reduce = linalg.MatmulOp(
                res=(),
                inputs=(args[0], args[1]),
                outputs=(args[2],),
            )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"i": Ki, "j": Kj},
                self.dims_sizes(),
            ],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k, _ = self.args
        return (i, k), (k, j)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j = self.args[:2]
        return ((i, j),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


@irdl_op_definition
class Conv2DNhwcHwFcOp(linalg.ConvOperation):
    """
    Performs 2-D convolution with inputs (N, H, W, C) (R, S, C F)

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nhwc_hwcf-linalgconv2dnhwchwcfop
    """

    name = "linalg.conv_2d_nhwc_hwcf"


class MlirOperatorConv2D(MlirOperator):
    DEFAULT_NAME = "conv2d"
    AXES = "bhwfrsc"
    KINDS = "PPPPRRR"

    DEFAULT_STRIDE = (1, 1)

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        attrs = {"stride": self.DEFAULT_STRIDE, **attrs}
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        b, h, w, f, r, s, c, _ = self.args
        return {"b": b, "h": h, "w": w, "f": f, "r": r, "s": s, "c": c}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc, dtype = self.args
        SH, SW = self.attrs["stride"]
        inps_dims = self.inputs_dims()
        out_dims = self.outputs_dims()[0]
        dtype = self.args[-1]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [
                MemRefType(elt_type, shape) for shape in [*inps_dims, out_dims]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            fill = linalg.FillOp(
                res=(),
                inputs=(cst0.results[0],),
                outputs=(args[2],),
            )
            # TODO: Does not work
            # strides = DenseIntOrFPElementsAttr.vector_from_list([SH, SW], i64)
            # dilations = DenseIntOrFPElementsAttr.vector_from_list([1, 1], i64)
            # reduce = Conv2DNhwcHwFcOp(
            #     inputs=(block.args[0], block.args[1]),
            #     outputs=(block.args[2],),
            #     dilations=dilations,
            #     strides=strides,
            # )
            iterator_types = [
                StringAttr({"P": "parallel", "R": "reduction"}[k]) for k in self.KINDS
            ]
            block_in = Block(arg_types=[f32, f32, f32])
            with ImplicitBuilder(block_in):
                mul = arith.MulfOp(block_in.args[0], block_in.args[1])
                add = arith.AddfOp(block_in.args[2], mul)
                linalg.YieldOp(add)
            reduce = linalg.GenericOp(
                inputs=(args[0], args[1]),
                outputs=(args[2],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                # ignore typing due to xdsl hints limitation
                indexing_maps=[
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c:  # type: ignore
                            (b, h * SH + r, w * SW + s, c)
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c:  # type: ignore
                            (r, s, c, f)
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c:  # type: ignore
                            (b, h, w, f)
                        )
                    ),
                ],
                iterator_types=iterator_types,
            )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"b": Kb, "h": Kh, "w": Kw, "f": Kf},
                self.dims_sizes(),
            ],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f, r, s, c, _ = self.args
        SH, SW = self.attrs["stride"]
        return ((b, h * SH + r - 1, w * SW + s - 1, c), (r, s, c, f))

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f = self.args[:4]
        return ((b, h, w, f),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorRelu(MlirOperator):
    DEFAULT_NAME = "relu"
    AXES = "i"
    KINDS = "P"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, _ = self.args
        return {"i": i}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Ki, dtype = self.args
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [MemRefType(elt_type, shape) for shape in [[Ki], [Ki]]]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        inp_shape, out_shape = [
            list(cast(MemRefType, arg.type).get_shape()) for arg in args
        ]
        inp_size, out_size = [mulall(shape) for shape in [inp_shape, out_shape]]
        assert inp_size == out_size
        with ImplicitBuilder(block):
            inp_reassociation = builtin.ArrayAttr(
                [
                    builtin.ArrayAttr(
                        [builtin.IntegerAttr(x, i64) for x in range(len(inp_shape))]
                    )
                ]
            )
            out_reassociation = builtin.ArrayAttr(
                [
                    builtin.ArrayAttr(
                        [builtin.IntegerAttr(x, i64) for x in range(len(out_shape))]
                    )
                ]
            )
            inp = memref.CollapseShapeOp(
                operands=[args[0]],
                properties=dict(reassociation=inp_reassociation),
                result_types=[MemRefType(elt_type, (inp_size,))],
            )
            out = memref.CollapseShapeOp(
                operands=[args[1]],
                properties=dict(reassociation=out_reassociation),
                result_types=[MemRefType(elt_type, (out_size,))],
            )
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            iterator_types = [
                StringAttr({"P": "parallel", "R": "reduction"}[k]) for k in self.KINDS
            ]
            block_in = Block(arg_types=[f32, f32, f32])
            with ImplicitBuilder(block_in):
                max = arith.MaximumfOp(block_in.args[0], block_in.args[1])
                linalg.YieldOp(max)
            relu = linalg.GenericOp(
                inputs=(inp.results[0], cst0.results[0]),
                outputs=(out.results[0],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                # ignore typing due to xdsl hints limitation
                indexing_maps=[
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda i:  # type: ignore
                            (i,)
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda _:  # type: ignore
                            ()
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda i:  # type: ignore
                            (i,)
                        )
                    ),
                ],
                iterator_types=iterator_types,
            )
        relu_node_id = f"{self.name}"
        relu.attributes[f"__xtc_id_{relu_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                relu_node_id: relu,
            },
            "dims_sizes": [
                self.dims_sizes(),
            ],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i = self.args[0]
        return ((i,),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i = self.args[0]
        return ((i,),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorPad(MlirOperator):
    DEFAULT_NAME = "pad"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        assert len(self.args[:-1]) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        dtype = self.args[-1]
        dims_value = list(self.args[:-1])
        padding = self.attrs["padding"]
        constant_value = self.attrs["constant_value"]
        if isinstance(padding, dict):
            dims_value_before_pad = list(dims_value)
            for i, pad_value in padding.items():
                dims_value_before_pad[i] -= sum(pad_value)
        else:
            dims_value_before_pad = [
                dim_value - sum(padding) for dim_value in dims_value
            ]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [
                MemRefType(elt_type, shape)
                for shape in [dims_value_before_pad, dims_value]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args[:-1]]
            for i, (pad_b, pad_a) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args[:-1]]
        sizes = list(dims_value_before_pad)
        strides = [1 for _ in self.args[:-1]]
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(constant_value, elt_size))
            fill = linalg.FillOp(
                res=(),
                inputs=(cst0.results[0],),
                outputs=(args[1],),
            )
            subview = memref.SubviewOp.from_static_parameters(
                source=args[1],
                source_type=args[1].type,  # type: ignore
                offsets=offsets,
                sizes=sizes,
                strides=strides,
            )
            copy = linalg.CopyOp(
                inputs=[args[0]],
                outputs=[subview.result],
                res=(),
            )
        fill_node_id = f"{self.name}_0"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                copy_node_id: copy,
            },
            "dims_sizes": [
                self.dims_sizes(),
                self.dims_sizes(),
            ],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        dims_value = list(self.args[:-1])
        if isinstance(padding, dict):
            for i, pad_value in padding.items():
                dims_value[i] -= sum(pad_value)
        else:
            dims_value = [value - sum(padding) for value in dims_value]
        return (tuple(dims_value),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorPad2D(MlirOperatorPad):
    DEFAULT_NAME = "pad2d"
    AXES = "bhwc"
    KINDS = "PPPP"


class MlirOperatorUnpad(MlirOperator):
    DEFAULT_NAME = "unpad"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        assert len(self.args[:-1]) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        dtype = self.args[-1]
        dims_values = list(self.args[:-1])
        padding = self.attrs["padding"]
        if isinstance(padding, dict):
            dims_values_before_unpad = list(dims_values)
            for i, pad_value in padding.items():
                dims_values_before_unpad[i] += sum(pad_value)
        else:
            dims_values_before_unpad = [
                dim_value + sum(padding) for dim_value in dims_values
            ]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        if block is None:
            ops_types = [
                MemRefType(elt_type, shape)
                for shape in [dims_values_before_unpad, dims_values]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, MemRefType) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args[:-1]]
            for i, (pad_b, _) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args[:-1]]
        sizes = dims_values
        strides = [1 for _ in self.args[:-1]]
        with ImplicitBuilder(block):
            subview = memref.SubviewOp.from_static_parameters(
                source=args[0],
                source_type=args[0].type,  # type: ignore
                offsets=offsets,
                sizes=sizes,
                strides=strides,
            )
            copy = linalg.CopyOp(
                inputs=[subview.result],
                outputs=[args[1]],
                res=(),
            )
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                copy_node_id: copy,
            },
            "dims_sizes": [
                self.dims_sizes(),
            ],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        inp_dims = list(self.args[:-1])
        if isinstance(padding, dict):
            for axis, (pad_b, pad_a) in padding.items():
                inp_dims[axis] += pad_b + pad_a
        else:
            inp_dims = [inp_dim + sum(padding) for inp_dim in inp_dims]
        return (tuple(inp_dims),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self.args[:-1],)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperators:
    @classmethod
    def from_name(cls, name: str) -> Type[MlirOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = MlirOperatorMatmul
    conv2d = MlirOperatorConv2D
    relu = MlirOperatorRelu
    pad2d = MlirOperatorPad2D
    unpad = MlirOperatorUnpad
    pad = MlirOperatorPad
