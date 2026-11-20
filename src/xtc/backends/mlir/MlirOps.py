#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, Type, TypeAlias, cast

from .MlirBackend import ELT_TYPES, ELT_SIZES

from xdsl.dialects import linalg, arith, builtin, memref, tensor, scf
from xdsl.dialects.builtin import (
    MemRefType,
    TensorType,
    IndexType,
    UnitAttr,
    StringAttr,
    AffineMapAttr,
)
from xdsl.ir import Block, BlockArgument, Region
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.irdl import irdl_op_definition
from xdsl.builder import ImplicitBuilder

from xtc.itf.graph import Operation
from xtc.graphs.xtc.data import XTCTensorType


__all__ = [
    "MlirOperation",
    "MlirOperator",
    "MlirOperators",
]

OpAttrs: TypeAlias = dict[str, Any]


def _is_natural_layout(layout: list[int] | None, ndim: int) -> bool:
    return layout is None or layout == list(range(ndim))


def _require_natural_layouts(xtc_op: Operation) -> None:
    for typ in xtc_op.inputs_types:
        if isinstance(typ, XTCTensorType) and not _is_natural_layout(
            typ.layout, typ.ndim
        ):
            raise NotImplementedError(
                "tensor layout is not yet implemented in MLIR backend"
            )


def _cast_type_to(
    op: Any, cur_dtype: str, cast_dtype: str | None, ret_ssa: bool = True
) -> Any:
    """Cast a scalar or tensor value from one float dtype to another."""
    if cast_dtype is None or cur_dtype == cast_dtype:
        return op
    assert cur_dtype in ELT_SIZES
    assert cast_dtype in ELT_SIZES
    cur_size = ELT_SIZES[cur_dtype]
    cast_size = ELT_SIZES[cast_dtype]
    if isinstance(op.type, TensorType):
        target: Any = TensorType(ELT_TYPES[cast_dtype], op.type.get_shape())
    else:
        assert not isinstance(op.type, MemRefType), (
            "cannot arith cast a memref buffer, accumulate in a scratch "
            "and cast back to the out dtype with a linalg.generic"
        )
        target = ELT_TYPES[cast_dtype]
    if cur_size < cast_size:
        func: Any = arith.ExtFOp(op, target)
    elif cur_size > cast_size:
        func = arith.TruncFOp(op, target)
    else:
        # only conversion between types of the same size, bf16 <-> f16
        # TODO: if there is a arith.convertf use that
        inter: Any = ELT_TYPES["float32"]
        if isinstance(target, TensorType):
            inter = TensorType(inter, target.get_shape())
        ext = arith.ExtFOp(op, inter)
        func = arith.TruncFOp(ext.results[0], target)
    return func.results[0] if ret_ssa else func


def _cast_generic_to(
    op: Any,
    target: Any,
    cur_dtype: str,
    cast_dtype: str,
    result_types: tuple[Any, ...],
) -> Any:
    """Emit the linalg.generic casting the cur dtype of op back to the cast
    dtype in target. A linalg.generic so the cast can be scheduled like
    the fill, arith casts cannot be tiled."""
    block_in = Block(arg_types=[ELT_TYPES[cur_dtype], ELT_TYPES[cast_dtype]])
    with ImplicitBuilder(block_in):
        value = _cast_type_to(block_in.args[0], cur_dtype, cast_dtype)
        linalg.YieldOp(value)
    rank = len(op.type.get_shape())
    map = AffineMapAttr(AffineMap.identity(rank))
    return linalg.GenericOp(
        inputs=(op,),
        outputs=(target,),
        body=Region([block_in]),  # type: ignore # mypy issue with dataclass
        # ignore typing due to xdsl hints limitation
        indexing_maps=[map, map],
        iterator_types=[StringAttr("parallel")] * rank,
        result_types=result_types,
    )


class MlirOperation:
    def __init__(
        self,
        operator: Type["MlirOperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
        in_dtypes: Sequence[str] = (),
        out_dtype: str | None = None,
    ) -> None:
        self.operator = operator(
            args,
            attrs,
            name=name,
            op_type=op_type,
            in_dtypes=in_dtypes,
            out_dtype=out_dtype,
        )
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
    def from_operation(
        cls,
        xtc_op: Operation,
        name: str | None,
        op_type: Type[MemRefType] | Type[TensorType],
    ) -> "MlirOperation":
        dims = xtc_op.dims.values()
        in_dtypes = tuple(str(inp.dtype) for inp in xtc_op.inputs_types)
        out_dtype = xtc_op.outputs_types[0].dtype
        attrs = xtc_op.attrs
        op_cls = MlirOperators.from_name(xtc_op.name)
        args = op_cls.args_from_operation(xtc_op, tuple([*dims]))
        return MlirOperation(
            op_cls,
            args,
            dict(attrs),
            name=name,
            op_type=op_type,
            in_dtypes=in_dtypes,
            out_dtype=out_dtype,
        )


class MlirOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(
        self,
        args: tuple[Any, ...],
        attrs: dict[str, Any],
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
        in_dtypes: Sequence[str] = (),
        out_dtype: str | None = None,
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME
        self.op_type = op_type
        self._in_dtypes = tuple(in_dtypes)
        self._out_dtype = out_dtype

    @property
    def in_dtypes(self) -> tuple[str, ...]:
        return self._in_dtypes

    @property
    def out_dtype(self) -> str:
        assert self._out_dtype is not None, "no out dtype specified"
        return self._out_dtype

    def _require_uniform_dtypes(self) -> None:
        dtypes = (*self._in_dtypes, self._out_dtype)
        if any(dtype != self._in_dtypes[0] for dtype in dtypes[1:]):
            raise NotImplementedError(
                f"mixed type computation is not yet implemented in the MLIR backend: {dtypes}"
            )

    def has_cast_stage(self) -> bool:
        """Whether generate_op emits a separate cast stage (mixed dtypes)."""
        acc = self.attrs.get("acc_dtype")
        return any((acc or dtype) != self.out_dtype for dtype in self.in_dtypes)

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
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]: ...

    def inputs_types(self) -> tuple[str, ...]:
        return self._in_dtypes

    def outputs_types(self) -> tuple[str, ...]:
        return (self.out_dtype,)

    def _dims(self, kind: str = "") -> tuple[str, ...]:
        if kind == "":
            return tuple(self.AXES)
        return tuple([a for a, k in zip(self.AXES, self.KINDS) if k == kind])

    @classmethod
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        _require_natural_layouts(xtc_op)
        return args


class MlirOperatorMatmul(MlirOperator):
    DEFAULT_NAME = "matmul"
    AXES = "ijk"
    KINDS = "PPR"

    @classmethod
    @override
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        transpose_a = xtc_op.inputs_types[0].layout == [1, 0]
        transpose_b = xtc_op.inputs_types[1].layout == [1, 0]
        return (*args, transpose_a, transpose_b)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, j, k, *_ = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Ki, Kj, Kk, transpose_a, transpose_b = self.args
        acc_dtype = self.attrs.get("acc_dtype", None)
        mixed = acc_dtype is not None and (
            acc_dtype != self.out_dtype
            or acc_dtype != self.in_dtypes[0]
            or acc_dtype != self.in_dtypes[1]
        )
        acc = acc_dtype if mixed else self.out_dtype
        assert acc is not None
        elt_types = [
            ELT_TYPES[self.in_dtypes[0]],
            ELT_TYPES[self.in_dtypes[1]],
            ELT_TYPES[self.out_dtype],
        ]
        acc_elt = ELT_TYPES[acc]
        using_tensors = self.op_type == TensorType
        convert_output = mixed and acc != self.out_dtype
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for elt_type, shape in zip(elt_types, [[Ki, Kk], [Kk, Kj], [Ki, Kj]])
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)

        with ImplicitBuilder(block):
            result: tuple[Any, ...] = (args[2].type,) if using_tensors else ()
            fill_output: Any = args[2]
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, acc_elt))
            acc_result: tuple[Any, ...]
            if convert_output and using_tensors:
                fill_output = tensor.EmptyOp(
                    dynamic_sizes=[],
                    tensor_type=TensorType(acc_elt, [Ki, Kj]),
                ).results[0]
                acc_result = (TensorType(acc_elt, [Ki, Kj]),)
            elif convert_output and not using_tensors:
                fill_output = memref.AllocaOp.get(
                    return_type=acc_elt, shape=[Ki, Kj], alignment=256
                ).results[0]
                acc_result = result
            else:
                acc_result = result
            fill = linalg.FillOp(
                res=acc_result,
                inputs=(cst0.results[0],),
                outputs=(fill_output,),
            )
            acc_value = fill.results[0] if using_tensors else fill_output
            if not mixed and not transpose_a and not transpose_b:
                reduce: Any = linalg.MatmulOp(
                    res=acc_result, inputs=(args[0], args[1]), outputs=(acc_value,)
                )
            else:
                block_in = Block(arg_types=[*elt_types[:2], acc_elt])
                with ImplicitBuilder(block_in):
                    a = _cast_type_to(block_in.args[0], self.in_dtypes[0], acc)
                    b = _cast_type_to(block_in.args[1], self.in_dtypes[1], acc)
                    mul = arith.MulfOp(a, b)
                    add = arith.AddfOp(block_in.args[2], mul)
                    linalg.YieldOp(add)
                reduce = linalg.GenericOp(
                    inputs=(args[0], args[1]),
                    outputs=(acc_value,),
                    body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                    # ignore typing due to xdsl hints limitation
                    indexing_maps=[
                        AffineMapAttr(
                            AffineMap.from_callable(
                                lambda i, j, k: (k, i) if transpose_a else (i, k)
                            )
                        ),
                        AffineMapAttr(
                            AffineMap.from_callable(
                                lambda i, j, k: (j, k) if transpose_b else (k, j)
                            )
                        ),
                        AffineMapAttr(AffineMap.from_callable(lambda i, j, k: (i, j))),
                    ],
                    iterator_types=[
                        StringAttr("parallel"),
                        StringAttr("parallel"),
                        StringAttr("reduction"),
                    ],
                    result_types=acc_result,
                )

            output = None
            if convert_output:
                output = _cast_generic_to(
                    reduce.results[0] if using_tensors else acc_value,
                    args[2],
                    acc,
                    self.out_dtype,
                    result,
                )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        output_node_id = f"{self.name}_cast"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        if output is not None:
            output.attributes[f"__xtc_id_{output_node_id}_"] = UnitAttr()
            output_nodes = [output]
        else:
            output_nodes = [reduce]

        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
                **({output_node_id: output} if output is not None else {}),
            },
            "dims_sizes": [
                {"i": Ki, "j": Kj},
                self.dims_sizes(),
                *([{"i": Ki, "j": Kj}] if output is not None else []),
            ],
            "output_nodes": output_nodes,
            "root_node": reduce_node_id,
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k, *_ = self.args
        return (i, k), (k, j)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j = self.args[:2]
        return ((i, j),)


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

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        b, h, w, f, r, s, c = self.args
        return {"b": b, "h": h, "w": w, "f": f, "r": r, "s": s, "c": c}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc = self.args
        SH, SW = self.attrs.get("stride", self.DEFAULT_STRIDE)
        inps_dims = self.inputs_dims()
        out_dims = self.outputs_dims()[0]
        acc_dtype = self.attrs.get("acc_dtype", None)
        mixed = acc_dtype is not None and (
            acc_dtype != self.out_dtype
            or acc_dtype != self.in_dtypes[0]
            or acc_dtype != self.in_dtypes[1]
        )
        acc = acc_dtype if mixed else self.out_dtype
        assert acc is not None
        elt_types = [
            ELT_TYPES[self.in_dtypes[0]],
            ELT_TYPES[self.in_dtypes[1]],
            ELT_TYPES[self.out_dtype],
        ]
        acc_elt = ELT_TYPES[acc]
        using_tensors = self.op_type == TensorType
        convert_output = mixed and acc != self.out_dtype
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for elt_type, shape in zip(elt_types, [*inps_dims, out_dims])
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        with ImplicitBuilder(block):
            result: tuple[Any, ...] = (args[2].type,) if using_tensors else ()
            fill_output: Any = args[2]
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, acc_elt))
            acc_result: tuple[Any, ...]
            if convert_output and using_tensors:
                fill_output = tensor.EmptyOp(
                    dynamic_sizes=[],
                    tensor_type=TensorType(acc_elt, out_dims),
                ).results[0]
                acc_result = (TensorType(acc_elt, out_dims),)
            elif convert_output and not using_tensors:
                fill_output = memref.AllocaOp.get(
                    return_type=acc_elt, shape=out_dims, alignment=256
                ).results[0]
                acc_result = result
            else:
                acc_result = result
            fill = linalg.FillOp(
                res=acc_result,
                inputs=(cst0.results[0],),
                outputs=(fill_output,),
            )
            acc_value = fill.results[0] if using_tensors else fill_output
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
            flags = arith.FastMathFlagsAttr("fast")
            block_in = Block(arg_types=[*elt_types[:2], acc_elt])
            with ImplicitBuilder(block_in):
                a = _cast_type_to(block_in.args[0], self.in_dtypes[0], acc)
                b = _cast_type_to(block_in.args[1], self.in_dtypes[1], acc)
                mul = arith.MulfOp(a, b, flags=flags)
                add = arith.AddfOp(block_in.args[2], mul, flags=flags)
                linalg.YieldOp(add)
            reduce = linalg.GenericOp(
                inputs=(args[0], args[1]),
                outputs=(acc_value,),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                # ignore typing due to xdsl hints limitation
                indexing_maps=[
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (b, h * SH + r, w * SW + s, c)
                            )
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (r, s, c, f)
                            )
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (b, h, w, f)
                            )
                        )
                    ),
                ],
                iterator_types=iterator_types,
                result_types=acc_result,
            )
            output = None
            if convert_output:
                output = _cast_generic_to(
                    reduce.results[0] if using_tensors else acc_value,
                    args[2],
                    acc,
                    self.out_dtype,
                    result,
                )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        output_node_id = f"{self.name}_cast"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        if output is not None:
            output.attributes[f"__xtc_id_{output_node_id}_"] = UnitAttr()
            output_nodes = [output]
        else:
            output_nodes = [reduce]
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
                **({output_node_id: output} if output is not None else {}),
            },
            "dims_sizes": [
                {"b": Kb, "h": Kh, "w": Kw, "f": Kf},
                self.dims_sizes(),
                *([{"b": Kb, "h": Kh, "w": Kw, "f": Kf}] if output is not None else []),
            ],
            "output_nodes": output_nodes,
            "root_node": reduce_node_id,
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f, r, s, c = self.args
        SH, SW = self.attrs.get("stride", self.DEFAULT_STRIDE)
        return ((b, h * SH + r - 1, w * SW + s - 1, c), (r, s, c, f))

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f = self.args[:4]
        return ((b, h, w, f),)


class MlirOperatorRelu(MlirOperator):
    # One parallel dim per input dim
    DEFAULT_NAME = "relu"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    def _shape(self) -> tuple[int, ...]:
        shape = tuple(self.args)
        assert len(shape) <= len(self.AXES), f"unsupported relu rank: {shape}"
        return shape

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)[: len(self._shape())]

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.AXES, self._shape()))

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        self._require_uniform_dtypes()
        shape = self._shape()
        elt_type = ELT_TYPES[self.in_dtypes[0]]
        if block is None:
            ops_types = [self.op_type(elt_type, shape) for _ in range(2)]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        rank = len(shape)
        with ImplicitBuilder(block):
            result = (args[1].type,) if self.op_type == TensorType else ()
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_type))
            block_in = Block(arg_types=[elt_type, elt_type, elt_type])
            with ImplicitBuilder(block_in):
                max = arith.MaximumfOp(block_in.args[0], block_in.args[1])
                linalg.YieldOp(max)
            relu = linalg.GenericOp(
                inputs=(args[0], cst0.results[0]),
                outputs=(args[1],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=[
                    AffineMapAttr(AffineMap.identity(rank)),  # input
                    AffineMapAttr(AffineMap(rank, 0, ())),  # scalar
                    AffineMapAttr(AffineMap.identity(rank)),  # output
                ],
                iterator_types=[StringAttr("parallel")] * rank,
                result_types=result,
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
            "output_nodes": [relu],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self._shape(),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self._shape(),)


def _broadcast_map(inp_shape: Sequence[int], out_shape: Sequence[int]) -> AffineMap:
    # Align shapes on trailing dims, broadcast unit dims are indexed by 0
    offset = len(out_shape) - len(inp_shape)
    results = [
        AffineConstantExpr(0)
        if size == 1 and out_shape[offset + pos] != 1
        else AffineDimExpr(offset + pos)
        for pos, size in enumerate(inp_shape)
    ]
    return AffineMap(len(out_shape), 0, tuple(results))


class MlirOperatorElementwiseBinary(MlirOperator):
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"
    BODY_OP: Any = None

    @classmethod
    @override
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        _require_natural_layouts(xtc_op)
        inps_shapes = tuple(
            cast(XTCTensorType, typ).constant_shape for typ in xtc_op.inputs_types
        )
        out_shape = cast(XTCTensorType, xtc_op.outputs_types[0]).constant_shape
        return (*args, inps_shapes, out_shape)

    def _rank(self) -> int:
        rank = len(self.args[-1])
        assert rank <= len(self.AXES), f"unsupported {self.name} rank: {rank}"
        return rank

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)[: self._rank()]

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.AXES, self.args[: self._rank()]))

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        dtype, inps_shapes, out_shape = self.args[-3:]
        elt_types = [
            ELT_TYPES[self.in_dtypes[0]],
            ELT_TYPES[self.in_dtypes[1]],
            ELT_TYPES[self.out_dtype],
        ]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for shape in [*inps_shapes, out_shape]
                for elt_type in elt_types
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        flags = arith.FastMathFlagsAttr("fast")
        rank = len(out_shape)
        with ImplicitBuilder(block):
            result = (args[2].type,) if self.op_type == TensorType else ()
            block_in = Block(arg_types=elt_types)
            with ImplicitBuilder(block_in):
                res = self.BODY_OP(block_in.args[0], block_in.args[1], flags=flags)
                linalg.YieldOp(res)
            generic = linalg.GenericOp(
                inputs=(args[0], args[1]),
                outputs=(args[2],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=[
                    AffineMapAttr(_broadcast_map(inps_shapes[0], out_shape)),
                    AffineMapAttr(_broadcast_map(inps_shapes[1], out_shape)),
                    AffineMapAttr(AffineMap.identity(rank)),
                ],
                iterator_types=[StringAttr("parallel")] * rank,
                result_types=result,
            )
        generic_node_id = f"{self.name}"
        generic.attributes[f"__xtc_id_{generic_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                generic_node_id: generic,
            },
            "dims_sizes": [
                self.dims_sizes(),
            ],
            "output_nodes": [generic],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return self.args[-2]

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-3]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self.args[-1],)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-3]
        return (dtype,)


class MlirOperatorAdd(MlirOperatorElementwiseBinary):
    DEFAULT_NAME = "add"
    BODY_OP = arith.AddfOp


class MlirOperatorMul(MlirOperatorElementwiseBinary):
    DEFAULT_NAME = "mul"
    BODY_OP = arith.MulfOp


class MlirOperatorPad(MlirOperator):
    DEFAULT_NAME = "pad"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        assert len(self.args) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args)}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        self._require_uniform_dtypes()
        elt_type = ELT_TYPES[self.in_dtypes[0]]
        dims_value = list(self.args)
        padding = self.attrs["padding"]
        constant_value = self.attrs["constant_value"]
        lows = [0] * len(dims_value)
        highs = [0] * len(dims_value)
        if isinstance(padding, dict):
            dims_value_before_pad = list(dims_value)
            for i, pad_value in padding.items():
                dims_value_before_pad[i] -= sum(pad_value)
                lows[i] = pad_value[0]
                highs[i] = pad_value[1]
        else:
            dims_value_before_pad = [
                dim_value - sum(padding) for dim_value in dims_value
            ]
            lows = [padding[0] for d in dims_value]
            highs = [padding[1] for d in dims_value]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for shape in [dims_value_before_pad, dims_value]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args]
            for i, (pad_b, pad_a) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args]
        sizes = list(dims_value_before_pad)
        strides = [1 for _ in self.args]
        using_tensors = self.op_type == TensorType
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(constant_value, elt_type))
            result = (args[1].type,) if using_tensors else ()
            fill_node_id = f"{self.name}_0"

            if using_tensors:
                fill = None
                empty = args[1]
                block_in = Block(arg_types=[elt_type])
                rank = len(dims_value)
                # pad written as a linalg.generic to enable producer fusion
                with ImplicitBuilder(block_in):
                    # gets the current iteration index for each dim (not constants)
                    output_indices = [linalg.IndexOp(i) for i in range(rank)]
                    input_indices = []
                    in_bounds_checks = []

                    zero = arith.ConstantOp.create(
                        properties={"value": builtin.IntegerAttr(0, IndexType())},
                        result_types=[IndexType()],
                    )

                    for dim_idx in range(rank):
                        lo_const = arith.ConstantOp.create(
                            properties={
                                "value": builtin.IntegerAttr(lows[dim_idx], IndexType())
                            },
                            result_types=[IndexType()],
                        )
                        input_idx = arith.SubiOp(output_indices[dim_idx], lo_const)
                        input_indices.append(input_idx)

                        # check to see if in the padding region or input tensor region
                        size_const = arith.ConstantOp.create(
                            properties={
                                "value": builtin.IntegerAttr(
                                    dims_value_before_pad[dim_idx], IndexType()
                                )
                            },
                            result_types=[IndexType()],
                        )

                        ge_zero = arith.CmpiOp(input_idx, zero, "sge")
                        lt_size = arith.CmpiOp(input_idx, size_const, "slt")

                        in_bounds_checks.append(ge_zero)
                        in_bounds_checks.append(lt_size)

                    all_in_bounds: arith.CmpiOp | arith.AndIOp = in_bounds_checks[0]
                    for check in in_bounds_checks[1:]:
                        all_in_bounds = arith.AndIOp(all_in_bounds, check)

                    if_region_then = Region([Block(arg_types=[])])
                    if_region_else = Region([Block(arg_types=[])])

                    with ImplicitBuilder(if_region_then.blocks[0]):
                        extracted = tensor.ExtractOp(
                            tensor=args[0],
                            indices=input_indices,  # type: ignore
                            result_type=elt_type,
                        )
                        scf.YieldOp(extracted)
                    with ImplicitBuilder(if_region_else.blocks[0]):
                        scf.YieldOp(cst0)

                    if_op = scf.IfOp(
                        cond=all_in_bounds,
                        true_region=if_region_then,
                        false_region=if_region_else,
                        return_types=[elt_type],
                    )

                    linalg.YieldOp(if_op.results[0])

                copy = linalg.GenericOp(
                    inputs=[],
                    outputs=[empty],
                    body=Region([block_in]),
                    indexing_maps=[AffineMapAttr(AffineMap.identity(rank))],
                    iterator_types=[StringAttr("parallel")] * rank,
                    result_types=[TensorType(elt_type, dims_value)],
                )
            else:
                fill = linalg.FillOp(
                    res=result,
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
                copy = linalg.CopyOp(  # type: ignore
                    inputs=[args[0]],
                    outputs=[subview.result],
                    res=result,
                )
                fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                **({fill_node_id: fill} if fill else {}),
                copy_node_id: copy,
            },
            "dims_sizes": [
                self.dims_sizes(),
                *([] if using_tensors else [self.dims_sizes()]),
            ],
            "output_nodes": [copy],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        dims_value = list(self.args)
        if isinstance(padding, dict):
            for i, pad_value in padding.items():
                dims_value[i] -= sum(pad_value)
        else:
            dims_value = [value - sum(padding) for value in dims_value]
        return (tuple(dims_value),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args),)


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
        assert len(self.args) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args)}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        self._require_uniform_dtypes()
        elt_type = ELT_TYPES[self.in_dtypes[0]]
        dims_values = list(self.args)
        padding = self.attrs["padding"]
        if isinstance(padding, dict):
            dims_values_before_unpad = list(dims_values)
            for i, pad_value in padding.items():
                dims_values_before_unpad[i] += sum(pad_value)
        else:
            dims_values_before_unpad = [
                dim_value + sum(padding) for dim_value in dims_values
            ]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for shape in [dims_values_before_unpad, dims_values]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args]
            for i, (pad_b, _) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args]
        sizes = dims_values
        strides = [1 for _ in self.args]
        using_tensors = self.op_type == TensorType
        with ImplicitBuilder(block):
            if using_tensors:
                copy = tensor.ExtractSliceOp.from_static_parameters(
                    source=args[0],
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                )
            else:
                subview = memref.SubviewOp.from_static_parameters(
                    source=args[0],
                    source_type=args[0].type,  # type: ignore
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                )
                copy = linalg.CopyOp(  # type: ignore
                    inputs=[subview.result],
                    outputs=[args[1]],
                    res=(),
                )
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                copy_node_id: None if using_tensors else copy,
            },
            "dims_sizes": [*([] if using_tensors else [self.dims_sizes()])],
            "output_nodes": [copy],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        inp_dims = list(self.args)
        if isinstance(padding, dict):
            for axis, (pad_b, pad_a) in padding.items():
                inp_dims[axis] += pad_b + pad_a
        else:
            inp_dims = [inp_dim + sum(padding) for inp_dim in inp_dims]
        return (tuple(inp_dims),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self.args,)


class MlirOperators:
    @classmethod
    def from_name(cls, name: str) -> Type[MlirOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = MlirOperatorMatmul
    conv2d = MlirOperatorConv2D
    relu = MlirOperatorRelu
    add = MlirOperatorAdd
    mul = MlirOperatorMul
    pad2d = MlirOperatorPad2D
    unpad = MlirOperatorUnpad
    pad = MlirOperatorPad
