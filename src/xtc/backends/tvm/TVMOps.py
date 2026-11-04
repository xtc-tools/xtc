#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, Type, TypeAlias, cast

from xtc.utils.math import mulall
from xtc.utils.text import to_cname

from xtc.itf.graph import Operation, Graph, Node
from xtc.graphs.xtc.data import XTCTensorType

__all__ = [
    "TVMBaseExpr",
    "TVMOperation",
    "TVMGraph",
    "TVMOperator",
    "TVMOperators",
]

import tvm
import tvm.te as te
import tvm.topi as topi

TETensor: TypeAlias = te.Tensor
TEIndexVar: TypeAlias = Any


class TVMBaseExpr(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def np_outputs_spec(self) -> list[dict[str, Any]]: ...
    @abstractmethod
    def np_inputs_spec(self) -> list[dict[str, Any]]: ...

    @classmethod
    def from_operation(cls, xtc_op: Operation, name: str | None) -> "TVMOperation":
        for typ in xtc_op.inputs_types:
            if isinstance(typ, XTCTensorType):
                if typ.layout is not None:
                    assert False, "tensor layout is not yet implemented in TVM backend"
        dims = xtc_op.dims.values()
        in_dtypes = tuple(str(inp.dtype) for inp in xtc_op.inputs_types)
        out_dtype = xtc_op.outputs_types[0].dtype
        args = tuple(dims)
        attrs = dict(xtc_op.attrs)
        attrs.update(
            dict(inp_shape=xtc_op.inputs_types[0].shape)
        )  # TODO: work around to pass shape
        return TVMOperation(
            TVMOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
            in_dtypes=in_dtypes,
            out_dtype=out_dtype,
        )

    @classmethod
    def from_graph(cls, xtc_graph: Graph) -> "TVMGraph":
        return TVMGraph(xtc_graph)


class TVMOperation(TVMBaseExpr):
    def __init__(
        self,
        operator: Type["TVMOperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
        in_dtypes: Sequence[str] = (),
        out_dtype: str | None = None,
    ) -> None:
        self.args = args
        self.attrs = attrs
        self.operator = operator(
            args, attrs, name=name, in_dtypes=in_dtypes, out_dtype=out_dtype
        )
        super().__init__(
            name=self.operator.name if name is None else name,
        )

    @override
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

    @override
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


class TVMGraph(TVMBaseExpr):
    def __init__(self, graph: Graph):
        super().__init__(name=graph.name)
        self._graph = graph
        self._operations = {
            node.name: TVMOperation.from_operation(node.operation, node.name)
            for node in self._graph.nodes.values()
        }
        self._variables: dict[str, TETensor] = {}
        self._params: dict[str, TETensor] = {}

    def _te_shape_dtype_from_node(self, node: Node) -> Any:
        if node.outputs_types is not None:
            type = node.outputs_types[0]
        else:
            assert hasattr(node, "_expr")
            type = node._expr.type  # type: ignore
        assert isinstance(type, XTCTensorType)
        if type.layout is not None:
            assert False, "tensor layout is not yet implemented in TVM backend"
        return type.shape, type.dtype

    def _te_tensor_from_node(self, node: Node) -> Any:
        shape, dtype = self._te_shape_dtype_from_node(node)
        return te.placeholder(
            shape,
            name=to_cname(node.name),
            dtype=dtype,
        )

    def _te_op_from_op(
        self,
        inputs: list[str],
        outputs: list[str],
        operation: TVMOperation,
        variables: dict[str, Any],
    ) -> tuple[TETensor, ...]:
        assert len(outputs) == 1
        in_outs = operation.operator.generate_op(
            [variables[to_cname(name)] for name in inputs]
        )
        variables[to_cname(outputs[0])] = in_outs[-1]
        return in_outs

    def _te_expr_from_graph(self) -> tuple[dict[str, TETensor], dict[str, TETensor]]:
        inputs = {
            te.name: te
            for te in [
                self._te_tensor_from_node(node) for node in self._graph.inputs_nodes
            ]
        }
        variables = {**inputs}
        for node, operation in zip(
            self._graph.nodes.values(), self._operations.values()
        ):
            self._te_op_from_op(node.inputs, node.outputs, operation, variables)
        params = {
            name: variables[name]
            for name in [
                to_cname(name) for name in [*self._graph.inputs, *self._graph.outputs]
            ]
        }
        return variables, params

    @override
    def np_inputs_spec(self) -> list[dict[str, Any]]:
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in [
                self._te_shape_dtype_from_node(node)
                for node in self._graph.inputs_nodes
            ]
        ]
        return inputs_spec

    @override
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in [
                self._te_shape_dtype_from_node(node)
                for node in self._graph.outputs_nodes
            ]
        ]
        return outputs_spec


class TVMOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(
        self,
        args: tuple[Any, ...],
        attrs: dict[str, Any],
        name: str | None = None,
        in_dtypes: Sequence[str] = (),
        out_dtype: str | None = None,
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME
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
                f"mixed type computation is not yet implemented in the TVM backend: {dtypes}"
            )

    def has_cast_stage(self) -> bool:
        """Whether generate_op emits a separate cast stage (mixed dtypes)."""
        acc = self.attrs.get("acc_dtype")
        return any((acc or dtype) != self.out_dtype for dtype in self.in_dtypes)

    @abstractmethod
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]: ...
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


class TVMOperatorMatmul(TVMOperator):
    DEFAULT_NAME = "matmul"
    AXES = "ijk"
    KINDS = "PPR"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, j, k = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        Ki, Kj, Kk = self.args
        if inputs is None:
            A = te.placeholder((Ki, Kk), name="A", dtype=self.in_dtypes[0])
            B = te.placeholder((Kk, Kj), name="B", dtype=self.in_dtypes[1])
        else:
            A, B = cast(Sequence[Any], inputs)
        acc_type_A = self.attrs.get("acc_dtype", self.in_dtypes[0])
        acc_type_B = self.attrs.get("acc_dtype", self.in_dtypes[1])
        Ashape = tuple(A.shape)
        Bshape = tuple(B.shape)
        Anewshape = (Ashape[0], mulall(list(Ashape[1:])))
        Bnewshape = (mulall(list(Bshape[:-1])), Bshape[-1])
        if Ashape != Anewshape:
            A = topi.reshape(A, newshape=Anewshape)
        if Bshape != Bnewshape:
            B = topi.reshape(B, newshape=Bnewshape)
        k = te.reduce_axis((0, Kk), "k")
        O = te.compute(
            (Ki, Kj),
            lambda i, j: cast(
                Any,
                te.sum(
                    A[i, k].astype(acc_type_A) * B[k, j].astype(acc_type_B),
                    axis=k,
                ),
            ),
            name=self.name,
        )
        if self.has_cast_stage():
            # Cannot use topi.cast. Need to have name for casting
            # to be schedule with the gpu if not crash with bf16
            acc = cast(Any, O)
            O = te.compute(
                (Ki, Kj),
                lambda i, j: acc[i, j].astype(self.out_dtype),
                name=f"{self.name}_cast",
            )
        return cast(tuple[TETensor], (A, B, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k = self.args
        return (i, k), (k, j)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j = self.args[:2]
        return ((i, j),)


class TVMOperatorRelu(TVMOperator):
    DEFAULT_NAME = "relu"
    DEFAULT_THRESHOLD = 0
    AXES = "ijklmnopqrstuvwxyz"

    def _input_shape(self) -> tuple[int, ...]:
        shape = self.attrs.get("inp_shape", self.args)
        return tuple(shape)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        rank = len(self._input_shape())
        assert rank <= len(self.AXES), f"unsupported relu input rank: {rank}"
        axes = tuple(self.AXES[:rank])
        if kind == "":
            return axes
        if kind == "P":
            return axes
        return ()

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.dims(), self._input_shape()))

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        self._require_uniform_dtypes()
        dtype = self.in_dtypes[0]
        shape = self._input_shape()
        if inputs is None:
            A = te.placeholder(shape, name="A", dtype=dtype)
        else:
            (A,) = cast(Sequence[Any], inputs)
        O = te.compute(
            shape,
            lambda *dims: tvm.tirx.max(
                self.attrs.get("threshold", self.DEFAULT_THRESHOLD), A[dims]
            ),
            name=self.name,
            varargs_names=list(self.dims()),
        )
        return cast(tuple[TETensor], (A, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self._input_shape(),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self._input_shape(),)


class TVMOperatorConv2D(TVMOperator):
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
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc = self.args
        inps_dims = self.inputs_dims()
        if inputs is None:
            A = te.placeholder(inps_dims[0], name="A", dtype=self.in_dtypes[0])
            W = te.placeholder(inps_dims[1], name="W", dtype=self.in_dtypes[1])
        else:
            A, W = cast(Sequence[Any], inputs)
        acc_type_A = self.attrs.get("acc_dtype", self.in_dtypes[0])
        acc_type_W = self.attrs.get("acc_dtype", self.in_dtypes[1])
        r = te.reduce_axis((0, Kr), "r")
        s = te.reduce_axis((0, Ks), "s")
        c = te.reduce_axis((0, Kc), "c")
        out_dims = self.outputs_dims()[0]
        Ksh, Ksw = self.attrs.get("stride", self.DEFAULT_STRIDE)
        O = te.compute(
            out_dims,
            lambda b, h, w, f: cast(
                Any,
                te.sum(
                    A[b, h * Ksh + r, w * Ksw + s, c].astype(acc_type_A)
                    * W[r, s, c, f].astype(acc_type_W),
                    axis=(r, s, c),
                ),
            ),
            name=self.name,
        )
        if self.has_cast_stage():
            # Cannot use topi.cast. Need to have name for casting
            # to be schedule with the gpu if not crash with bf16
            acc = cast(Any, O)
            O = te.compute(
                out_dims,
                lambda b, h, w, f: acc[b, h, w, f].astype(self.out_dtype),
                name=f"{self.name}_cast",
            )
        return cast(tuple[TETensor], (A, W, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f, r, s, c = self.args
        sh, sw = self.attrs.get("stride", self.DEFAULT_STRIDE)
        return ((b, h * sh + r - 1, w * sw + s - 1, c), (r, s, c, f))

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f = self.args[:4]
        return ((b, h, w, f),)


class TVMOperatorPad(TVMOperator):
    DEFAULT_NAME = "pad"
    AXES = "ijklmnopqrstuvwxyz"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        rank = len(self.args)
        assert rank <= len(self.AXES), f"unsupported pad input rank: {rank}"
        axes = tuple(self.AXES[:rank])
        if kind == "":
            return axes
        if kind == "P":
            return axes
        return ()

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.dims(), self.args))

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        self._require_uniform_dtypes()
        dtype = self.in_dtypes[0]
        dims_values = tuple(self.dims_sizes().values())
        padding = self.attrs["padding"]
        constant_value = self.attrs["constant_value"]
        if isinstance(padding, dict):
            dims_values_all = list(dims_values)
            for i, pad_value in padding.items():
                dims_values_all[i] -= sum(pad_value)
        else:
            dims_values_all = [dim_value - sum(padding) for dim_value in dims_values]

        if inputs is None:
            A = te.placeholder(tuple(dims_values_all), name="A", dtype=dtype)
        else:
            (A,) = cast(Sequence[Any], inputs)

        def get_indexes(*args: int) -> tuple[int, ...]:
            indexes = list(args)
            if isinstance(padding, dict):
                for i, (pad_b, _) in padding.items():
                    indexes[i] -= pad_b
            else:
                indexes = [i - padding[0] for i in indexes]
            return tuple(indexes)

        def get_args_bounds(*args: int) -> list[Any]:
            indexes = list(args)
            if isinstance(padding, dict):
                return [
                    tvm.tirx.all(
                        indexes[i] - pad_b >= 0,
                        indexes[i] - pad_b < dims_values_all[i],
                    )
                    for i, (pad_b, _) in padding.items()
                ]
            else:
                return [
                    tvm.tirx.all(
                        index - padding[0] >= 0,
                        index - padding[0] < dims_values_all[i],
                    )
                    for i, index in enumerate(indexes)
                ]

        O = te.compute(
            tuple(dims_values),
            lambda *dims: tvm.tirx.if_then_else(
                tvm.tirx.all(*get_args_bounds(*dims)),
                A[get_indexes(*dims)],
                constant_value,
            ),
            name=self.name,
            varargs_names=list(self.dims()),
        )
        return cast(tuple[TETensor], (A, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        inp_dims = list(self.dims_sizes().values())
        if isinstance(padding, dict):
            for i, pad_value in padding.items():
                inp_dims[i] -= sum(pad_value)
        else:
            inp_dims = [i_dim - sum(padding) for i_dim in inp_dims]
        return (tuple(inp_dims),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.dims_sizes().values()),)


class TVMOperatorPad2D(TVMOperatorPad):
    DEFAULT_NAME = "pad2d"
    AXES = "bhwc"
    KINDS = "PPPP"


class TVMOperatorUnpad(TVMOperator):
    DEFAULT_NAME = "unpad"
    AXES = "ijklmnopqrstuvwxyz"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        rank = len(self.args)
        assert rank <= len(self.AXES), f"unsupported unpad input rank: {rank}"
        axes = tuple(self.AXES[:rank])
        if kind == "":
            return axes
        if kind == "P":
            return axes
        return ()

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.dims(), self.args))

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        self._require_uniform_dtypes()
        dtype = self.in_dtypes[0]
        dims_values = tuple(self.dims_sizes().values())
        padding = self.attrs["padding"]
        if inputs is None:
            dims_values_before_unpad = list(dims_values)
            if isinstance(padding, dict):
                for axis, pad in padding.items():
                    dims_values_before_unpad[axis] += sum(pad)
            else:
                dims_values_before_unpad = [
                    dims_value + sum(padding) for dims_value in dims_values
                ]
            A = te.placeholder(tuple(dims_values_before_unpad), name="A", dtype=dtype)
        else:
            (A,) = cast(Sequence[Any], inputs)

        def get_indexes(*args: int) -> tuple[int, ...]:
            indexes = list(args)
            if isinstance(padding, dict):
                for axis, (pad_b, _) in padding.items():
                    indexes[axis] += pad_b
            else:
                indexes = [index + padding[0] for index in indexes]
            return tuple(indexes)

        O = te.compute(
            tuple(dims_values),
            lambda *dims: A[get_indexes(*dims)],
            name=self.name,
            varargs_names=list(self.dims()),
        )
        return cast(tuple[TETensor], (A, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        inp_dims = list(self.dims_sizes().values())
        if isinstance(padding, dict):
            for axis, pad in padding.items():
                inp_dims[axis] += sum(pad)
        else:
            inp_dims = [i_dim + sum(padding) for i_dim in inp_dims]
        return (tuple(inp_dims),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.dims_sizes().values()),)


class TVMOperatorTranspose(TVMOperator):
    DEFAULT_NAME = "transpose"
    AXES = "i"
    KINDS = "P"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i = self.args[0]
        return {"i": i}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        i = self.args[0]
        self._require_uniform_dtypes()
        if inputs is None:
            A = te.placeholder(
                self.attrs["inp_shape"], name="A", dtype=self.in_dtypes[0]
            )
        else:
            (A,) = cast(Sequence[Any], inputs)
        shape = A.shape
        axes = self.attrs["axes"]
        if axes == ():
            axes = list(range(len(A.shape)))[::-1]
        assert len(axes) == len(shape)
        out_shape = tuple([shape[n] for n in axes])
        in_axes = {o: i for i, o in enumerate(axes)}

        def transpose_i(*ivs: TEIndexVar):
            ii = ivs[0]
            out_indices = []
            div = 1
            for size in out_shape[::-1]:
                out_indices.append(ii // div % size)
                div *= size
            out_indices = out_indices[::-1]
            indices = [out_indices[in_axes[axis]] for axis in range(len(shape))]
            return A(*indices)

        O = te.compute((i,), transpose_i, name=self.name)
        if out_shape != (i,):
            O = topi.reshape(O, newshape=out_shape)
        return cast(tuple[TETensor], (A, O))

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        shape = self.attrs["inp_shape"]
        return (shape,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        shape = self.attrs["inp_shape"]
        axes = self.attrs["axes"]
        if axes == ():
            axes = list(range(len(shape)))[::-1]
        out_shape = tuple([shape[n] for n in axes])
        return (out_shape,)


class TVMOperators:
    @classmethod
    def from_name(cls, name: str) -> Type[TVMOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = TVMOperatorMatmul
    relu = TVMOperatorRelu
    conv2d = TVMOperatorConv2D
    pad2d = TVMOperatorPad2D
    unpad = TVMOperatorUnpad
    pad = TVMOperatorPad
    transpose = TVMOperatorTranspose
