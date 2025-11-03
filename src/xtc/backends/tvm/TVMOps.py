#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, Type, TypeAlias
import tarfile
import shutil

from xtc.utils.math import mulall
from xtc.utils.host_tools import target_triple

from xtc.itf.graph import Operation, Graph, Node

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

TETensor: TypeAlias = Any  # Use instead of te.Tensor to avoids type errors
TEIndexVar: TypeAlias = Any


def get_tvm_native_target_options() -> str:
    """
    Returm the tvm target options to pass to llvm.
    """
    from cpuinfo import get_cpu_info

    info = get_cpu_info()
    arch = info["arch_string_raw"]
    flags = info["flags"]
    triple = target_triple(arch)
    cpu, attrs = "", ""
    if arch == "x86_64":
        if "avx512f" in flags:
            cpu = "skylake-avx512"
        elif "avx2" in flags:
            cpu = "core-avx2"
    elif arch == "aarch64":
        if "asimd" in flags:
            cpu = "cortex-a72"
            attrs = "+neon"
    target_options = []
    if triple:
        target_options.append(f"-mtriple={triple}")
    if cpu:
        target_options.append(f"-mcpu={cpu}")
    if attrs:
        target_options.append(f"-mattrs={attrs}")
    return " ".join(target_options)


def tvm_build_crt_args(target: str) -> dict[str, Any]:
    # We use system-lib with crt runtime such that DSO loading works
    # The generated .so can then be used:
    # - for static compilation as soon as the tvm runtime is provided
    # - for dynamic loading from python
    # Recent version of tvm (i.e. 0.19) have a Runtime object
    # Older version (i.e. 0.16) support passing runtime options in target
    try:
        from tvm.relay.backend import Runtime

        runtime_kwargs = {
            "runtime": Runtime("crt", {"system-lib": True}),
        }
    except:
        runtime_kwargs = {}
        target = f"{target} --system-lib --runtime=c"
    return {
        "target": target,
        **runtime_kwargs,
    }


def tvm_build_crt(
    sch: Any, operation: Any, target: str, name: str | None = None
) -> Any:
    build_kwargs = tvm_build_crt_args(target)
    return tvm.build(sch, operation, name=name, **build_kwargs)


def tvm_emit_c(
    sch: te.Schedule, tensors: Sequence[TETensor], target: str, dir: str, name: str
) -> Any:
    target = "c -keys=arch -march=generic -mcpu=generic"
    build_kwargs = tvm_build_crt_args(target)
    config = {
        "tir.disable_vectorize": True,
    }
    with tvm.transform.PassContext(opt_level=3, config=config):
        built = tvm.build(
            sch,
            list(tensors),
            name=name,
            **build_kwargs,
        )
    tar_file = f"{dir}.tar"
    built.export_library(tar_file)
    shutil.rmtree(dir, ignore_errors=True)
    with tarfile.open(tar_file) as tf:
        tf.extractall(dir)


class TVMBaseExpr(ABC):
    def __init__(self, name: str, target: str | None = None) -> None:
        self.name = name
        if target is not None:
            self.target_options = ""
            self.target = target
        else:
            self.target_options = get_tvm_native_target_options()
            self.target = "llvm"
        self.tgt = tvm.target.Target(target=f"{self.target} {self.target_options}")
        self.dev = tvm.device(self.tgt.kind.name, 0)

    @abstractmethod
    def generate(self) -> tuple[TETensor, ...]: ...
    @abstractmethod
    def schedule(
        self, tensors: Sequence[TETensor], schedule: Any = None
    ) -> te.Schedule: ...
    @abstractmethod
    def np_outputs_spec(self) -> list[dict[str, Any]]: ...
    @abstractmethod
    def np_inputs_spec(self) -> list[dict[str, Any]]: ...

    def build(
        self,
        tensors: Sequence[TETensor],
        sch: te.Schedule,
        func_name: str | None = None,
    ) -> Any:
        if func_name is None:
            func_name = self.name
        return tvm_build_crt(
            sch,
            tensors,
            name=func_name,
            target=self.tgt,
        )

    def emit_c(
        self,
        tensors: Sequence[TETensor],
        sch: te.Schedule,
        dir: str,
        func_name: str | None = None,
    ) -> None:
        if func_name is None:
            func_name = self.name
        tvm_emit_c(sch, tensors, self.tgt, dir, func_name)

    def lower(self, tensors: Sequence[TETensor], sch: te.Schedule) -> str:
        return tvm.lower(sch, list(tensors), simple_mode=True)

    @classmethod
    def from_operation(cls, xtc_op: Operation, name: str | None) -> "TVMOperation":
        dims = xtc_op.dims.values()
        dtype = xtc_op.inputs_types[0].dtype  # TODO: infer dtype form first input
        args = tuple([*dims, dtype])
        attrs = dict(xtc_op.attrs)
        attrs.update(
            dict(inp_shape=xtc_op.inputs_types[0].shape)
        )  # TODO: work around to pass shape
        return TVMOperation(
            TVMOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
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
        target: str | None = None,
    ) -> None:
        self.args = args
        self.attrs = attrs
        self.operator = operator(args, attrs, name=name)
        super().__init__(
            name=self.operator.name if name is None else name, target=target
        )

    @override
    def generate(self) -> tuple[Any, ...]:
        return self.operator.generate_op()

    @override
    def schedule(
        self, tensors: Sequence[TETensor], schedule: Any = None
    ) -> te.Schedule:
        sch = te.create_schedule(tensors[-1].op)
        if schedule is None:
            return sch
        schedule_map = schedule.schedule_impl
        tensors_map = {t.name: t for t in tensors}
        for sched in schedule_map.values():
            if sched:
                exec(sched, {"sch": sch, "obj": tensors_map}, {})
        return sch

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
    def __init__(self, graph: Graph, target: str | None = None):
        super().__init__(name=graph.name, target=target)
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
        return type.shape, type.dtype

    def _te_tensor_from_node(self, node: Node) -> Any:
        shape, dtype = self._te_shape_dtype_from_node(node)
        return te.placeholder(
            shape,
            name=node.name,
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
        in_outs = operation.operator.generate_op([variables[name] for name in inputs])
        variables[outputs[0]] = in_outs[-1]
        return in_outs

    def _te_expr_from_graph(self) -> tuple[dict[str, TETensor], dict[str, TETensor]]:
        inputs = {
            node.name: self._te_tensor_from_node(node)
            for node in self._graph.inputs_nodes
        }
        variables = {**inputs}
        for node, operation in zip(
            self._graph.nodes.values(), self._operations.values()
        ):
            self._te_op_from_op(node.inputs, node.outputs, operation, variables)
        params = {
            name: variables[name]
            for name in [*self._graph.inputs, *self._graph.outputs]
        }
        return variables, params

    @override
    def generate(self) -> tuple[TETensor, ...]:
        self._variables, self._params = self._te_expr_from_graph()
        return tuple(self._params.values())

    @override
    def schedule(
        self, tensors: Sequence[TETensor], schedule: Any = None
    ) -> te.Schedule:
        sch = te.create_schedule(tensors[-1].op)
        if schedule is None:
            return sch
        schedule_map = schedule.schedule_impl
        tensors_map = {t.name: t for t in self._variables.values()}
        for sched in schedule_map.values():
            if sched:
                exec(sched, {"sch": sch, "obj": tensors_map}, {})
        return sch

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
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME

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
    def inputs_types(self) -> tuple[str, ...]: ...
    @abstractmethod
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def outputs_types(self) -> tuple[str, ...]: ...

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
        i, j, k, _ = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        Ki, Kj, Kk, dtype = self.args
        if inputs is None:
            A = te.placeholder((Ki, Kk), name="A", dtype=dtype)
            B = te.placeholder((Kk, Kj), name="B", dtype=dtype)
        else:
            A, B = inputs
        k = te.reduce_axis((0, Kk), "k")
        O = te.compute(
            (Ki, Kj),
            lambda i, j: (
                te.sum(
                    A[i, k] * B[k, j],
                    axis=k,
                )
            ),
            name=self.name,
        )
        return A, B, O

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


class TVMOperatorRelu(TVMOperator):
    DEFAULT_NAME = "relu"
    DEFAULT_THRESHOLD = 0
    AXES = "i"
    KINDS = "P"

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        attrs = {"threshold": self.DEFAULT_THRESHOLD, **attrs}
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, _ = self.args
        return {"i": i}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        Ki, dtype = self.args
        if inputs is None:
            A = te.placeholder((Ki,), name="A", dtype=dtype)
        else:
            (A,) = inputs
        shape = tuple(A.shape)
        size = mulall(A.shape)
        newshape = (size,)
        if shape != newshape:
            A = topi.reshape(A, newshape=(size,))
        O = te.compute(
            (Ki,),
            lambda i,: tvm.tir.max(self.attrs["threshold"], A[i]),
            name=self.name,
        )
        if shape != newshape:
            O = topi.reshape(O, newshape=shape)
        return A, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, _ = self.args
        return ((i,),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, _ = self.args
        return ((i,),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)


class TVMOperatorConv2D(TVMOperator):
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
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc, dtype = self.args
        inps_dims = self.inputs_dims()
        if inputs is None:
            A = te.placeholder(inps_dims[0], name="A", dtype=dtype)
            W = te.placeholder(inps_dims[1], name="W", dtype=dtype)
        else:
            A, W = inputs
        r = te.reduce_axis((0, Kr), "r")
        s = te.reduce_axis((0, Ks), "s")
        c = te.reduce_axis((0, Kc), "c")
        out_dims = self.outputs_dims()[0]
        Ksh, Ksw = self.attrs["stride"]
        O = te.compute(
            out_dims,
            lambda b, h, w, f: (
                te.sum(
                    A[b, h * Ksh + r, w * Ksw + s, c] * W[r, s, c, f],
                    axis=(r, s, c),
                )
            ),
            name=self.name,
        )
        return A, W, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f, r, s, c, _ = self.args
        sh, sw = self.attrs["stride"]
        return ((b, h * sh + r - 1, w * sw + s - 1, c), (r, s, c, f))

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


class TVMOperatorPad2D(TVMOperator):
    DEFAULT_NAME = "pad2d"
    AXES = "bhwc"
    KINDS = "PPPP"

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        attrs = {**attrs}
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        return {axis: size for axis, size in zip(["b", "h", "w", "c"], self.args[:3])}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        b, h, w, c, dtype = self.args
        hb, he, wb, we = self.attrs["padding"]
        ha, wa = h - hb - he, w - wb - we
        if inputs is None:
            A = te.placeholder((b, ha, wa, c), name="A", dtype=dtype)
        else:
            (A,) = inputs
        O = te.compute(
            (b, h, w, c),
            lambda bi, hi, wi, ci: (
                tvm.tir.if_then_else(
                    tvm.tir.all(
                        hi - hb >= 0,
                        hi - hb < ha,
                        wi - wb >= 0,
                        wi - wb < wa,
                    ),
                    A[bi, hi - hb, wi - wb, ci],
                    0,
                )
            ),
            name=self.name,
        )
        return A, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, c = self.args[:4]
        hb, he, wb, we = self.attrs["padding"]
        return ((b, h - hb - he, w - wb - we, c),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, c, _ = self.args
        return ((b, h, w, c),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)


class TVMOperatorTranspose(TVMOperator):
    DEFAULT_NAME = "transpose"
    AXES = "i"
    KINDS = "P"

    def __init__(
        self, args: tuple[Any, ...], attrs: dict[str, Any], name: str | None = None
    ) -> None:
        attrs = {**attrs}
        super().__init__(args, attrs, name)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, _ = self.args
        return {"i": i}

    @override
    def generate_op(
        self, inputs: Sequence[TETensor] | None = None
    ) -> tuple[TETensor, ...]:
        i, dtype = self.args
        if inputs is None:
            A = te.placeholder(self.attrs["inp_shape"], name="A", dtype=dtype)
        else:
            (A,) = inputs
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
        return A, O

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        shape = self.attrs["inp_shape"]
        return (shape,)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        shape = self.attrs["inp_shape"]
        axes = self.attrs["axes"]
        if axes == ():
            axes = list(range(len(shape)))[::-1]
        out_shape = tuple([shape[n] for n in axes])
        return (out_shape,)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        _, dtype = self.args
        return (dtype,)


class TVMOperators:
    @classmethod
    def from_name(cls, name: str) -> Type[TVMOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = TVMOperatorMatmul
    relu = TVMOperatorRelu
    conv2d = TVMOperatorConv2D
    pad2d = TVMOperatorPad2D
    transpose = TVMOperatorTranspose
