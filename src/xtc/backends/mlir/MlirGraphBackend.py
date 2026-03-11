#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast, Any
from typing_extensions import override

from xdsl.dialects.func import FuncOp as xdslFuncOp
from xdsl.dialects import func, memref
from xdsl.dialects.builtin import (
    MemRefType,
    f32,
    f64,
    ArrayAttr,
    UnitAttr,
    DictionaryAttr,
)
from xdsl.ir import Region, Block, Operation
from xdsl.builder import ImplicitBuilder

from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph, XTCNode
from xtc.graphs.xtc.data import XTCTensorType

from .MlirNodeBackend import MlirNodeBackend
from .MlirBackend import MlirBackend
from .MlirOps import MlirOperation


class MlirGraphBackend(MlirBackend):
    def __init__(
        self,
        xdsl_func: xdslFuncOp | Graph,
        nodes: list[MlirNodeBackend] | None = None,
        concluding_passes: list[str] = [],
        always_vectorize: bool = False,
        no_alias: bool = True,
    ):
        if isinstance(xdsl_func, XTCGraph):
            assert nodes is None
            graph = xdsl_func
            function, nodes_dict = self._init_from_graph(graph)
        else:
            assert isinstance(xdsl_func, xdslFuncOp)
            assert nodes is not None
            graph = None
            function, nodes_dict = self._init_from_xdsl(xdsl_func, nodes)
        self.nodes = nodes_dict
        super().__init__(
            xdsl_func=function,
            always_vectorize=always_vectorize,
            concluding_passes=concluding_passes,
            no_alias=no_alias,
            graph=graph,
        )

    def _init_from_xdsl(
        self,
        function: xdslFuncOp,
        nodes: list[MlirNodeBackend],
    ) -> tuple[xdslFuncOp, dict[str, MlirNodeBackend]]:
        nodes_dict = {}
        for impl in nodes:
            first_block = cast(Block, function.body.first_block)
            assert impl.source_op in first_block.ops
            nodes_dict[impl.payload_name] = impl
        return function, nodes_dict

    def _xdsl_generate_node(
        self, node: XTCNode, block: Block, variables: dict[str, Any]
    ):
        operation = MlirOperation.from_operation(node.operation, name=node.name)
        names = [*node.inputs, *node.outputs]
        assert node.inputs_types is not None and node.outputs_types is not None
        types = [*node.inputs_types, *node.outputs_types]
        for name, type in zip(names, types):
            if name in variables:
                continue
            with ImplicitBuilder(block):
                elt_type, shape = self._xdsl_elt_shape_from_tensortype(type)
                alloca = memref.AllocaOp.get(
                    return_type=elt_type,
                    shape=shape,
                    alignment=256,  # Take the default of dlpack lib
                )
            variables[name] = alloca.results[0]
        args = [variables[name] for name in names]
        _, attrs = operation.generate(block=block, args=args)
        return attrs

    def _init_from_graph(
        self,
        graph: XTCGraph,
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
        no_alias: bool = False,
    ) -> tuple[xdslFuncOp, dict[str, MlirNodeBackend]]:
        inputs_types = graph.inputs_types
        outputs_types = graph.outputs_types
        assert inputs_types is not None and outputs_types is not None, (
            f"graph types must be forwarded for graph {graph.name}"
        )
        params_types = [
            self._xdsl_type_from_tensortype(cast(XTCTensorType, tensor_type))
            for tensor_type in [*inputs_types, *outputs_types]
        ]
        arg_attrs = ArrayAttr(
            [
                DictionaryAttr(
                    self._xdsl_attrs_from_tensortype(cast(XTCTensorType, tensor_type))
                )
                for tensor_type in [*inputs_types, *outputs_types]
            ]
        )
        inlined_block = Block(arg_types=params_types)
        variables = {
            name: arg
            for name, arg in zip([*graph.inputs, *graph.outputs], inlined_block.args)
        }
        block_attrs = []
        for node in graph.nodes.values():
            node_attrs = self._xdsl_generate_node(node, inlined_block, variables)
            block_attrs.append(node_attrs)
        with ImplicitBuilder(inlined_block):
            func.ReturnOp()
        region = Region([inlined_block])  # type: ignore # issue with mypy
        payload = xdslFuncOp(
            name=graph.name,
            function_type=(params_types, []),
            region=region,
            arg_attrs=arg_attrs,
        )
        nodes_dict = {}
        for attrs in block_attrs:
            for (node_id, node), dims in zip(
                attrs["nodes_map"].items(), attrs["dims_sizes"]
            ):
                nodes_dict[node_id] = MlirNodeBackend(
                    payload_name=node_id,
                    source_op=cast(Operation, node),
                    dims=dims,
                    no_alias=no_alias,
                    always_vectorize=always_vectorize,
                    concluding_passes=concluding_passes,
                    id=f"__xtc_id_{node_id}_",
                )
        return payload, nodes_dict

    def _xdsl_elt_shape_from_tensortype(self, type: XTCTensorType) -> tuple[Any, Any]:
        elt_type = {"float32": f32, "float64": f64}[type.constant_dtype]
        return (elt_type, type.constant_shape)

    def _xdsl_type_from_tensortype(self, type: XTCTensorType) -> Any:
        elt_type, shape = self._xdsl_elt_shape_from_tensortype(type)
        return MemRefType(elt_type, shape)

    def _xdsl_attrs_from_tensortype(self, type: XTCTensorType):
        if type.device is not None:
            return {"memref.on_device": UnitAttr()}
        return {}

    def _np_types_spec(
        self, types: list[MemRefType]
    ) -> list[dict[str, tuple[int, ...] | str]]:
        types_map = {"f32": "float32", "f64": "float64"}
        types_spec: list[dict[str, tuple[int, ...] | str]] = [
            {
                "shape": t.get_shape(),
                "dtype": types_map[str(t.get_element_type())],
            }
            for t in types
        ]
        return types_spec

    @override
    def np_inputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        inputs_args_types = [arg.type for arg in self.xdsl_func.args[:-1]]
        list_memref_tys = cast(list[MemRefType], inputs_args_types)
        return self._np_types_spec(list_memref_tys)

    @override
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        outputs_args_types = [arg.type for arg in self.xdsl_func.args[-1:]]
        list_memref_tys = cast(list[MemRefType], outputs_args_types)
        return self._np_types_spec(list_memref_tys)
