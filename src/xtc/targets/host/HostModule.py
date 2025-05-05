#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, cast
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph
from xtc.graphs.xtc.data import XTCTensor
from xtc.graphs.xtc.expr import XTCTensorExpr

from .HostEvaluator import HostExecutor, HostEvaluator


__all__ = [
    "HostModule",
]


class HostModule(itf.comp.Module):
    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._file_type = file_type
        assert self._file_type == "shlib", "only support shlib for JIR Module"
        assert self._file_name.endswith(".so"), "file name is not a shlib"
        self._bare_ptr = kwargs.get("bare_ptr", True)
        self._graph = graph
        if self._graph is not None:
            self._np_inputs_spec = self._graph_np_inputs_spec
            self._np_outputs_spec = self._graph_np_outputs_spec
            self._reference_impl = self._graph_reference_impl
        else:
            self._np_inputs_spec = kwargs.get("np_inputs_spec")
            self._np_outputs_spec = kwargs.get("np_outputs_spec")
            self._reference_impl = kwargs.get("reference_impl")

    def _graph_np_inputs_spec(self) -> list[dict[str, Any]]:
        assert isinstance(self._graph, XTCGraph)
        assert all(
            [
                isinstance(node._expr, XTCTensorExpr) and node._expr.type.is_constant()
                for node in self._graph.inputs_nodes
            ]
        ), f"graph inputs are not tensors"
        inputs_types = [
            cast(XTCTensorExpr, node._expr).type for node in self._graph.inputs_nodes
        ]
        return [
            {
                "shape": type.constant_shape,
                "dtype": type.constant_dtype,
            }
            for type in inputs_types
        ]

    def _graph_np_outputs_spec(self) -> list[dict[str, Any]]:
        assert isinstance(self._graph, XTCGraph)
        assert all(
            [node._outputs_types is not None for node in self._graph.outputs_nodes]
        ), f"graph types were not forwarded"
        return [
            {
                "shape": type.constant_shape,
                "dtype": type.constant_dtype,
            }
            for type in [
                cast(list, node._outputs_types)[0] for node in self._graph.outputs_nodes
            ]
        ]

    def _graph_reference_impl(self, *args: Any) -> None:
        assert self._graph is not None
        inputs = [XTCTensor(inp) for inp in args[: len(self._graph.inputs)]]
        outputs = self._graph.forward(inputs)
        for idx, out in enumerate(args[len(self._graph.inputs) :]):
            out[:] = outputs[idx].numpy()

    @property
    @override
    def file_type(self) -> str:
        return self._file_type

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def payload_name(self) -> str:
        return self._payload_name

    @property
    @override
    def file_name(self) -> str:
        return self._file_name

    @override
    def export(self) -> None:
        pass

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        return HostEvaluator(
            self,
            **kwargs,
        )

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        return HostExecutor(
            self,
            **kwargs,
        )
