#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, cast
import threading

from .graph import XTCGraph
from .node import XTCNode
from .expr import XTCExpr, XTCOpExpr, XTCTensorExpr


__all__ = [
    "XTCGraphScope",
]


class XTCGraphScope:
    _expr_nodes_map: dict[int, XTCNode] = {}

    def __init__(self, **graph_kwargs: Any) -> None:
        self._exprs: list[XTCExpr] = []
        self._names: list[str | None] = []
        self._outputs: list[XTCExpr] = []
        self._inputs: list[XTCExpr] = []
        self._graph_kwargs = graph_kwargs

    def add_expr(self, expr: XTCExpr, name: str | None = None) -> None:
        self._exprs.append(expr)
        self._names.append(name)

    def add_outputs(self, *outs: XTCExpr) -> None:
        self._outputs.extend(outs)

    def add_inputs(self, *inps: XTCExpr) -> None:
        self._inputs.extend(inps)

    def set_outputs(self, *outs: XTCExpr) -> None:
        self._outputs = list(outs)

    def set_inputs(self, *inps: XTCExpr) -> None:
        self._inputs = list(inps)

    def _infer_inputs(self, inps_seed: list[XTCExpr]) -> list[XTCExpr]:
        defs = set(self._exprs)
        uses = []
        for expr in self._exprs:
            if isinstance(expr, XTCOpExpr):
                for arg in expr.args:
                    uses.append(arg)
        inputs = list({use._idx: use for use in uses if use not in defs}.values())
        # when no order specified, sort by expr id
        inputs = sorted(inputs, key=lambda x: x._idx)
        inputs = inps_seed + inputs
        inputs = list({expr._idx: expr for expr in inputs}.values())
        return inputs

    def _infer_outputs(self, outs_seed: list[XTCExpr]) -> list[XTCExpr]:
        uses = set()
        for expr in self._exprs:
            if isinstance(expr, XTCOpExpr):
                for arg in expr.args:
                    uses.add(arg)
        outputs = outs_seed + [expr for expr in self._exprs if expr not in uses]
        outputs = list({expr._idx: expr for expr in outputs}.values())
        return outputs

    @classmethod
    def _node_from_expr(cls, expr: XTCExpr, name: str | None = None) -> XTCNode:
        with threading.Lock():
            if expr._idx not in cls._expr_nodes_map:
                cls._expr_nodes_map[expr._idx] = XTCNode(expr, name=name)
        return cls._expr_nodes_map[expr._idx]

    @property
    def graph(self) -> XTCGraph:
        graph = XTCGraph(**self._graph_kwargs)
        inputs = self._infer_inputs(self._inputs)
        outputs = self._infer_outputs(self._outputs)
        nodes = [
            self._node_from_expr(expr, name)
            for expr, name in zip(self._exprs, self._names)
        ]
        nodes_inputs = [self._node_from_expr(expr) for expr in inputs]
        nodes_outputs = [self._node_from_expr(expr) for expr in outputs]
        graph.add_nodes(nodes)
        graph.set_inputs(nodes_inputs)
        graph.set_outputs(nodes_outputs)
        if all(
            [
                isinstance(node._expr, XTCTensorExpr) and node._expr.type.is_constant()
                for node in graph.inputs_nodes
            ]
        ):
            inputs_types = [
                cast(XTCTensorExpr, node._expr).type for node in graph.inputs_nodes
            ]
            graph.forward_types(inputs_types)
        return graph


class XTCGraphScopes(threading.local):
    _scopes: list[XTCGraphScope]

    def __init__(self) -> None:
        # Initialize a global graph builder by default
        self._scopes = [XTCGraphScope()]

    def push(self, **graph_kwargs: Any) -> None:
        self._scopes.append(XTCGraphScope(**graph_kwargs))

    def pop(self) -> XTCGraphScope:
        assert len(self._scopes) > 1
        return self._scopes.pop()

    @property
    def current(self) -> XTCGraphScope:
        return self._scopes[-1]

    def append(self, expr: XTCExpr, name: str | None = None) -> XTCExpr:
        for scope in self._scopes:
            scope.add_expr(expr, name)
        return expr


XTCGraphContext = XTCGraphScopes()
