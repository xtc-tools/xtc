#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import threading

from xtc.itf.graph import Graph

from .graph import XTCGraph
from .node import XTCNode
from .expr import XTCExpr, XTCOpExpr


__all__ = [
    "XTCGraphContext",
]


class XTCGraphScope:
    _expr_nodes_map: dict[int, XTCNode] = {}

    def __init__(self, **graph_kwargs: Any) -> None:
        self._exprs: list[XTCExpr] = []
        self._outputs: list[XTCExpr] = []
        self._inputs: list[XTCExpr] = []
        self._graph_kwargs = graph_kwargs

    def add_exprs(self, *exprs: XTCExpr) -> None:
        self._exprs.extend(exprs)

    def add_outputs(self, *outs: XTCExpr) -> None:
        self._outputs.extend(outs)

    def add_inputs(self, *inps: XTCExpr) -> None:
        self._inputs.extend(inps)

    def _infer_inputs(self, inps_seed: list[XTCExpr]) -> list[XTCExpr]:
        defs = set(self._exprs)
        uses = []
        for expr in self._exprs:
            if isinstance(expr, XTCOpExpr):
                for arg in expr.args:
                    uses.append(arg)
        inputs = inps_seed + [use for use in uses if use not in defs]
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
    def _node_from_expr(cls, expr: XTCExpr) -> XTCNode:
        with threading.Lock():
            if expr._idx not in cls._expr_nodes_map:
                cls._expr_nodes_map[expr._idx] = XTCNode(expr)
        return cls._expr_nodes_map[expr._idx]

    @property
    def graph(self) -> Graph:
        graph = XTCGraph(**self._graph_kwargs)
        inputs = self._infer_inputs(self._inputs)
        outputs = self._infer_outputs(self._outputs)
        nodes = [self._node_from_expr(expr) for expr in self._exprs]
        nodes_inputs = [self._node_from_expr(expr) for expr in inputs]
        nodes_outputs = [self._node_from_expr(expr) for expr in outputs]
        graph.add_nodes(nodes)
        graph.set_inputs(nodes_inputs)
        graph.set_outputs(nodes_outputs)
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

    def append(self, expr: XTCExpr) -> XTCExpr:
        for scope in self._scopes:
            scope.add_exprs(expr)
        return expr

    def outputs(self, *outs: XTCExpr) -> None:
        return self.current.add_outputs(*outs)

    def inputs(self, *inps: XTCExpr) -> None:
        return self.current.add_inputs(*inps)


XTCGraphContext = XTCGraphScopes()
