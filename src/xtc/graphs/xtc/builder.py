#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

from xtc.graphs.xtc.node import XTCNode

from .graph import XTCGraph
from .context import XTCGraphContext
from .expr import XTCTensorExpr
from . import op_factory


class graph_builder:
    def __init__(self, **graph_kwargs: Any) -> None:
        self._graph_kwargs = graph_kwargs
        self._graph: XTCGraph | None = None

    def __enter__(self) -> "graph_builder":
        XTCGraphContext.push(**self._graph_kwargs)
        return self

    def __exit__(self, *_: Any) -> None:
        scope = XTCGraphContext.pop()
        self._graph = scope.graph

    @property
    def graph(self) -> XTCGraph:
        assert self._graph is not None, "can't get graph inside builder context"
        return self._graph

    @classmethod
    def from_dict(cls, graph_dict: dict[str, Any]) -> Any:
        XTCGraphContext.push()

        expr_uid_map = {}
        for inp in graph_dict["inputs"]:
            expr_uid_map[inp["uid"]] = XTCTensorExpr.from_dict(inp["expr"])

        for node in graph_dict["nodes"]:
            expr = node["expr"]
            args = [expr_uid_map.get(arg) for arg in expr["args"]]
            if "name" in node:
                args.append(node["name"])
            op_func = getattr(op_factory, expr["op"]["name"])
            expr_uid_map[node["uid"]] = op_func(*args, **expr["op"]["attrs"])

        scope = XTCGraphContext.pop()
        print(scope.graph)
        return graph_dict
