#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

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
    def from_dict(self, graph_dict: dict[str, Any]) -> Any:
        XTCGraphContext.push()
        # TODO: scan for names, store in context _names
        # start at deepest expr than use factory to make OpExprs
        def build(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "op" in obj:
                    args = build(obj["args"])
                    func_name: str = obj["op"]["name"]
                    op_func = getattr(op_factory, func_name)
                    print(f"calling {func_name}")
                    obj = op_func(*args, **obj["op"]["attrs"])
                    return obj
                elif "idx" in obj:
                    obj = XTCTensorExpr.from_dict(obj)
                    return obj
                else:
                    print(obj)
                    return None
            elif isinstance(obj, list):
                obj = [build(o) for o in obj]
                return obj
            else:
                print(f"wtf is {obj}")
                return None

        
        for out in graph_dict["outputs"]:
            build(out["expr"])
        scope = XTCGraphContext.pop()
        print(scope.graph.inputs)
        for inp in scope.graph._inputs:
            print(inp.uid)
        print(scope.graph)
        return graph_dict
