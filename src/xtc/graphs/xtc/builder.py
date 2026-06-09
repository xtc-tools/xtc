#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import yaml
from pathlib import Path

from .graph import XTCGraph
from .context import XTCGraphContext
from .expr import XTCTensorExpr, XTCExpr
from .operators import XTCOperator
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
        def tuplify(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: tuplify(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return tuple(tuplify(v) for v in obj)
            else:
                return obj

        if "name" in graph_dict:
            XTCGraphContext.name(graph_dict["name"])

        expr_uid_map = {}
        for inp in graph_dict["inputs"]:
            expr_uid_map[inp["uid"]] = XTCTensorExpr.from_dict(inp["expr"])
        XTCGraphContext.inputs(*expr_uid_map.values())

        for node in graph_dict["nodes"]:
            expr = node["expr"]
            args = [expr_uid_map.get(arg) for arg in expr["args"]]
            if "name" in node:
                args.append(node["name"])
            if not hasattr(op_factory, expr["op"]["name"]):
                version_mismatch = (
                    "version mismatch detected, "
                    if XTCOperator.version_string() != graph_dict["ops_version"]
                    else ""
                )
                raise ValueError(
                    version_mismatch
                    + f"serialized op {expr['op']['name']} is not implemented."
                )
            op_func = getattr(op_factory, expr["op"]["name"])
            expr_uid_map[node["uid"]] = op_func(*args, **tuplify(expr["op"]["attrs"]))

        outputs = [expr_uid_map[out["uid"]] for out in graph_dict["outputs"]]
        XTCGraphContext.outputs(*outputs)
        return graph_dict

    @classmethod
    def loads(cls, yaml_str: str) -> None:
        cls.from_dict(yaml.safe_load(yaml_str))

    @classmethod
    def load(cls, file_name: str | Path) -> None:
        with open(file_name, "r") as f:
            cls.from_dict(yaml.safe_load(f))

    def set_outputs(self, *outs: XTCExpr) -> None:
        return XTCGraphContext.current.set_outputs(*outs)

    def set_inputs(self, *inps: XTCExpr) -> None:
        return XTCGraphContext.current.set_inputs(*inps)
