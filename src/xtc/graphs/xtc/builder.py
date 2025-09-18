#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

from .graph import XTCGraph
from .context import XTCGraphContext
from .expr import XTCExpr


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

    def set_outputs(self, *outs: XTCExpr) -> None:
        return XTCGraphContext.current.set_outputs(*outs)

    def set_inputs(self, *inps: XTCExpr) -> None:
        return XTCGraphContext.current.set_inputs(*inps)
