#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend

from .IREEScheduler import IREEScheduler

__all__ = [
    "IREEBackend",
]


class IREEBackend(itf.back.Backend):
    def __init__(self, source: Graph, **kwargs: Any) -> None:
        assert isinstance(source, XTCGraph), "IREE backend only supports graphs"
        self._graph: Graph = source
        # Reuse the MLIR backend to obtain linalg-on-tensors MLIR.
        self._mlir_backend = MlirGraphBackend(source, use_tensor_dialect=True)
        self.nodes_info = self._build_nodes_info(source)

    @staticmethod
    def _build_nodes_info(graph: XTCGraph) -> dict[str, dict[str, Any]]:
        nodes_info: dict[str, dict[str, Any]] = {}
        for node in graph.nodes.values():
            op = node.operation
            dims = list(op.dims.keys())
            parallel = set(op.dims_kind("P"))
            kinds = ["P" if d in parallel else "R" for d in dims]
            nodes_info[node.name] = {
                "op_id": f"__xtc_id_{node.name}_",
                "dims": dims,
                "kinds": kinds,
            }
        return nodes_info

    @property
    def mlir_backend(self) -> MlirGraphBackend:
        return self._mlir_backend

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return IREEScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        raise NotImplementedError("IREE compiler is added in a later patch")

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        return self._graph
