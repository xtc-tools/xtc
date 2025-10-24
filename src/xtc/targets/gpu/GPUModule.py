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

from .GPUEvaluator import GPUExecutor, GPUEvaluator
from xtc.targets.host import HostModule


__all__ = [
    "GPUModule",
]


class GPUModule(HostModule):
    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, payload_name, file_name, file_type, graph, **kwargs)

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        return GPUEvaluator(
            self,
            **kwargs,
        )

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        return GPUExecutor(
            self,
            **kwargs,
        )
