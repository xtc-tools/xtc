#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import tempfile
from typing import Any
from pathlib import Path
from typing_extensions import override

import xtc.itf as itf

from .TVMOps import TVMOperation
from .TVMScheduler import TVMScheduler
from .TVMCompiler import TVMCompiler

__all__ = [
    "TVMBackend",
]


class TVMOperator(itf.operator.Operator):
    def __init__(
        self,
        source_op: TVMOperation,
    ):
        self._operator = source_op.operator

    @property
    @override
    def name(self) -> str:
        return self._operator.name

    @override
    def apply(self, inputs: list[itf.data.Tensor]) -> list[itf.data.Tensor]:
        # TODO
        return []

    @override
    def applyType(self, inputs: list[itf.data.TensorType]) -> list[itf.data.TensorType]:
        # TODO
        return []


class TVMNode(itf.graph.Node):
    def __init__(
        self,
        name: str,
        operator: TVMOperator,
        dims: dict[str, int],
        parallel_dims: list[str],
    ) -> None:
        self._name = name
        self._operator = operator
        self._dims = dims
        self._parallel_dims = parallel_dims

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def operator(self) -> itf.operator.Operator:
        return self._operator

    @property
    @override
    def inputs(self) -> list[str]:
        # TODO: Node inputs undefined for now
        return []

    @property
    @override
    def outputs(self) -> list[str]:
        # Tensor output 0 is node name
        return [self.name]


class TVMGraph(itf.graph.Graph):
    def __init__(
        self,
        name: str,
        nodes: list[TVMNode],
    ) -> None:
        assert len(nodes) > 0
        self._name = name
        self._nodes = nodes
        self._inputs = [nodes[0].name]
        self._outputs = [nodes[-1].name]

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def nodes(self) -> dict[str, itf.graph.Node]:
        return {node.name: node for node in self._nodes}

    @property
    @override
    def inputs(self) -> list[str]:
        return self._inputs

    @property
    @override
    def outputs(self) -> list[str]:
        return self._outputs


class TVMBackend(itf.back.Backend):
    def __init__(
        self,
        source_op: TVMOperation,
        dims: dict[str, int],
        parallel_dims: list[str],
    ) -> None:
        self.op = source_op
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.payload_name = self.op.name
        self._nodes = [
            TVMNode(self.payload_name, TVMOperator(source_op), dims, parallel_dims),
        ]
        self._graph = TVMGraph(self.payload_name, self._nodes)

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return TVMScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return TVMCompiler(self, **kwargs)

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        return self._graph

    def evaluate(
        self,
        schedule: itf.schd.Schedule,
        compiler_args: dict = {},
        evaluate_args: dict = {},
    ) -> float | str:
        with tempfile.TemporaryDirectory() as dirname:
            libpath = Path(dirname) / f"payload_{self.payload_name}"
            compiler = self.get_compiler(
                dump_file=str(libpath),
                shared_lib=True,
                **compiler_args,
            )
            module = compiler.compile(schedule)
            evaluator = module.get_evaluator(
                validate=True,
                **evaluate_args,
            )
            results, code, error_msg = evaluator.evaluate()
        return min(results) if code == 0 else error_msg
