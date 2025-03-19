#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any
from pathlib import Path
import tempfile

from jir.node import JIRFunction
from jir.backend.util.annotate_fastmath import annotate_fastmath
from jir.parser import JIRParser
from jir.backend.xdsl.compiler import PolygeistCompiler

from xtc.utils.tools import (
    get_geist_prefix,
)

import xtc.itf as itf

from .JIROps import JIROperation
from .JIRScheduler import JIRScheduler
from .JIRCompiler import JIRCompiler
from .JIRGraph import JIROperator, JIRNode, JIRGraph

__all__ = [
    "JIRBackend",
]


class JIRBackend(itf.back.Backend):
    def __init__(
        self,
        source_op: JIROperation,
        dims: dict[str, int],
        **kwargs: Any,
    ) -> None:
        self.payload_name = source_op.name
        self.source_op = source_op
        self.args = self.source_op.args
        self.dims = dims
        self._geist_install_dir = get_geist_prefix()
        self._op_function_str, self._jir_function_str = self.source_op.generate()
        self._jir_function_op = self._parse_function(self._jir_function_str)
        self._op_function_mlir = self._parse_primitives(self._op_function_str)
        self._nodes = [
            JIRNode(
                self.payload_name,
                JIROperator(source_op),
                dims,
            ),
        ]
        self._graph = JIRGraph(self.payload_name, self._nodes)

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return JIRScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return JIRCompiler(self, **kwargs)

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

    def _parse_function(self, jir_function: str) -> JIRFunction:
        return JIRParser().parse_function(jir_function)

    def _parse_primitives(self, op_function: str) -> str:
        polygeist_compiler = PolygeistCompiler(f"{self._geist_install_dir}/bin/cgeist")
        return annotate_fastmath(polygeist_compiler(op_function))
