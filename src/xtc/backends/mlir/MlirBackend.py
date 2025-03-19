#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from typing import Any
from typing_extensions import override
from pathlib import Path
import tempfile

from mlir.dialects import transform
from xdsl.dialects import func as xdslfunc

import xtc.itf as itf

from .MlirCompiler import MlirCompiler
from .MlirScheduler import MlirScheduler


class MlirBackend(itf.back.Backend):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        always_vectorize: bool,
        no_alias: bool,
        concluding_passes: list[str],
    ):
        self.xdsl_func = xdsl_func
        self.no_alias = no_alias
        self.always_vectorize = always_vectorize
        self.concluding_passes = concluding_passes
        self.payload_name = str(xdsl_func.sym_name).replace('"', "")

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        assert False, "Implementation missing"

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return MlirScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return MlirCompiler(self, **kwargs)

    @abstractmethod
    def np_inputs_spec(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def reference_impl(self, *args: Any) -> None:
        pass

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
