#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from typing import Any
from typing_extensions import override

from xdsl.dialects import func as xdslfunc
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bf16,
    f16,
    f32,
    f64,
)

import xtc.itf as itf

from .MlirCompiler import MlirCompiler
from .MlirScheduler import MlirScheduler


ELT_TYPES: dict[str, BFloat16Type | Float16Type | Float32Type | Float64Type] = {
    "bfloat16": bf16,
    "float16": f16,
    "float32": f32,
    "float64": f64,
}
TYPES_MAP = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64"}


class MlirBackend(itf.back.Backend):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        always_vectorize: bool,
        no_alias: bool,
        concluding_passes: list[str],
        graph: itf.graph.Graph | None = None,
    ):
        self.xdsl_func = xdsl_func
        self.no_alias = no_alias
        self.always_vectorize = always_vectorize
        self.concluding_passes = concluding_passes
        self._payload_name = str(xdsl_func.sym_name).replace('"', "")
        self._graph = graph

    @property
    @override
    def payload_name(self) -> str:
        return self._payload_name

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        assert self._graph is not None
        return self._graph

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
