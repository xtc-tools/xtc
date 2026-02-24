#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

from xtc.itf.graph import Graph
import xtc.itf as itf

from .AIEEvaluator import AIEEvaluator, AIEExecutor

from xtc.utils.evaluation import (
    graph_np_inputs_spec,
    graph_np_outputs_spec,
    graph_reference_impl,
)

from mlir_sdist.extras.run_aie import AIEModuleWrapper

__all__ = [
    "AIEModule",
]


class AIEModule(itf.comp.Module):
    def __init__(
        self,
        name: str,
        payload: AIEModuleWrapper,
        payload_name: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._payload_wrapper = payload
        self._payload_name = payload_name
        self._file_name = "wrapper"
        self._file_type = "wrapper"
        self._bare_ptr = kwargs.get("bare_ptr", True)
        self._graph = graph
        if self._graph is not None:
            self._np_inputs_spec = graph_np_inputs_spec(self._graph)
            self._np_outputs_spec = graph_np_outputs_spec(self._graph)
            self._reference_impl = graph_reference_impl(self._graph)
        else:
            self._np_inputs_spec = kwargs.get("np_inputs_spec")
            self._np_outputs_spec = kwargs.get("np_outputs_spec")
            self._reference_impl = kwargs.get("reference_impl")

    @property
    @override
    def file_type(self) -> str:
        return self._file_type

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def payload_name(self) -> str:
        return self._payload_name

    @property
    @override
    def file_name(self) -> str:
        return self._file_name

    @property
    def wrapper(self) -> AIEModuleWrapper:
        return self._payload_wrapper

    @override
    def export(self) -> None:
        raise NotImplementedError("AcceleratorModule.export is not implemented")

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        return AIEEvaluator(
            self,
            **kwargs,
        )

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        return AIEExecutor(
            self,
            **kwargs,
        )
