#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.utils.evaluation import (
    graph_np_inputs_spec,
    graph_np_outputs_spec,
    graph_reference_impl,
)
from .GPUEvaluator import GPUExecutor, GPUEvaluator


__all__ = [
    "GPUModule",
]


class GPUModule(itf.comp.Module):
    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._file_type = file_type
        assert self._file_type == "shlib", "only support shlib for JIR Module"
        lib_suffixes = ("so", "dylib")
        assert self._file_name.endswith(lib_suffixes), (
            f"file name {self._file_name} is not a shlib"
        )
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

    @override
    def export(self) -> None:
        raise NotImplementedError("GPUModule.export is not implemented")

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
