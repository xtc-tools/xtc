#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Callable
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.utils.evaluation import (
    graph_np_inputs_spec,
    graph_np_outputs_spec,
    graph_reference_impl,
)

from .IREEEvaluator import IREEEvaluator, IREEExecutor

__all__ = [
    "IREEModule",
]


class IREEModule(itf.comp.Module):
    """A Module backed by an IREE VM flatbuffer (``.vmfb``)."""

    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        graph: Graph | None = None,
        parallelized: bool = False,
        **kwargs: Any,
    ) -> None:
        assert file_name.endswith(".vmfb"), "file name is not a vmfb"
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._graph = graph
        # When the schedule parallelizes nothing, execution defaults to
        # single-threaded.
        self._parallelized = parallelized
        # Reference numpy input/output specs and implementation, used by the
        # evaluator to build inputs and validate outputs.
        self._np_inputs_spec: Callable[[], list[dict[str, Any]]] | None
        self._np_outputs_spec: Callable[[], list[dict[str, Any]]] | None
        self._reference_impl: Callable[..., None] | None
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
        return "vmfb"

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
        # The vmfb is already written to file_name by the compiler.
        pass

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        kwargs.setdefault("single_thread", not self._parallelized)
        return IREEEvaluator(self, **kwargs)

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        kwargs.setdefault("single_thread", not self._parallelized)
        return IREEExecutor(self, **kwargs)
