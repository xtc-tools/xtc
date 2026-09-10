#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import weakref
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


def _unlink_quietly(path: str) -> None:
    """Remove ``path`` if it still exists (a caller may have removed it)."""
    try:
        os.unlink(path)
    except OSError:
        pass


class IREEModule(itf.comp.Module):
    """A Module backed by an IREE VM flatbuffer (``.vmfb``)."""

    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        graph: Graph | None = None,
        parallelized: bool = False,
        owns_file: bool = False,
        **kwargs: Any,
    ) -> None:
        assert file_name.endswith(".vmfb"), "file name is not a vmfb"
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._graph = graph
        # An anonymous (temp) vmfb has no owner but this module, so tie its
        # deletion to the module's lifetime; a caller-provided dump_file is left
        # in place. finalize (not __del__) stays safe at interpreter shutdown.
        if owns_file:
            weakref.finalize(self, _unlink_quietly, file_name)
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

    def _resolve_num_threads(self, kwargs: dict[str, Any]) -> None:
        # A non-parallelized schedule runs single-threaded (local-sync); a
        # parallelized one uses `threads` workers (from the search) if given,
        # else the evaluator's own default. `threads` is consumed here.
        if "num_threads" in kwargs:
            return
        threads = kwargs.pop("threads", None)
        if not self._parallelized:
            kwargs["num_threads"] = 1
        elif threads is not None:
            kwargs["num_threads"] = int(threads)

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        self._resolve_num_threads(kwargs)
        return IREEEvaluator(self, **kwargs)

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        self._resolve_num_threads(kwargs)
        return IREEExecutor(self, **kwargs)
