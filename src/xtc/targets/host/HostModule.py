#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any, Callable

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.utils.evaluation import (
    graph_np_inputs_spec,
    graph_np_outputs_spec,
    graph_reference_impl,
)

from .HostEvaluator import HostExecutor, HostEvaluator
from .HostCEvaluator import HostCExecutor, HostCEvaluator
from .HostAREvaluator import HostARExecutor, HostAREvaluator


__all__ = [
    "HostModule",
]


class HostModule(itf.comp.Module):
    def __init__(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        headers: list[str] = [],
        headers_path: list[str] = [],
        shlibs: list[str] = [],
        arlibs: list[str] = [],
        csrcs: list[str] = [],
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._file_type = file_type
        shlib_suffixes = ("so", "dylib")
        assert self._file_type in ["shlib", "csrc", "arlib"], (
            "only support shlib/csrc/arlib Module"
        )
        assert self._file_type != "shlib" or self._file_name.endswith(shlib_suffixes), (
            "file name is not a shlib"
        )
        assert self._file_type != "csrc" or self._file_name.endswith(".c"), (
            "file name is not c file"
        )
        assert self._file_type != "arlib" or self._file_name.endswith(".a"), (
            "file name is not an archive"
        )
        self._shlibs = shlibs
        self._arlibs = arlibs
        self._headers = headers
        self._headers_path = headers_path
        self._csrcs = csrcs
        self._bare_ptr = kwargs.get("bare_ptr", True)
        self._graph = graph
        self._np_inputs_spec: Callable[[], list[dict[str, Any]]] | None
        self._np_outputs_spec: Callable[[], list[dict[str, Any]]] | None
        self._reference_impl: Callable[[], None] | None
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
        pass

    @override
    def get_evaluator(self, **kwargs: Any) -> itf.exec.Evaluator:
        if self.file_type == "shlib":
            return HostEvaluator(self, **kwargs)
        if self.file_type == "csrc":
            return HostCEvaluator(self, **kwargs)
        return HostAREvaluator(self, **kwargs)

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        if self.file_type == "shlib":
            return HostExecutor(self, **kwargs)
        if self.file_type == "csrc":
            return HostCExecutor(self, **kwargs)
        return HostARExecutor(self, **kwargs)

    @property
    def shlibs(self) -> list[str]:
        return self._shlibs

    @property
    def arlibs(self) -> list[str]:
        return self._arlibs

    @property
    def csrcs(self) -> list[str]:
        return self._csrcs

    @property
    def headers(self) -> list[str]:
        return self._headers

    @property
    def headers_path(self) -> list[str]:
        return self._headers_path
