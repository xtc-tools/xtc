#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph

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
    ) -> None:
        assert file_name.endswith(".vmfb"), "file name is not a vmfb"
        self._name = name
        self._payload_name = payload_name
        self._file_name = file_name
        self._graph = graph
        # When the schedule parallelizes nothing, execution defaults to
        # single-threaded.
        self._parallelized = parallelized

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
        raise NotImplementedError("IREE evaluator is added in a later patch")

    @override
    def get_executor(self, **kwargs: Any) -> itf.exec.Executor:
        raise NotImplementedError("IREE executor is added in a later patch")
