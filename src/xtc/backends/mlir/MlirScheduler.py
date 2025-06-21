#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import cast

import xtc.itf as itf
import xtc.backends.mlir as backend

from .MlirNodeScheduler import MlirNodeScheduler, MlirNodeSchedule

__all__ = [
    "MlirScheduler",
    "MlirSchedule",
]


class MlirScheduler(itf.schd.Scheduler):
    def __init__(
        self,
        backend: "backend.MlirBackend",
        nodes_schedulers: list["MlirScheduler"] | None = None,
    ) -> None:
        from .MlirGraphBackend import MlirGraphBackend
        from .MlirNodeBackend import MlirNodeBackend

        self._backend = backend
        if isinstance(backend, MlirGraphBackend):
            if nodes_schedulers is None:
                self._nodes_schedulers = [
                    MlirScheduler(node_impl) for node_impl in backend.nodes.values()
                ]
            else:
                self._nodes_schedulers = nodes_schedulers
            # TODO: by default, when a graph, we assume to schedule the
            # output node which is assumed to be the last
            self._scheduler: MlirNodeScheduler | None = (
                self._nodes_schedulers[-1]._scheduler
                if len(self._nodes_schedulers) > 0
                else None
            )
        else:
            assert isinstance(backend, MlirNodeBackend)
            assert nodes_schedulers is None
            self._nodes_schedulers = [self]
            self._scheduler = MlirNodeScheduler(
                node_name=backend.payload_name,
                node_ident=backend.op_id_attribute,
                dims=list(backend.dims),
                loop_stamps=backend.loop_stamps,
            )

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        from .MlirGraphBackend import MlirGraphBackend
        from .MlirNodeBackend import MlirNodeBackend

        if isinstance(self._backend, MlirGraphBackend):
            assert not any(
                [scheduler._scheduler is None for scheduler in self._nodes_schedulers]
            )
            nodes_schedules = [
                cast(MlirNodeScheduler, scheduler._scheduler).mlir_node_schedule()
                for scheduler in self._nodes_schedulers
            ]
        else:
            assert isinstance(self._backend, MlirNodeBackend)
            assert self._scheduler is not None
            nodes_schedules = [self._scheduler.mlir_node_schedule()]
        return MlirSchedule(scheduler=self, nodes_schedules=nodes_schedules)

    @override
    def split(self, dim: str, segments: dict[str, int]) -> None:
        assert self._scheduler is not None
        self._scheduler.split(dim, segments)

    @override
    def tile(self, dim: str, tiles: dict[str, int]) -> None:
        assert self._scheduler is not None
        self._scheduler.tile(dim, tiles)

    @override
    def interchange(self, permutation: list[str]) -> None:
        assert self._scheduler is not None
        self._scheduler.interchange(permutation)

    @override
    def buffer_at(self, axis: str, mtype: str | None = None) -> None:
        assert self._scheduler is not None
        assert mtype is None or mtype == "write"
        # TODO: not implemented for now
        pass

    @override
    def vectorize(self, axes: list[str]) -> None:
        assert self._scheduler is not None
        self._scheduler.vectorize(axes)

    @override
    def parallelize(self, axes: list[str]) -> None:
        assert self._scheduler is not None
        self._scheduler.parallelize(axes)

    @override
    def unroll(self, unrolls: dict[str, int]) -> None:
        assert self._scheduler is not None
        self._scheduler.unroll(unrolls)


class MlirSchedule(itf.schd.Schedule):
    def __init__(
        self, scheduler: MlirScheduler, nodes_schedules: list[MlirNodeSchedule]
    ) -> None:
        self._scheduler = scheduler
        self._nodes_schedules = nodes_schedules

    @property
    def schedule_impl(self) -> list[MlirNodeSchedule]:
        return self._nodes_schedules

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @override
    def __str__(self) -> str:
        return str(self._nodes_schedules)
