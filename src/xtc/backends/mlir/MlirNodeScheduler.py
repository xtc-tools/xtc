#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from dataclasses import dataclass, asdict

from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.plain_schedule import PlainNodeSchedule, PlainNodeScheduler

__all__ = [
    "MlirNodeScheduler",
    "MlirNodeSchedule",
]


@dataclass(frozen=True)
class MlirNodeSchedule(PlainNodeSchedule):
    pass


class MlirNodeScheduler:
    def __init__(
        self,
        node_name: str,
        node_ident: str,
        dims: list[str],
        loop_stamps: list[str] = [],
    ) -> None:
        self._plain_sch = PlainNodeScheduler(
            node_name,
            node_ident,
            dims,
        )

    @property
    def node_name(self) -> str:
        return self._plain_sch.node_name

    @property
    def node_ident(self) -> str:
        return self._plain_sch.node_ident

    def set_dims(self, dims: list[str]) -> None:
        self._plain_sch.set_dims(dims)

    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.split(dim, segments, root)

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.tile(dim, tiles, root)

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.interchange(permutation, root)

    def vectorize(
        self,
        axes: list[str] | dict[str, int | None],
        root: str = DEFAULT_ROOT,
    ) -> None:
        self._plain_sch.vectorize(axes, root)

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.parallelize(axes, root)

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.unroll(unrolls, root)

    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.buffer_at(axis, mtype, root)

    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ):
        self._plain_sch.pack_at(axis, input_idx, mtype, pad, root)

    def define_memory_mesh(self, axes: dict[str, int]):
        self._plain_sch.define_memory_mesh(axes)

    def define_processor_mesh(self, axes: dict[str, int]):
        self._plain_sch.define_processor_mesh(axes)

    def distribute(self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT):
        self._plain_sch.distribute(axis, processor_axis, root)

    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ):
        self._plain_sch.distributed_buffer_at(axis, input_idx, memory_axes, root)

    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.fuse_producer_at(axis, input_idx, root)

    def fuse_consumer_at(self, axis: str, root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.fuse_consumer_at(axis, root)

    def map_gpu_threads(self, axes: list[str], root: str = DEFAULT_ROOT):
        assert len(axes) <= 3, "We cannot map more than 3 dimension for gpu thread"
        assert len(axes) == len(set(axes)), "Duplicate in the axes for gpu thread"
        self._plain_sch.gpu_thread(axes, root)

    def map_gpu_blocks(self, axes: list[str], root: str = DEFAULT_ROOT):
        assert len(axes) == len(set(axes)), "Duplicate in the axes for gpu block"
        assert len(axes) <= 3, "We cannot map more than 3 dimension for gpu block"
        self._plain_sch.gpu_block(axes, root)

    def map_gpu_lanes(self, axes: list[str], root: str = DEFAULT_ROOT):
        assert len(axes) <= 3, "We cannot map more than 3 dimension for gpu lane"
        assert len(axes) == len(set(axes)), "Duplicate in the axes for gpu lane"
        self._plain_sch.gpu_lane(axes, root)

    def map_gpu_warps(self, axes: list[str], root: str = DEFAULT_ROOT):
        assert len(axes) == len(set(axes)), "Duplicate in the axes for gpu warp"
        assert len(axes) <= 3, "We cannot map more than 3 dimension for gpu warp"
        self._plain_sch.gpu_warp(axes, root)

    def get_node_schedule(self) -> MlirNodeSchedule:
        plain_schedule = self._plain_sch.get_plain_schedule()
        return MlirNodeSchedule(**asdict(plain_schedule))

    @override
    def __str__(self) -> str:
        return str(self.get_node_schedule())
