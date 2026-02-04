#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from dataclasses import dataclass, asdict
from pprint import pformat
from xtc.itf.schd.scheduler import DEFAULT_ROOT, ROOT_SEP

__all__ = [
    "MlirNodeScheduler",
    "MlirNodeSchedule",
]


def basename(loop_name: str) -> str:
    return loop_name.split(ROOT_SEP)[-1]


@dataclass(frozen=True)
class MlirNodeSchedule:
    node_name: str
    node_ident: str
    dims: list[str]
    loop_stamps: list[str]
    splits: dict[str, dict[str, int]]
    tiles: dict[str, dict[str, int]]
    permutation: dict[str, list[str]]
    vectorization: list[str]
    parallelization: list[str]
    unrolling: dict[str, int]
    packed_buffers: dict[str, list[int]]
    memory_mesh: dict[str, int]
    processor_mesh: dict[str, int]
    distribution: dict[str, str]
    distributed_buffers: dict[str, dict]

    def index_of_dim(self, dim: str) -> int:
        return list(self.dims).index(dim)

    def is_tile(self, loop_name: str) -> bool:
        for tiles in self.tiles.values():
            for tile in tiles:
                if loop_name == tile:
                    return True
        return False

    def is_base(self, loop_name: str) -> bool:
        return basename(loop_name) in self.dims

    def dim_of_tile(self, loop_name: str) -> str:
        # Base dimension
        bn = basename(loop_name)
        if bn in self.dims:
            return bn
        # Tiled dimension
        for dim, tiles in self.tiles.items():
            for tile in tiles:
                if bn == dim or loop_name == tile:
                    return dim
        assert False

    def size_of_tile(self, tile_name: str) -> int | None:
        for tiles in self.tiles.values():
            if tile_name in tiles:
                return tiles[tile_name]
        return None

    @override
    def __str__(self):
        return pformat(asdict(self))


class MlirNodeScheduler:
    def __init__(
        self,
        node_name: str,
        node_ident: str,
        dims: list[str],
        loop_stamps: list[str] = [],
    ) -> None:
        self.node_name = node_name
        self.node_ident = node_ident
        self.loop_stamps = loop_stamps  # Specification of transformations
        self.dims = dims[:]
        self.splits: dict[str, dict[str, int]] = {}
        self.tiles: dict[str, dict[str, int]] = {k: {} for k in self.dims}
        self.permutation: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.packed_buffers: dict[str, list[int]] = {}
        self.memory_mesh: dict[str, int] = {}
        self.processor_mesh: dict[str, int] = {}
        self.distribution: dict[str, str] = {}
        self.distributed_buffers: dict[str, dict] = {}

    def mlir_node_schedule(self) -> MlirNodeSchedule:
        if not self.permutation:
            self.permutation[DEFAULT_ROOT] = self.get_default_interchange(DEFAULT_ROOT)

        return MlirNodeSchedule(
            node_name=self.node_name,
            node_ident=self.node_ident,
            dims=self.dims,
            loop_stamps=self.loop_stamps,
            tiles=self.tiles,
            splits=self.splits,
            permutation=self.permutation,
            vectorization=self.vectorization,
            parallelization=self.parallelization,
            unrolling=self.unrolling,
            memory_mesh=self.memory_mesh,
            packed_buffers=self.packed_buffers,
            processor_mesh=self.processor_mesh,
            distribution=self.distribution,
            distributed_buffers=self.distributed_buffers,
        )

    @override
    def __str__(self) -> str:
        return str(self.mlir_node_schedule())

    def get_default_interchange(self, root: str) -> list[str]:
        ret = [f"{root}{ROOT_SEP}{d}" for d in self.dims.copy()]
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for _, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                ret.append(dim_name)
        return ret

    def set_dims(self, dims: list[str]) -> None:
        assert len(dims) == len(self.dims)
        self.dims = dims[:]
        self.tiles = {k: {} for k in self.dims}

    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        segments_renamed = {
            f"{root}{ROOT_SEP}{key}": val for key, val in segments.items()
        }
        self.splits[dim] = segments_renamed
        for s in segments_renamed:
            self.tiles[s] = {}

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT):
        for d, s in tiles.items():
            tile_name = f"{root}{ROOT_SEP}{d}"
            self.tiles[dim][tile_name] = s

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT):
        self.permutation[root] = [f"{root}{ROOT_SEP}{a}" for a in permutation]

    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.vectorization += [f"{root}{ROOT_SEP}{a}" for a in axes]

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.parallelization = [f"{root}{ROOT_SEP}{a}" for a in axes]

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT):
        for dim, ufactor in unrolls.items():
            self.unrolling[f"{root}{ROOT_SEP}{dim}"] = ufactor

    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ):
        axis_key = f"{root}{ROOT_SEP}{axis}"
        if axis_key not in self.packed_buffers.keys():
            self.packed_buffers[axis_key] = [input_idx]
        else:
            self.packed_buffers[axis_key].append(input_idx)

    def define_memory_mesh(self, axes: dict[str, int]):
        assert len(self.memory_mesh) == 0, "Memory mesh has already been defined"
        self.memory_mesh = axes

    def define_processor_mesh(self, axes: dict[str, int]):
        assert len(self.processor_mesh) == 0, "Processor mesh has already been defined"
        assert self.memory_mesh, "Memory mesh has not been defined"
        assert len(self.memory_mesh) <= len(axes), (
            "Memory mesh must be a subset of the processor mesh"
        )
        for i, memory_size in enumerate(self.memory_mesh.values()):
            assert list(axes.values())[i] == memory_size, (
                "Memory mesh must be a subset of the processor mesh"
            )
        self.processor_mesh = axes

    def distribute(self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT):
        assert self.processor_mesh, "Processor mesh has not been defined"
        assert processor_axis in self.processor_mesh or processor_axis == "*", (
            "Processor axis not found in processor mesh"
        )
        axis_key = f"{root}{ROOT_SEP}{axis}"
        self.parallelization.append(axis_key)
        self.distribution[axis_key] = processor_axis

    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ):
        assert self.memory_mesh, "Memory mesh has not been defined"
        for ma in memory_axes:
            assert ma in self.memory_mesh or ma == "*", (
                "Memory axis not found in memory mesh"
            )
        axis_key = f"{root}{ROOT_SEP}{axis}"
        self.distributed_buffers[axis_key] = {
            "input_idx": input_idx,
            "memory_axes": memory_axes,
        }
