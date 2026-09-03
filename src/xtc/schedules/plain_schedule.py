#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing_extensions import override
from dataclasses import dataclass, asdict, field
from pprint import pformat
from copy import deepcopy

from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.loop_names import make_loop_name, basename


@dataclass(frozen=True)
class PlainNodeSchedule:
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
    packed_buffers: dict[str, list[tuple[int, str | None, bool]]]
    write_buffers: dict[str, list[str | None]]
    memory_mesh: dict[str, int]
    processor_mesh: dict[str, int]
    distribution: dict[str, str]
    distributed_buffers: dict[str, dict]
    fused_producers: list[tuple[str, int]]
    fused_consumers: list[str]
    # Optional caller-provided vector sizes, keyed by vectorized axis name.
    # When an axis has a size, its dimension is vectorized with masking for
    # non-divisible extents; axes absent from this mapping are vectorized to
    # their (static) tile shape.
    vectorization_sizes: dict[str, int] = field(default_factory=dict)

    def index_of_dim(self, dim: str) -> int:
        return self.dims.index(basename(dim))

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
                    return basename(dim)
        assert False

    def size_of_tile(self, tile_name: str) -> int | None:
        for tiles in self.tiles.values():
            if tile_name in tiles:
                return tiles[tile_name]
        return None

    @override
    def __str__(self):
        return pformat(asdict(self), sort_dicts=False)

    def __post_init__(self):
        # TODO: for now keep legacy behavior which forces interchange
        # and empty tiling dict for the root_node
        if not self.tiles:
            for dim in self.dims:
                self.tiles[make_loop_name(DEFAULT_ROOT, dim)] = {}
        if not self.permutation:
            self.permutation[DEFAULT_ROOT] = []
            for dim in self.dims:
                self.permutation[DEFAULT_ROOT].extend(
                    [make_loop_name(DEFAULT_ROOT, dim)]
                    + list(self.tiles[make_loop_name(DEFAULT_ROOT, dim)])
                )


class PlainNodeScheduler:
    def __init__(
        self,
        node_name: str,
        node_ident: str,
        dims: list[str],
        loop_stamps: list[str] = [],
    ) -> None:
        self.node_name = node_name
        self.node_ident = node_ident
        self.dims = dims[:]
        self.loop_stamps = loop_stamps[:]
        self.splits: dict[str, dict[str, int]] = {}
        self.tiles: dict[str, dict[str, int]] = {}
        self.permutation: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.vectorization_sizes: dict[str, int] = {}
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.packed_buffers: dict[str, list[tuple[int, str | None, bool]]] = {}
        self.write_buffers: dict[str, list[str | None]] = {}
        self.memory_mesh: dict[str, int] = {}
        self.processor_mesh: dict[str, int] = {}
        self.distribution: dict[str, str] = {}
        self.distributed_buffers: dict[str, dict] = {}
        self.fused_producers: list[tuple[str, int]] = []
        self.fused_consumers: list[str] = []

    def get_plain_schedule(self) -> PlainNodeSchedule:
        return PlainNodeSchedule(
            node_name=self.node_name,
            node_ident=self.node_ident,
            dims=deepcopy(self.dims),
            loop_stamps=deepcopy(self.loop_stamps),
            tiles=deepcopy(self.tiles),
            splits=deepcopy(self.splits),
            permutation=deepcopy(self.permutation),
            vectorization=deepcopy(self.vectorization),
            parallelization=deepcopy(self.parallelization),
            unrolling=deepcopy(self.unrolling),
            memory_mesh=deepcopy(self.memory_mesh),
            packed_buffers=deepcopy(self.packed_buffers),
            write_buffers=deepcopy(self.write_buffers),
            processor_mesh=deepcopy(self.processor_mesh),
            distribution=deepcopy(self.distribution),
            distributed_buffers=deepcopy(self.distributed_buffers),
            fused_producers=deepcopy(self.fused_producers),
            fused_consumers=deepcopy(self.fused_consumers),
            vectorization_sizes=deepcopy(self.vectorization_sizes),
        )

    @override
    def __str__(self) -> str:
        return str(self.get_plain_schedule())

    def _get_default_interchange(self, root: str) -> list[str]:
        ret = [make_loop_name(root, d) for d in self.dims]
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

    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        segments_renamed = {
            make_loop_name(root, key): val for key, val in segments.items()
        }
        self.splits[make_loop_name(root, dim)] = segments_renamed

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT):
        tile_root = make_loop_name(root, dim)
        for d, s in tiles.items():
            if tile_root not in self.tiles:
                self.tiles[tile_root] = {}
            tile_name = make_loop_name(root, d)
            self.tiles[tile_root][tile_name] = s

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT):
        self.permutation[root] = [make_loop_name(root, a) for a in permutation]

    def vectorize(
        self,
        axes: list[str] | dict[str, int | None],
        root: str = DEFAULT_ROOT,
    ):
        # A list is equivalent to a mapping with None (full) widths.
        widths = axes if isinstance(axes, dict) else {a: None for a in axes}
        for axis, width in widths.items():
            loop = make_loop_name(root, axis)
            self.vectorization.append(loop)
            # Only explicit widths are recorded; a None means full vectorization.
            if width is not None:
                self.vectorization_sizes[loop] = width

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.parallelization += [make_loop_name(root, a) for a in axes]

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT):
        for dim, ufactor in unrolls.items():
            self.unrolling[make_loop_name(root, dim)] = ufactor

    def buffer_at(
        self,
        axis: str,
        mtype: str | None = None,
        root: str = DEFAULT_ROOT,
    ) -> None:
        buffer_axis = make_loop_name(root, axis)
        if buffer_axis not in self.write_buffers:
            self.write_buffers[buffer_axis] = []
        self.write_buffers[buffer_axis].append(mtype)

    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ):
        pack_axis = make_loop_name(root, axis)
        if pack_axis not in self.packed_buffers:
            self.packed_buffers[pack_axis] = []
        self.packed_buffers[pack_axis].append((input_idx, mtype, pad))

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
        dist_axis = make_loop_name(root, axis)
        self.parallelization.append(dist_axis)
        self.distribution[dist_axis] = processor_axis

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
        dist_axis = make_loop_name(root, axis)
        self.distributed_buffers[dist_axis] = {
            "input_idx": input_idx,
            "memory_axes": memory_axes,
        }

    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        fuse_axis = make_loop_name(root, axis)
        self.fused_producers.append((fuse_axis, input_idx))

    def fuse_consumer_at(self, axis: str, root: str = DEFAULT_ROOT) -> None:
        fuse_axis = make_loop_name(root, axis)
        self.fused_consumers.append(fuse_axis)
