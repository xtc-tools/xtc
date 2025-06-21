#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
from typing_extensions import override
from typing import TextIO, TypeAlias
from io import StringIO
from copy import deepcopy

import xtc.backends.tvm as backend
import xtc.itf as itf


# Actual backend Schedule implementation is the string
# for the TVM schedule
ScheduleImpl: TypeAlias = str


class TVMScheduler(itf.schd.Scheduler):
    def __init__(self, backend: "backend.TVMBackend") -> None:
        self.dims: dict[str, int] = {**backend.dims}
        self.parallel_dims: list[str] = [*backend.parallel_dims]
        self.tiles: dict[str, dict[str, int]] = {
            k: {k: v} for k, v in self.dims.items()
        }
        self.permutation: list[str] = []
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.write_caches: list[str] = []
        self._update_loops()
        self._backend = backend

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        io = StringIO()
        self._dump_tvm_schedule(outf=io)
        schedule_impl = io.getvalue()
        return TVMSchedule(scheduler=self, schedule_impl=schedule_impl)

    def _update_loops(self):
        loops = dict()
        parallels = []
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
                if k in self.parallel_dims:
                    parallels.append(dim_name)
        self._working_dims = loops
        self._working_parallel_dims = parallels
        self._working_permutation = list(self._working_dims.keys())
        self._parallel_axes = [k for k in self.dims.keys() if k in self.parallel_dims]
        self._reduction_axes = [
            k for k in self.dims.keys() if k not in self.parallel_dims
        ]

    @override
    def split(self, dim: str, segments: dict[str, int]) -> None: ...

    @override
    def tile(
        self,
        dim: str,
        tiles: dict[str, int],
    ) -> None:
        ndims = list(tiles.keys())
        tiles_sizes = list(tiles.values())

        assert len(ndims) == len(tiles_sizes)

        previous_tile_size = self.dims[dim]
        for ts in tiles_sizes:
            assert previous_tile_size % ts == 0
            previous_tile_size = ts

        dims = [dim] + ndims
        sizes = [self.dims[dim]] + tiles_sizes
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self._update_loops()

    @override
    def interchange(self, permutation: list[str]) -> None:
        self.permutation = permutation

    @override
    def buffer_at(self, axis: str, mtype: str | None = None) -> None:
        assert mtype is None or mtype == "write"
        self.write_caches.append(axis)

    @override
    def vectorize(self, axes: list[str]) -> None:
        for p in axes:
            assert p in self._working_parallel_dims
        self.vectorization = axes

    @override
    def parallelize(self, axes: list[str]) -> None:
        for p in axes:
            assert p in self._working_parallel_dims
        self.parallelization = axes

    @override
    def unroll(self, unrolls: dict[str, int]) -> None:
        self.unrolling = unrolls

    def _full_order(self) -> list[str]:
        permutation = self.permutation + self._working_permutation
        permutation = list(dict.fromkeys(permutation))
        return permutation

    def _full_tilings(self) -> dict[str, tuple[str, str, int]]:
        order = self._full_order()
        tilings = {}
        for dim, tiles in self.tiles.items():
            t_axes = [dim] + list(tiles.keys())
            t_sizes = list(tiles.values())
            tilings[dim] = (dim, "", self.dims[dim])
            for idx in range(1, len(tiles)):
                tilings[t_axes[idx + 1]] = (dim, t_axes[idx], t_sizes[idx])
        tilings = {axis: tilings[axis] for axis in order}
        return tilings

    def _write_buffer_tiling(
        self, axis: str, tilings: dict[str, tuple[str, str, int]]
    ) -> tuple[dict[str, tuple[str, str, int]], dict[str, tuple[str, str, int]]]:
        child = {
            parent: (dim, axis, factor)
            for axis, (dim, parent, factor) in tilings.items()
        }
        tiles_axis = list(tilings)
        tiles = list(tilings.items())
        tile_idx = tiles_axis.index(axis)
        outer_tiles = dict(list(tilings.items())[: tile_idx + 1])
        inner_tiles = dict(list(tilings.items())[tile_idx + 1 :])
        for axis, (dim, parent, factor) in list(outer_tiles.items()):
            factor = child.get(axis, ("", "", 1))[2]
            outer_tiles[f"{dim}_"] = (dim, axis, factor)
        dims = set()
        for axis, (dim, parent, factor) in list(inner_tiles.items()):
            if dim not in dims:
                dims.add(dim)
                parent = ""
            inner_tiles[axis] = (dim, parent, factor)
        return outer_tiles, inner_tiles

    def _full_write_buffers(
        self,
    ) -> dict[tuple[str, str, str], dict[str, tuple[str, str, int]]]:
        tilings = self._full_tilings()
        reorder_idx = {axis: idx for idx, axis in enumerate(tilings)}
        write_axis = sorted(self.write_caches, key=lambda axis: reorder_idx[axis])
        buffer_tilings = {}
        last_idx = 0
        out = ("O", "", "")
        tiling = tilings
        for idx, axis in enumerate(write_axis):
            outer_tiling, inner_tiling = self._write_buffer_tiling(axis, tiling)
            buffer_tilings[out] = outer_tiling
            out = (f"O_W{idx}", out[0], axis)
            tiling = inner_tiling
        buffer_tilings[out] = tiling
        return buffer_tilings

    def _emit_assign_axis(self, sch: str, tens: str, outf: TextIO) -> None:
        if self._parallel_axes:
            print(f"{', '.join(self._parallel_axes)}, = {tens}.op.axis", file=outf)
        if self._reduction_axes:
            print(
                f"{', '.join(self._reduction_axes)}, = {tens}.op.reduce_axis", file=outf
            )

    def _emit_assign_tilings(
        self,
        sch: str,
        tens: str,
        tilings: dict[str, tuple[str, str, int]],
        outf: TextIO,
    ) -> None:
        for axis, (dim, parent, factor) in tilings.items():
            if not parent:
                if axis != dim:
                    print(f"{axis} = {dim}", file=outf)
                continue
            print(
                f"{parent}, {axis} = {sch}[{tens}].split({parent}, factor={factor})",
                file=outf,
            )
        print(f"{sch}[{tens}].reorder({', '.join(tilings)})", file=outf)

    def _dump_tvm_schedule(
        self, obj: str = "obj", sch: str = "sch", outf: TextIO = sys.stdout
    ):
        tilings = self._full_write_buffers()
        for tens, parent, _ in tilings:
            if not parent:
                print(f"{tens} = {obj}[-1]", file=outf)
            else:
                print(f'{tens} = {sch}.cache_write({parent}, "local")', file=outf)
        for idx, ((tens, parent, axis), tiles) in enumerate(tilings.items()):
            if parent:
                print(f"{sch}[{tens}].compute_at({sch}[{parent}], {axis})", file=outf)
            self._emit_assign_axis(sch, tens, outf)
            self._emit_assign_tilings(sch, tens, tiles, outf)
            for axis, unroll in self.unrolling.items():
                if axis in tiles:
                    print(f"{sch}[{tens}].unroll({axis})", file=outf)
            for axis in self.vectorization:
                if axis in tiles:
                    print(f"{sch}[{tens}].vectorize({axis})", file=outf)
            if self.parallelization:
                if self.parallelization[0] in tiles:
                    if len(self.parallelization) > 1:
                        print(
                            f"{self.parallelization[-1]} = {sch}[{tens}].fuse({', '.join(self.parallelization)})",
                            file=outf,
                        )
                    print(
                        f"{sch}[{tens}].parallel({self.parallelization[-1]})", file=outf
                    )

    def dump_schedule(self, obj: str | None = None, outf: TextIO = sys.stdout):
        if obj is None:
            obj = "sch"
        for dim, tiles in self.tiles.items():
            t_tiles = {k: v for i, (k, v) in enumerate(tiles.items()) if i >= 1}
            print(f"{obj}.tile('{dim}', {t_tiles})", file=outf)
        print(f"{obj}.interchange({self.permutation})", file=outf)
        print(f"{obj}.vectorize({self.vectorization})", file=outf)
        print(f"{obj}.unroll({self.unrolling})", file=outf)
        print(f"{obj}.parallelize({self.parallelization})", file=outf)

    def get_schedule_str(self, obj: str | None = None) -> str:
        io = StringIO()
        self.dump_schedule(outf=io)
        return io.getvalue()

    @override
    def __str__(self) -> str:
        return self.get_schedule_str()


class TVMSchedule(itf.schd.Schedule):
    def __init__(self, scheduler: "TVMScheduler", schedule_impl: ScheduleImpl) -> None:
        self._scheduler = scheduler
        self._schedule_impl = schedule_impl

    @property
    def schedule_impl(self) -> ScheduleImpl:
        return self._schedule_impl

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @override
    def __str__(self) -> str:
        return str(self._schedule_impl)
