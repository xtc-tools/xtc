#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any

import xtc.itf as itf
import xtc.backends.jir as backend

from .JIROps import JIROperation

__all__ = [
    "JIRScheduler",
    "JIRSchedule",
]


class JIRSchedulerAdaptor:
    def __init__(
        self,
        dims: dict[str, int],
        op_dims_names: tuple[str, ...],
        op_axes_names: tuple[str, ...],
    ) -> None:
        self.dims = dims
        self.op_dims_names = op_dims_names
        self.op_axes_names = op_axes_names
        self.tiles: dict[str, dict[str, int]] = {
            k: {k: v} for k, v in self.dims.items()
        }
        self.vectorized: list[str] = []
        self.parallelized: list[str] = []
        self.unrolled: dict[str, int] = {}
        self.order: list[str] = []
        self._axes_map: dict[str, str] = {}
        self._axes_dim_map: dict[str, str] = {}
        self._working_dims: dict[str, int] = {}
        self._working_permutation: list[str] = []

    def _update_axis_maps(self) -> None:
        self._axes_map = dict(zip(self.dims.keys(), self.op_axes_names))
        self._axes_dim_map = dict(zip(self.dims.keys(), self.op_dims_names))
        for axis, tiles in self.tiles.items():
            for idx, name in enumerate(tiles.keys()):
                if idx == 0:
                    continue
                self._axes_map[name] = f"{self._axes_map[axis]}_{name}"
                self._axes_dim_map[name] = f"{self._axes_dim_map[axis]}_{idx}"

    def _update_loops(self):
        loops = dict()
        parallels = []
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                axis_name, tile_size = list(v.items())[tile_level]
                loops[axis_name] = tile_size
        self._working_dims = loops
        permutation = self.order + list(loops.keys())
        self._working_permutation = list(dict.fromkeys(permutation))
        self._update_axis_maps()

    def split(self, dim: str, segments: dict[str, int]) -> None: ...

    def tile(self, axis: str, tiles: dict[str, int]) -> None:
        parent_size = self.dims[axis]
        for size in tiles.values():
            assert parent_size % size == 0
            parent_size = size
        self.tiles[axis] = {
            axis: self.dims[axis],
            **tiles,
        }
        self._update_loops()

    def vectorize(self, axes: list[str]) -> None:
        self.vectorized = axes

    def parallelize(self, axes: list[str]) -> None:
        self.parallelized = axes

    def unroll(self, axes_unroll: dict[str, int]) -> None:
        self.unrolled = axes_unroll

    def interchange(self, axes_order: list[str]) -> None:
        self.order = axes_order
        self._update_loops()

    def _generate_tiles_cmds(self) -> list[str]:
        cmds = []
        for axis, tiles in self.tiles.items():
            axes_names = [self._axes_map[ax] for ax in tiles]
            dim_names = [self._axes_dim_map[ax] for ax in tiles]
            for idx in range(len(axes_names) - 1):
                parent_size = dim_names[idx]
                sub_sizes = dim_names[idx + 1], f"{dim_names[idx + 1]}$"
                cmds.extend(
                    [
                        f"subdim parent={parent_size} sub=[{' '.join(sub_sizes)}]",
                        f"compl dim={sub_sizes[0]} other={sub_sizes[1]}",
                    ]
                )
            for idx in range(len(axes_names) - 1):
                parent = axes_names[idx]
                inner = axes_names[idx + 1]
                size = dim_names[idx + 1]
                tile_cmd = f"tile target={parent} tile={size} inner={inner}"
                cmds.append(tile_cmd)
        return cmds

    def _get_transform_dims(self) -> dict[str, int]:
        tiles_dims = {}
        for axis, tiles in self.tiles.items():
            parent_size = 0
            for idx, (_, size) in enumerate(tiles.items()):
                if idx == 0:
                    dim_name = f"{self._axes_dim_map[axis]}"
                    tiles_dims[dim_name] = size
                else:
                    assert parent_size % size == 0
                    dim_name = f"{self._axes_dim_map[axis]}_{idx}"
                    compl = parent_size // size
                    tiles_dims[dim_name] = size
                    tiles_dims[f"{dim_name}$"] = compl
                parent_size = size
        return tiles_dims

    def _get_tiles_dims(self) -> dict[str, int]:
        return {
            tile: size for tiles in self.tiles.values() for tile, size in tiles.items()
        }

    def _generate_vector_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self._axes_map[axis]} vector={dims[axis]}"
            for axis in self.vectorized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_unroll_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        # We skip unroll if the loop is already fully vectorized
        cmds = [
            f"update_props target={self._axes_map[axis]} unroll={size}"
            for axis, size in self.unrolled.items()
            if dims[axis] != 1 and axis not in self.vectorized
        ]
        return cmds

    def _generate_parallel_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self._axes_map[axis]} parallel"
            for axis in self.parallelized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_interchange_cmds(self) -> list[str]:
        def generate_inter(current: list[str], order: list[str]):
            inter = []
            assert len(current) == len(order), f"len mismatch {current} and {order}"
            for idx in range(len(order)):
                tgt_idx = current.index(order[idx])
                while tgt_idx != idx:
                    inter.append(current[tgt_idx - 1])
                    current[tgt_idx] = current[tgt_idx - 1]
                    current[tgt_idx - 1] = order[idx]
                    tgt_idx -= 1
            assert current == order
            return inter

        current_order = list(self.dims.keys())
        for axis, tiles in self.tiles.items():
            pos = current_order.index(axis)
            for idx, tile_axis in enumerate(tiles.keys()):
                if idx > 0:
                    current_order.insert(pos + idx, tile_axis)
        inter = generate_inter(current_order, self._working_permutation)
        cmds = [f"interchange target={self._axes_map[axis]}" for axis in inter]
        return cmds

    def generate_transform(self) -> tuple[list[str], dict[str, int]]:
        self._update_loops()
        transform_dims = self._get_transform_dims()
        tiles_cmds = self._generate_tiles_cmds()
        interchange_cmds = self._generate_interchange_cmds()
        vector_cmds = self._generate_vector_cmds()
        unroll_cmds = self._generate_unroll_cmds()
        parallel_cmds = self._generate_parallel_cmds()
        cmds = [
            *tiles_cmds,
            *interchange_cmds,
            *vector_cmds,
            *unroll_cmds,
            *parallel_cmds,
        ]
        return cmds, transform_dims


class JIRSchedule(itf.schd.Schedule):
    def __init__(self, scheduler: "JIRScheduler") -> None:
        self._scheduler = scheduler
        self._schedule = self._scheduler._generate_transform()

    def get_schedule_impl(self) -> tuple[list[str], dict[str, int]]:
        return self._schedule

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @override
    def __str__(self) -> str:
        cmds, dims = self._schedule
        if len(cmds) == 0:
            out = "commands: []\n"
        else:
            out = "commands:\n" + "".join([f"  - {cmd}\n" for cmd in cmds]) + "\n"
        if len(dims) == 0:
            out += "dims: {}\n"
        else:
            out += (
                "dims:\n" + "".join([f"  {k}: {v}\n" for k, v in dims.items()]) + "\n"
            )
        return out


class JIRScheduler(itf.schd.Scheduler):
    def __init__(self, backend: "backend.JIRBackend", **kwargs: Any) -> None:
        self._backend = backend
        self._transformer = JIRSchedulerAdaptor(
            self._backend.dims,
            self._backend.op.dim_names,
            self._backend.op.axes_names,
        )

    def _generate_transform(self) -> tuple[list[str], dict[str, int]]:
        return self._transformer.generate_transform()

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        return JIRSchedule(scheduler=self)

    @override
    def split(self, dim: str, segments: dict[str, int]) -> None: ...

    @override
    def tile(self, dim: str, tiles: dict[str, int]) -> None:
        self._transformer.tile(dim, tiles)

    @override
    def interchange(self, permutation: list[str]) -> None:
        self._transformer.interchange(permutation)

    @override
    def vectorize(self, axes: list[str]) -> None:
        self._transformer.vectorize(axes)

    @override
    def parallelize(self, axes: list[str]) -> None:
        self._transformer.parallelize(axes)

    @override
    def unroll(self, unrolls: dict[str, int]) -> None:
        self._transformer.unroll(unrolls)

    @override
    def buffer_at(self, axis: str, mtype: str | None = None) -> None:
        # TODO: not implemented
        pass

    def get_schedule_str(self) -> str:
        return str(JIRSchedule(scheduler=self))
