#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from JIROps import Operation

__all__ = [
    "JIRSchedulerAdaptor",
]


class JIRSchedulerAdaptor:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
    ) -> None:
        self.source_op = source_op
        self.dims = dims
        self.axes_dim_map = {k: v for k, v in zip(dims.keys(), source_op.dim_names)}
        self.axes_map = {k: v for k, v in zip(dims.keys(), source_op.axes_names)}
        self.reset()

    def reset(self) -> None:
        self.tiled = {}
        self.vectorized = []
        self.parallelized = []
        self.unrolled = {}
        self.order = []

    def tile(self, axis: str, tiles: dict[str, int]) -> None:
        self.tiled[axis] = tiles

    def vectorize(self, axes: list[str]) -> None:
        self.vectorized = axes

    def parallelize(self, axes: list[str]) -> None:
        self.parallelized = axes

    def unroll(self, axes_unroll: dict[str, int]) -> None:
        self.unrolled = axes_unroll

    def interchange(self, axes_order: list[str]) -> None:
        self.order = axes_order

    def _update_axis_maps(self) -> None:
        for axis, tiles in self.tiled.items():
            for idx, name in enumerate(tiles.keys()):
                self.axes_map[name] = f"{self.axes_map[axis]}_{name}"
                self.axes_dim_map[name] = f"{self.axes_dim_map[axis]}_{idx + 1}"

    def _generate_tiles_cmds(self) -> list[str]:
        if not self.tiled:
            return []
        dims = self._get_tiles_dims()
        cmds = []
        for axis, tiles in self.tiled.items():
            axes_names = [self.axes_map[ax] for ax in [axis, *tiles.keys()]]
            dim_names = [self.axes_dim_map[ax] for ax in [axis, *tiles.keys()]]
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

    def _get_tiles_dims(self) -> dict[str, int]:
        tiles_dims = {f"{ax}": size for ax, size in self.dims.items()}
        for axis, tiles in self.tiled.items():
            for tile, size in tiles.items():
                tiles_dims[tile] = size
        return tiles_dims

    def _get_transform_dims(self) -> dict[str, int]:
        tiles_dims = {}
        for axis, tiles in self.tiled.items():
            dim_name = f"{self.axes_dim_map[axis]}"
            parent_dim = self.dims[axis]
            tiles_dims[dim_name] = parent_dim
            for idx, size in enumerate(tiles.values()):
                assert parent_dim % size == 0
                dim_name = f"{self.axes_dim_map[axis]}_{idx + 1}"
                compl = parent_dim // size
                tiles_dims[dim_name] = size
                tiles_dims[f"{dim_name}$"] = compl
                parent_dim = size
        return tiles_dims

    def _generate_vector_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map[axis]} vector={dims[axis]}"
            for axis in self.vectorized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_unroll_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        # We skip unroll if the loop is already fully vectorized
        cmds = [
            f"update_props target={self.axes_map[axis]} unroll={size}"
            for axis, size in self.unrolled.items()
            if dims[axis] != 1 and axis not in self.vectorized
        ]
        return cmds

    def _generate_parallel_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map[axis]} parallel"
            for axis in self.parallelized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_interchange_cmds(self) -> list[str]:
        def generate_inter(current, order):
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

        if not self.order:
            return []
        current_order = list(self.dims.keys())
        for axis, tiles in self.tiled.items():
            idx = current_order.index(axis)
            for tile in tiles.keys():
                idx += 1
                current_order.insert(idx, tile)
        inter = generate_inter(current_order, self.order)
        cmds = [f"interchange target={self.axes_map[axis]}" for axis in inter]
        return cmds

    def generate_transform(self) -> tuple[str, dict[str, int]]:
        cmds = []
        self._update_axis_maps()
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
