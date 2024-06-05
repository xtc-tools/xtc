#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from JIROps import Operation

__all__ = [
    "JIRSchedulerAdaaptor",
]


class JIRSchedulerAdaptor:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
    ) -> None:
        self.source_op = source_op
        self.dims = dims
        self.axes_map = {
            k: v for k, v in zip(dims.keys(), source_op.args_names[: len(self.dims)])
        }
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

    def _generate_tiles_cmds(self) -> list[str]:
        if not self.tiled:
            return []
        dims = self._get_tiles_dims()
        cmds = []
        for axis, tiles in self.tiled.items():
            dim_names = [f"{axis}{idx}" for idx in range(1 + len(tiles.keys()))]
            assert len(dim_names) == 2, f"for now only 1 level tiling supported"
            axes_names = [self.axes_map[axis]] + list(tiles.keys())
            subs = " ".join(dim_names)
            cmds.extend(
                [
                    f"subdim parent={self.axes_map[axis]} sub=[{subs}]",
                    f"compl dim={dim_names[0]} other={dim_names[1]}",
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
            dim = self.dims[axis]
            last = f"{axis}0"
            assert len(tiles.items()) == 1, f"for now only 1 level tiling supported"
            for tile, size in tiles.items():
                assert dim % size == 0
                dim = dim // size
                tiles_dims[last] = dim
                tiles_dims[tile] = size
                last = tile
        return tiles_dims

    def _generate_vector_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} vector={dims[axis]}"
            for axis in self.vectorized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_unroll_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} unroll={size}"
            for axis, size in self.unrolled.items()
            if dims[axis] != 1
        ]
        return cmds

    def _generate_parallel_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} parallel"
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
        dims = self._get_tiles_dims()
        inter = generate_inter(current_order, self.order)
        cmds = [f"interchange target={self.axes_map.get(axis, axis)}" for axis in inter]
        return cmds

    def generate_transform(self) -> tuple[str, dict[str, int]]:
        cmds = []
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
