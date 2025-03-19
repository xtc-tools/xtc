#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "MlirNodeScheduler",
    "MlirNodeSchedule",
]


@dataclass(frozen=True)
class MlirNodeSchedule:
    node_name: str
    node_ident: str
    dims: list[str]
    loop_stamps: list[str]
    tiles: dict[str, dict[str, int]]
    permutation: list[str]
    vectorization: list[str]
    parallelization: list[str]
    unrolling: dict[str, int]


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
        self.tiles = {k: {k: 1} for k in self.dims}
        self.permutation = self.get_default_interchange()
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}

    def mlir_node_schedule(self) -> MlirNodeSchedule:
        return MlirNodeSchedule(
            node_name=self.node_name,
            node_ident=self.node_ident,
            dims=self.dims,
            loop_stamps=self.loop_stamps,
            tiles=self.tiles,
            permutation=self.permutation,
            vectorization=self.vectorization,
            parallelization=self.parallelization,
            unrolling=self.unrolling,
        )

    @override
    def __str__(self) -> str:
        return str(self.mlir_node_schedule())

    def loops(self) -> dict[str, int]:
        loops: dict[str, int] = dict()
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for _, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
        return loops

    def get_default_interchange(self) -> list[str]:
        return list(self.loops().keys())

    def tile(
        self,
        dim: str,
        tiles: dict[str, int],
    ):
        tiles_names = []
        tiles_sizes = []
        for tile_name, tile_size in tiles.items():
            tiles_names.append(tile_name)
            tiles_sizes.append(tile_size)
        dims = [dim] + tiles_names
        sizes = tiles_sizes + [1]
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self.permutation = self.get_default_interchange()

    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def vectorize(self, vectorization: list[str]):
        for dim_vect in vectorization:
            # Identify the tile level
            tile_level = None
            for dim, tiles in self.tiles.items():
                for i, (tile_name, tile_size) in enumerate(tiles.items()):
                    if tile_name == dim_vect:
                        tile_level = i
            assert not tile_level is None
            # Gather the tile level
            tiles_of_level = {}
            for dim, tiles in self.tiles.items():
                for i, (tile_name, tile_size) in enumerate(tiles.items()):
                    if i == tile_level:
                        tiles_of_level[tile_name] = tile_size
            # In the general case, we vectorize the whole tile level,
            # in order to let MLIR vectorization algorithms do
            # fancy stuff. But when the vectorized operation is
            # too big, we just vectorize the specified dimension because
            # the generated code is too heavy and stresses the back-end's
            # parser.
            vectorize_all_level = True
            for tile_name, tile_size in tiles_of_level.items():
                if tile_size > 64 or tile_size == 1:
                    vectorize_all_level = False
            # Update the dimensions to be vectorized
            if vectorize_all_level:
                for tile in tiles_of_level:
                    if tile in self.unrolling:
                        del self.unrolling[tile]
                    if not tile in self.vectorization:
                        self.vectorization.append(tile)
            else:
                self.vectorization.append(dim_vect)

    def parallelize(self, parallelization: list[str]):
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        for dim, ufactor in unrolling.items():
            if not dim in self.vectorization:
                self.unrolling[dim] = ufactor
