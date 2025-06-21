#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from dataclasses import dataclass

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
    splits: dict[str, dict[str, int]]
    tiles: dict[str, dict[str, int]]
    permutation: dict[str, list[str]]
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
        self.splits: dict[str, dict[str, int]] = {}
        self.tiles = {k: {k: 1} for k in self.dims}
        self.permutation: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}

    def mlir_node_schedule(self) -> MlirNodeSchedule:
        if not self.permutation:
            self.permutation["."] = self.get_default_interchange()
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

    def split(self, dim: str, segments: dict[str, int]) -> None:
        self.splits[dim] = segments
        for s in segments:
            self.tiles[s] = {}

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

    def interchange(self, permutation: list[str]):
        assert permutation
        if self.node_name in self.permutation or self.node_name in permutation:
            root = permutation[0]
            interchange = permutation[1:]
        else:
            root = self.node_name
            interchange = permutation
        self.permutation[root] = interchange

    def vectorize(self, vectorization: list[str]):
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        for dim, ufactor in unrolling.items():
            # if not dim in self.vectorization:
            self.unrolling[dim] = ufactor
