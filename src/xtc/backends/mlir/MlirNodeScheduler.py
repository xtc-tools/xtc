#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from dataclasses import dataclass
from xtc.itf.schd.scheduler import DEFAULT_ROOT

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
        self.tiles: dict[str, dict[str, int]] = {k: {} for k in self.dims}
        self.permutation: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}

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
        )

    @override
    def __str__(self) -> str:
        return str(self.mlir_node_schedule())

    def get_default_interchange(self, root: str) -> list[str]:
        ret = [f"{root}/{d}" for d in self.dims.copy()]
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
        segments_renamed = {f"{root}/{key}": val for key, val in segments.items()}
        self.splits[dim] = segments_renamed
        for s in segments_renamed:
            self.tiles[s] = {}

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT):
        for d, s in tiles.items():
            tile_name = f"{root}/{d}"
            self.tiles[dim][tile_name] = s

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT):
        self.permutation[root] = [f"{root}/{a}" for a in permutation]

    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.vectorization += [f"{root}/{a}" for a in axes]

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.parallelization = [f"{root}/{a}" for a in axes]

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT):
        for dim, ufactor in unrolls.items():
            self.unrolling[f"{root}/{dim}"] = ufactor
