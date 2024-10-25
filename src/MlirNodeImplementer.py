#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast
from typing_extensions import override
import numpy as np

from xdsl.dialects.builtin import UnitAttr as xdslUnitAttr
from xdsl.ir import Operation as xdslOperation
from xdsl.dialects.builtin import AnyMemRefType as xdslAnyMemRefType

from mlir.dialects.transform.structured import structured_match
from mlir.dialects import transform
from mlir.dialects.transform import (
    structured,
)
from mlir.dialects.transform.loop import loop_unroll
from mlir.ir import UnitAttr
from xdsl_aux import xdsl_operator_to_function
from MlirImplementer import MlirImplementer

ty_tiles = dict[str, int]


class MlirNodeImplementer(MlirImplementer):
    count = 0

    def __init__(
        self,
        mlir_install_dir: str,
        source_op: xdslOperation,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
        payload_name: str = "f",
        concluding_passes: list[str] = [],
        loop_stamps: list[str] = [],
        vectors_size: int = 16,
    ):
        #
        # Build the payload
        self.op_id_attribute = f"id{MlirNodeImplementer.count}"
        source_op.attributes[self.op_id_attribute] = xdslUnitAttr()
        MlirNodeImplementer.count += 1
        # To discard
        self.source_op = source_op
        #
        xdsl_func = xdsl_operator_to_function(source_op, payload_name)
        #
        super().__init__(mlir_install_dir, xdsl_func, vectors_size, concluding_passes)
        #
        self.loop_stamps = loop_stamps
        #
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.reduction_dims = reduction_dims
        self.tiles = {k: {k: 1} for k, _ in self.dims.items()}
        self.tile_sizes_cache: dict[str, int] = {}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling: dict[str, int] = dict([])

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
        tiles: ty_tiles,
    ):
        #
        for k, v in tiles.items():
            self.tile_sizes_cache[k] = v
        #
        ndims = list(tiles.keys())
        tiles_sizes = list(tiles.values())

        assert len(ndims) == len(tiles_sizes)

        previous_tile_size = self.dims[dim]
        for ts in tiles_sizes:
            assert previous_tile_size % ts == 0
            previous_tile_size = ts

        dims = [dim] + ndims
        sizes = tiles_sizes + [1]
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self.permutation = self.get_default_interchange()

        if dim in self.parallel_dims:
            self.parallel_dims += ndims
        if dim in self.reduction_dims:
            self.reduction_dims += ndims

    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def vectorize(self, vectorization: list[str]):
        self.vectorization = vectorization
        self.propagate_vectorization()

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self.parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling
        self.propagate_vectorization()

    def propagate_vectorization(self):
        # TODO: is it useful ? Evaluation needed
        # nvect: list[str] = []
        # for v in self.vectorization:
        #     ind = self.permutation.index(v)
        #     for i in list((range(0, ind)))[::-1]:
        #         dim = self.permutation[i]
        #         if dim in self.unrolling and dim in self.parallel_dims:
        #             nvect.append(dim)
        #         else:
        #             break
        # for nv in nvect:
        #     self.vectorization.append(nv)
        #     self.unrolling.pop(nv)
        return

    def generate_node_tiling(self, handle):
        # Produce the sequence of commands needed for the tiling
        tiling_arrays: dict[str, list[int]] = {}
        deepest_tiling = max(self.tiles.values(), key=len)
        depth_deepest_tiling = len(deepest_tiling)
        for tile_level in range(depth_deepest_tiling):
            for index_of_dim, (_, tiles) in enumerate(self.tiles.items()):
                # This dimension is not tiled at this level.
                if tile_level >= len(tiles):
                    continue

                # Create the array describing the tiling of this
                # dimension. If I have a (x,y,z) nest and I want
                # to tile the y dimension with a tile size of 16,
                # the resulting array is [0,16,0].
                tile_dim_name = list(tiles.keys())[tile_level]
                tiling_array = [
                    tiles[tile_dim_name] if i == index_of_dim else 0
                    for i in range(len(self.tiles))
                ]
                tiling_arrays[tile_dim_name] = tiling_array
        # Reorder the tiling according to permutation.
        tiling_arrays: dict[str, list[int]] = dict(
            [(p, tiling_arrays[p]) for p in self.permutation]
        )
        # Materialize loops
        op_to_tile = handle
        all_loops = []
        for tile_name, tiling_array in tiling_arrays.items():
            # Useless to materialize a loop which will be vectorized
            if tile_name in self.vectorization:
                break
            # Generate the tiling itself
            if tile_name in self.parallelization:
                tiling_command = structured.TileUsingForallOp(
                    op_to_tile, tile_sizes=tiling_array
                )
            else:
                tiling_command = structured.TileUsingForOp(
                    op_to_tile, sizes=tiling_array
                )
            # Annotate the resulting loop
            generated_loop = tiling_command.results[1]
            transform.AnnotateOp(generated_loop, f"{self.op_id_attribute}_{tile_name}")
            all_loops.append(generated_loop)
            #
            op_to_tile = tiling_command.results[0]

        # Stamp the outermost loop
        outer_loop = all_loops[0]
        for s in self.loop_stamps:
            transform.AnnotateOp(outer_loop, s)

        return outer_loop

    def generate_node_unroll(self, handle):
        for dim, factor in self.unrolling.items():
            match0 = structured_match(
                results_=transform.AnyOpType.get(),
                target=handle,
                op_attrs={f"{self.op_id_attribute}_{dim}": UnitAttr.get()},
            )
            loop_unroll(match0, factor)

    @override
    def generate_unroll(self, handle):
        self.generate_node_unroll(handle)

    @override
    def generate_tiling(self):
        match0 = structured_match(
            results_=transform.AnyOpType.get(),
            target=self.named_sequence.bodyTarget,
            op_attrs={self.op_id_attribute: UnitAttr.get()},
        )
        return self.generate_node_tiling(match0)

    @classmethod
    def _np_types_spec(
        cls, types: list[xdslAnyMemRefType]
    ) -> list[dict[str, tuple[int, ...] | str]]:
        types_map = {"f32": "float32", "f64": "float64"}
        types_spec: list[dict[str, tuple[int, ...] | str]] = [
            {
                "shape": t.get_shape(),
                "dtype": types_map[str(t.get_element_type())],
            }
            for t in types
        ]
        return types_spec

    @override
    def np_inputs_spec(self):
        list_attr_tys = [i.type for i in self.source_op.operands]
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)

    @override
    def np_outputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        list_attr_tys = [i.type for i in self.source_op.results]
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)

    @override
    def reference_impl(self, *operands):
        if self.source_op.name == "linalg.matmul":
            np.matmul(operands[0], operands[1], out=operands[2])
        else:
            assert 0, f"unknown implementation for operation: {self.source_op.name}"
