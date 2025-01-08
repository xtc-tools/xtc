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
        source_op: xdslOperation,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
        payload_name: str = "f",
        concluding_passes: list[str] = [],
        loop_stamps: list[str] = [],
        always_vectorize: bool = True,
        no_alias: bool = False,
        id: str | None = None,
    ):
        if id is None:
            self.op_id_attribute = f"__id{MlirNodeImplementer.count}__"
            MlirNodeImplementer.count += 1
            source_op.attributes[self.op_id_attribute] = xdslUnitAttr()
        else:
            self.op_id_attribute = id
        assert self.op_id_attribute in source_op.attributes
        xdsl_func = xdsl_operator_to_function(source_op, payload_name)
        # Call the parent constructor
        super().__init__(
            xdsl_func=xdsl_func,
            concluding_passes=concluding_passes,
            always_vectorize=always_vectorize,
            no_alias=no_alias,
        )
        # Used for validation
        self.source_op = source_op
        # Specification of transformations
        self.loop_stamps = loop_stamps
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.reduction_dims = reduction_dims
        self.tiles = {k: {k: 1} for k, _ in self.dims.items()}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling: dict[str, int] = dict([])

    def string_of_schedule(self) -> str:
        return (
            f"dims: {self.dims},"
            + f"tiles: {self.tiles},"
            + f"order: {self.permutation},"
            + f"vector: {self.vectorization},"
            + f"parallel: {self.parallelization},"
            + f"unrolling: {self.unrolling}"
        )

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
        tiles_names = []
        tiles_sizes = []
        for tile_name, tile_size in tiles.items():
            if tile_size == 1:
                tile_size = 0
            tiles_names.append(tile_name)
            tiles_sizes.append(tile_size)

        dims = [dim] + tiles_names
        sizes = tiles_sizes + [1]
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self.permutation = self.get_default_interchange()
        if dim in self.parallel_dims:
            self.parallel_dims += tiles_names
        if dim in self.reduction_dims:
            self.reduction_dims += tiles_names

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
        for p in parallelization:
            assert p in self.parallel_dims
        # TODO
        ...

    def unroll(self, unrolling: dict[str, int]):
        for dim, ufactor in unrolling.items():
            if not dim in self.vectorization:
                self.unrolling[dim] = ufactor

    def needs_vectorization(self):
        return len(self.vectorization) > 0

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
            # Annotate the resulting loop if successfully generated
            if len(tiling_command.results) > 1:
                generated_loop = tiling_command.results[1]
                transform.AnnotateOp(
                    generated_loop, f"{self.op_id_attribute}{tile_name}"
                )
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
                op_attrs={f"{self.op_id_attribute}{dim}": UnitAttr.get()},
            )
            # TODO: LLVM metadata instead of transform unroll may put less pressure
            # on MLIR front-end
            # https://llvm.org/docs/LangRef.html#llvm-loop
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

    def get_all_tiles_sizes(self):
        all_dims_sizes = {}
        for dim, tiles in self.tiles.items():
            divided_dim = dim
            for tile_name, tile_size in tiles.items():
                if tile_size > 1:
                    divided_dim = divided_dim // tile_size
                    all_dims_sizes[tile_name] = tile_size
            all_dims_sizes[dim]

    @override
    def check_consistency(self):
        # Check the tiling
        all_dims_sizes = {}
        for dim, tiles in self.tiles.items():
            assert dim in self.dims
            divided_dim = self.dims[dim]
            for tile_name, tile_size in tiles.items():
                if tile_size == 1:
                    tile_size = divided_dim
                assert self.dims[dim] >= tile_size
                if tile_size > 0:
                    assert self.dims[dim] % tile_size == 0
                    divided_dim = divided_dim // tile_size
                all_dims_sizes[tile_name] = tile_size
        # Check the unrolling
        # TODO bug: the sizes in self.tiles are not the size of
        # the dim, but the size of the upper tile of the dim.
        # for dim, ufactor in self.unrolling.items():
        #     assert dim in all_dims_sizes
        #     dim_size = all_dims_sizes[dim]
        #     assert dim_size >= ufactor
        #     assert dim_size % ufactor == 0

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
