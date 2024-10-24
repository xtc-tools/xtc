#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast
import numpy as np

from xdsl.dialects.builtin import UnitAttr as xdslUnitAttr
from xdsl.ir import Operation as xdslOperation
from xdsl.dialects.builtin import AnyMemRefType as xdslAnyMemRefType

from MlirImplementer import MlirImplementer
import transform
from xdsl_aux import xdsl_operator_to_function

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
        concluding_passes: list[str] = [],
        loop_stamps: list[str] = [],
        vectors_size: int = 16,
        payload_name: str | None = None,
    ):
        #
        # Build the payload
        self.op_id_attribute = f"id{MlirNodeImplementer.count}"
        payload_name = (
            payload_name if payload_name else f"payload{MlirNodeImplementer.count}"
        )
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
        nvect: list[str] = []
        for v in self.vectorization:
            ind = self.permutation.index(v)
            for i in list((range(0, ind)))[::-1]:
                dim = self.permutation[i]
                if dim in self.unrolling and dim in self.parallel_dims:
                    nvect.append(dim)
                else:
                    break
        for nv in nvect:
            self.vectorization.append(nv)
            self.unrolling.pop(nv)

    def schedule_kernel(
        self,
        signature: str,
        input_var: str,
    ) -> list[str]:
        handle, body = self.materialize_schedule(input_var=input_var)
        for p in self.concluding_passes:
            handle, instr = transform.get_registered_pass(handle, p)
            body.append(instr)
        kernel = [signature, "{"] + body + [transform.get_empty_terminator(), "}"]
        return handle, kernel

    def materialize_tiling(self, global_handle: str) -> tuple[list[str], str]:
        loop_to_tile, match_attr = transform.match_by_attribute(
            op=global_handle, attr=self.op_id_attribute
        )

        # Build the transform vectors corresponding to each tiling instruction
        positions = {}
        dims_vectors = {}
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for dim_to_tile, (_, v) in enumerate(self.tiles.items()):
                if tile_level >= len(v):
                    continue
                #
                dim_name = list(v.keys())[tile_level]
                dims_vectors[dim_name] = [
                    v[dim_name] if i == dim_to_tile else 0
                    for i in range(len(self.tiles))
                ]
                positions[dim_name] = dim_to_tile

        # Reorder the vectors (according to the permutation)
        dims_vectors: dict[str, list[int]] = dict(
            [(p, dims_vectors[p]) for p in self.permutation]
        )

        # Actually produce the tiling instructions and annotate the resulting
        # loops
        current_state = loop_to_tile
        tiling_instrs: list[str] = [match_attr]
        loops: list[str] = []
        for dim, dims_vector in dims_vectors.items():
            # Useless to materialize a loop which will be vectorized
            if dim in self.vectorization:
                # if self.vectorization_backpropagation_possible(dim):
                break
            # The actual tiling
            current_state, new_loop, new_instr = transform.produce_tiling_instr(
                current_state=current_state,
                dims_vector=dims_vector,
                parallel=dim in self.parallelization,
            )
            loops.append(new_loop)
            # Name the resulting loop
            annot = transform.annotate(new_loop, f"{self.op_id_attribute}_{dim}")
            #
            tiling_instrs += [new_instr + annot]

        outer_loop = loops[0]

        for s in self.loop_stamps:
            annot = transform.annotate(outer_loop, s)
            tiling_instrs.append(annot)

        return tiling_instrs, outer_loop

    def normalize_and_vectorize(self, tiled_loop: str) -> tuple[list[str], str]:
        parent, parent_instr = transform.get_parent(tiled_loop)

        # Canonicalize the code produced by the tiling operations
        norm_instrs = transform.tiling_apply_patterns(parent)
        parent_and_norm_instrs = [parent_instr] + norm_instrs

        # Produce the vectorization instructions
        vect_instrs = []
        vectorized = parent
        if len(self.vectorization) > 0:
            handler, vectorize = transform.get_vectorize_children(parent)
            pre_hoist = transform.vector_pre_hoist_apply_patterns(handler)
            hoisted0, get_hoist0 = transform.vector_hoist(handler)
            vect_instrs = [vectorize] + pre_hoist + [get_hoist0]
            vectorized = hoisted0
        else:
            vectorized, scalarization = transform.get_scalarize(vectorized)
            vect_instrs = [scalarization]

        return parent_and_norm_instrs + vect_instrs, vectorized

    def materialize_unrolling(self, vectorized: str) -> tuple[list[str], str]:
        last_handler = vectorized
        # Produce the unrolling instructions using the annotations on loops
        unroll_instrs: list[str] = []
        for dim, factor in self.unrolling.items():
            # loop,match_loop = transform.match_by_attribute(vectorized,dim)
            loop, match_loop = transform.match_by_attribute(
                vectorized, f"{self.op_id_attribute}_{dim}"
            )
            unroll = transform.get_unroll(loop, factor)
            unroll_instrs += [match_loop, unroll]
            last_handler = loop

        postprocess = []
        if len(self.vectorization) > 0:
            hoisted, get_hoist = transform.vector_hoist(vectorized)
            get_lower = transform.vector_lower_outerproduct_patterns(hoisted)
            postprocess = [get_hoist] + get_lower
            last_handler = hoisted

        return unroll_instrs + postprocess, last_handler

    def materialize_schedule(self, input_var: str) -> tuple[str, list[str]]:
        #
        tiling_instrs, tiled_loop = self.materialize_tiling(global_handle=input_var)
        #
        vect_instrs, vectorized = self.normalize_and_vectorize(tiled_loop)
        #
        unroll_instrs, unrolled = self.materialize_unrolling(vectorized)
        #
        tiling_and_vect_instrs = tiling_instrs + stamp_instrs + vect_instrs
        full_schedule = tiling_and_vect_instrs + unroll_instrs

        return unrolled, full_schedule

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

    def np_inputs_spec(self):
        list_attr_tys = [i.type for i in self.source_op.operands]
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)

    def np_outputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        list_attr_tys = [i.type for i in self.source_op.results]
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)

    def reference_impl(self, *operands):
        if self.source_op.name == "linalg.matmul":
            np.matmul(operands[0], operands[1], out=operands[2])
        else:
            assert 0, f"unknown implementation for operation: {self.source_op.name}"
