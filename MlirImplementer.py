#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Tuple
import numpy as np

from mlir.ir import (
    Type,
)
from mlir.dialects import (
    func,
)
from xdsl.dialects.builtin import UnitAttr as xdslUnitAttr

from xdsl.ir import Operation as xdslOperation
from xdsl.ir import Attribute as xdslAttribute

from AbsImplementer import AbsImplementer
import transform
from MlirModule import MlirModule
from xdsl_aux import xdsl_operator_to_function

ty_tiles = dict[str, int]


class MlirImplementer(AbsImplementer):
    count = 0

    tiles: dict[str, ty_tiles]
    ext_rtclock: func.FuncOp

    def __init__(
        self,
        mlir_install_dir: str,
        source_op: xdslOperation,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
        vectors_size: int = 16,
        payload_name: str | None = None,
    ):
        super().__init__(mlir_install_dir)
        #
        self.vectors_size = vectors_size
        # Build the payload
        self._payload_name = (
            payload_name if payload_name else f"payload{MlirImplementer.count}"
        )
        self.op_id_attribute = f"id{MlirImplementer.count}"
        MlirImplementer.count += 1
        self.source_op = source_op
        self.source_op.attributes[self.op_id_attribute] = xdslUnitAttr()
        xdsl_func = xdsl_operator_to_function(source_op, self._payload_name)
        # Build MLIR primitives
        self._mlir_module = MlirModule()
        payload_func = self._mlir_module.parse_and_add_function(str(xdsl_func))
        self._mlir_module.measure_execution_time(
            new_function_name="entry",
            measured_function_name=str(payload_func.name),
        )
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

    @property
    def payload_name(self) -> str:
        return self._payload_name

    @property
    def ctx(self):
        return self._mlir_module.ctx

    @property
    def module(self):
        return self._mlir_module.module

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

    # This one will be overriden when multiple schedules
    def schedule_kernel(
        self,
        signature: str,
        input_var: str,
    ) -> list[str]:
        body = self.materialize_schedule(input_var=input_var)
        # kernel = [signature,"{"] + body + [transform.get_terminator(),"}"]
        kernel = [signature, "{"] + body + [transform.get_empty_terminator(), "}"]
        return kernel

    def inject_schedule(self):
        sym_name = "@__transform_main"
        myvar = transform.get_new_var()
        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_var=myvar, sym_name=sym_name
        )
        kernel = self.schedule_kernel(signature=seq_sig, input_var=input_var)
        self._mlir_module.inject_schedule(kernel)

    def materialize_tiling(self, global_handle: str) -> Tuple[list[str], str]:
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

        return tiling_instrs, loops[0]

    def normalize_and_vectorize(self, tiled_loop: str) -> Tuple[list[str], str]:
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

    def materialize_unrolling(self, vectorized: str) -> Tuple[list[str], str]:
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

        # if len(unroll_instrs) > 0:
        if False:
            continuation, parent_instr = transform.get_parent(last_handler)
            postprocess.append(parent_instr)
        else:
            continuation = last_handler

        return unroll_instrs + postprocess, continuation

    def materialize_schedule(self, input_var: str) -> list[str]:
        tiling_instrs, tiled_loop = self.materialize_tiling(global_handle=input_var)
        vect_instrs, vectorized = self.normalize_and_vectorize(tiled_loop)
        unroll_instrs, _ = self.materialize_unrolling(vectorized)
        tiling_and_vect_instrs = tiling_instrs + vect_instrs
        full_schedule = tiling_and_vect_instrs + unroll_instrs

        return full_schedule

    @classmethod
    def _np_types_spec(cls, types):
        types_map = {"f32": "float32", "f64": "float64"}
        types_spec = [
            {
                "shape": t.get_shape(),
                "dtype": types_map[str(t.get_element_type())],
            }
            for t in types
        ]
        return types_spec

    def np_inputs_spec(self):
        return self._np_types_spec([i.type for i in self.source_op.operands])

    def np_outputs_spec(self):
        return self._np_types_spec([i.type for i in self.source_op.results])

    def reference_impl(self, *operands):
        if self.source_op.name == "linalg.matmul":
            np.matmul(operands[0], operands[1], out=operands[2])
        else:
            assert 0, f"unknown implementation for operation: {self.source_op.name}"
