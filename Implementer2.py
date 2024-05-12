#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess

from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.ir import Block, Region, MLContext, Operation
from xdsl.dialects.builtin import (
    ModuleOp,
    DenseIntOrFPElementsAttr,
    TensorType,
    MemRefType,
    f32,
    f64,
)
from xdsl.dialects import func, arith, linalg
from AbsImplementer import AbsImplementer
import transform


class Implementer(AbsImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        source_op: Operation,
        dims: dict[str, int],
        parallel_dims: list[str],
    ):
        super().__init__(mlir_install_dir)
        #
        self.source_op = source_op
        # TODO: should be computed
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.tiles = {k: {k: 1} for k, v in self.dims.items()}
        self.tile_sizes_cache = {}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling = dict([])

    def payload(self):
        # Fetch data
        operands = self.source_op.operands
        inputs = self.source_op.inputs
        inputs_types = [o.type for o in inputs]
        results_types = [r.type for r in self.source_op.results]
        #
        payload = Block(arg_types=inputs_types)
        outputs = self.outputs_init()
        outputs_vars = []
        for o in outputs:
            outputs_vars += o.results
        concrete_operands = list(payload.args) + outputs_vars
        value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

        new_op = self.source_op.clone(value_mapper=value_mapper)
        payload.add_ops(outputs + [new_op, func.Return(new_op)])
        payload_func = func.FuncOp.from_region(
            self.payload_name, inputs_types, results_types, Region(payload)
        )
        return payload_func

    def inputs_init(self):
        inputs_types = [o.type for o in self.source_op.inputs]
        return [
            arith.Constant(
                DenseIntOrFPElementsAttr.tensor_from_list(
                    [1.0], ty.get_element_type(), ty.get_shape()
                )
            )
            for ty in inputs_types
        ]

    def outputs_init(self):
        outputs_types = [o.type for o in self.source_op.outputs]
        return [
            arith.Constant(
                DenseIntOrFPElementsAttr.tensor_from_list(
                    [0.0], ty.get_element_type(), ty.get_shape()
                )
            )
            for ty in outputs_types
        ]

    def uniquely_match(self):
        dims = self.dims.values()

        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=False,
            has_output=True,
        )

        res_var, global_match_sig = transform.get_match_sig(input_var)
        bb_input_var, bb_header = transform.get_bb_header()

        match_dims = transform.get_match_dims(bb_input_var, dims)

        match_opname = transform.get_match_op_name(bb_input_var, self.source_op.name)

        tmyield = transform.get_match_structured_terminator(bb_input_var)

        tyield = transform.get_terminator(result=res_var)

        lines = (
            [
                seq_sig,
                "{",
                global_match_sig,
                "{",
                bb_header,
            ]
            + match_dims
            + [match_opname, tmyield, "}", tyield, "}"]
        )

        return sym_name, "\n".join(lines)

    def materialize_schedule(self):
        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=True,
            has_output=False,
        )

        # Build the transform vectors corresponding to each tiling instruction
        positions = {}
        dims_vectors = {}
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for dim_to_tile, (k, v) in enumerate(self.tiles.items()):
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
        dims_vectors = dict([(p, dims_vectors[p]) for p in self.permutation])

        # Actually produce the tiling instructions and annotate the resulting
        # loops
        current_state = input_var
        tiling_instrs = []
        vect_instrs = []
        loops = []
        for dim, dims_vector in dims_vectors.items():
            # Useless to materialize a loop which will be vectorized
            if dim in self.vectorization:
                break
            # The actual tiling
            current_state, new_loop, new_instr = transform.produce_tiling_instr(
                current_state=current_state,
                dims_vector=dims_vector,
                parallel=dim in self.parallelization,
            )
            loops.append(new_loop)
            # Name the resulting loop
            annot = transform.annotate(new_loop, dim)
            #
            tiling_instrs += [new_instr + annot]

        # If no vectorial tile, we scalarize the linalg op just after tiling
        if len(self.vectorization) == 0:
            scalarized, scalarization = transform.get_scalarize(current_state)
            current_state = scalarized

        # Obtain a handler for patterns application
        parent, parent_instr = transform.get_parent(loops[0])
        tiling_instrs.append(parent_instr)

        # Canonicalize the code produced by the tiling operations
        tiling_instrs += transform.tiling_apply_patterns(parent)

        # Produce the vectorization instructions
        vectorized, vectorize = transform.get_vectorize_children(parent)
        vect_instrs.append(vectorize)
        pre_hoist = transform.vector_pre_hoist_apply_patterns(vectorized)
        vect_instrs += pre_hoist

        # Produce the unrolling instructions using the annotations on loops
        unroll_instrs = []
        for dim, factor in self.unrolling.items():
            loop, match_loop = transform.match_by_attribute(vectorized, dim)
            unroll = transform.get_unroll(loop, factor)
            unroll_instrs += [match_loop, unroll]

        hoisted, get_hoist = transform.vector_hoist(vectorized)
        get_lower = transform.vector_lower_outerproduct_patterns(hoisted)

        lines = (
            [seq_sig, "{"]
            + tiling_instrs
            + vect_instrs
            + unroll_instrs
            + [get_hoist]
            + get_lower
            + [transform.get_terminator(), "}"]
        )
        return sym_name, "\n".join(lines)

    def materialize_schedule_deprecated(self):
        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=True,
            has_output=False,
        )

        # Build the transform vectors corresponding to each tiling instruction
        dims_vectors = {}
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for dim_to_tile, (k, v) in enumerate(self.tiles.items()):
                if tile_level >= len(v):
                    continue
                #
                dim_name = list(v.keys())[tile_level]
                dims_vectors[dim_name] = [
                    v[dim_name] if i == dim_to_tile else 0
                    for i in range(len(self.tiles))
                ]

        # Reorder the vectors (according to the permutation)
        dims_vectors = dict([(p, dims_vectors[p]) for p in self.permutation])

        # Actually produce the tiling instructions and annotate the resulting
        # loops
        current_state = input_var
        tiling_instrs = []
        vect_instrs = []
        for dim, dims_vector in dims_vectors.items():
            # Useless to materialize a loop which will be vectorized
            if dim in self.vectorization:
                continue
            # The actual tiling
            current_state, new_loop, new_instr = transform.produce_tiling_instr(
                current_state=current_state,
                dims_vector=dims_vector,
                parallel=dim in self.parallelization,
            )
            # Name the resulting loop
            annot = transform.annotate(new_loop, dim)
            #
            tiling_instrs += [new_instr + annot]

        # If no vectorial tile, we scalarize the linalg op just after tiling
        if len(self.vectorization) == 0:
            scalarized, scalarization = transform.get_scalarize(current_state)
            current_state = scalarized

        # Obtain a handler for patterns application
        parent, parent_instr = transform.get_parent(current_state)
        tiling_instrs.append(parent_instr)

        # Canonicalize the code produced by the tiling operations
        tiling_instrs += transform.tiling_apply_patterns(parent)

        # Produce the vectorization instructions
        vectorized, vectorize = transform.get_vectorize_children(parent)
        apply_patterns0 = transform.vector_pre_hoist_apply_patterns(vectorized)
        hoisted, hoist = transform.vector_hoist(vectorized)
        lower_contract = transform.vector_lower_outerproduct_patterns(hoisted)
        vect_instrs += [vectorize] + apply_patterns0 + [hoist] + lower_contract

        # Produce the unrolling instructions using the annotations on loops
        unroll_instrs = []
        for dim, factor in self.unrolling.items():
            loop, match_loop = transform.match_by_attribute(hoisted, dim)
            unroll = transform.get_unroll(loop, factor)
            unroll_instrs += [match_loop, unroll]

        lines = (
            [seq_sig, "{"]
            + tiling_instrs
            + vect_instrs
            + unroll_instrs
            + [transform.get_terminator(), "}"]
        )
        return sym_name, "\n".join(lines)

    def main(self, ext_rtclock, ext_printF64, payload_func, init_payload):
        results_types = [r.type for r in self.source_op.results]
        #
        inputs = self.inputs_init()
        rtclock_call1 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        outputs = self.outputs_init()
        payload_call = func.Call(payload_func.sym_name.data, inputs, results_types)
        rtclock_call2 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        elapsed = arith.Subf(rtclock_call2, rtclock_call1)
        print_elapsed = func.Call(ext_printF64.sym_name.data, [elapsed], [])
        main = Block()
        main.add_ops(
            inputs
            + [
                rtclock_call1,
                payload_call,
                rtclock_call2,
                elapsed,
                print_elapsed,
                func.Return(),
            ]
        )
        main_func = func.FuncOp.from_region("main", [], [], Region(main))
        return main_func

    def build_rtclock(self):
        return func.FuncOp.external("rtclock", [], [f64])

    def build_printF64(self):
        return func.FuncOp.external("printF64", [f64], [])

    def init_payload(self):
        #
        results_types = [r.type for r in self.source_op.results]
        #
        init_payload = Block()
        outputs = self.outputs_init()
        outputs_vars = []
        for o in outputs:
            outputs_vars += o.results
        ret = func.Return(*(o for o in outputs_vars))
        init_payload.add_ops(outputs + [ret])
        #
        init_payload_func = func.FuncOp.from_region(
            self.init_payload_name, [], results_types, Region(init_payload)
        )
        return init_payload_func

    def loops(self):
        loops = dict()
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
        return loops

    def get_default_interchange(self):
        return list(self.loops().keys())

    def tile(
        self,
        dim: str,
        tiles: dict[str, int],
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

    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def vectorize(self, vectorization: list[str]):
        # for d in vectorization:
        #     vector_size = self.tile_sizes_cache[d]
        #     vector_index = self.permutation.index(d)
        #     assert(vector_index > 0)
        #     containing_dim = self.permutation[vector_index - 1]
        #     for ad,cd in self.tiles.items():
        #         if containing_dim in cd:
        #             cd[containing_dim] = vector_size
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self.parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling
