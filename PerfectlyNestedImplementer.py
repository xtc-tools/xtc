#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from xdsl.ir import Operation
from AbsImplementer import AbsImplementer
import transform


class PerfectlyNestedImplementer(AbsImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        source_op: Operation,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
    ):
        super().__init__(mlir_install_dir)
        #
        self.source_op = source_op
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.reduction_dims = reduction_dims
        self.tiles = {k: {k: 1} for k, v in self.dims.items()}
        self.tile_sizes_cache = {}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling = dict([])

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

        hoisted0, get_hoist0 = transform.vector_hoist(vectorized)

        # Produce the unrolling instructions using the annotations on loops
        unroll_instrs = []
        for dim, factor in self.unrolling.items():
            # loop,match_loop = transform.match_by_attribute(vectorized,dim)
            loop, match_loop = transform.match_by_attribute(hoisted0, dim)
            unroll = transform.get_unroll(loop, factor)
            unroll_instrs += [match_loop, unroll]

        hoisted, get_hoist = transform.vector_hoist(hoisted0)
        # hoisted,get_hoist = transform.vector_hoist(vectorized)
        get_lower = transform.vector_lower_outerproduct_patterns(hoisted)

        lines = (
            [seq_sig, "{"]
            + tiling_instrs
            + vect_instrs
            + [get_hoist0]
            + unroll_instrs
            + [get_hoist]
            + get_lower
            + [transform.get_terminator(), "}"]
        )
        return sym_name, "\n".join(lines)

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
        nvect = []
        for v in self.vectorization:
            ind = self.permutation.index(v)
            for i in list((range(0, ind)))[::-1]:
                dim = self.permutation[i]
                if dim in self.reduction_dims or dim in self.vectorization:
                    break
                elif dim in self.unrolling:
                    nvect.append(dim)
        for nv in nvect:
            self.vectorization.append(nv)
            self.unrolling.pop(nv)

    @abstractmethod
    def payload(self, m, elt_type):
        pass

    @abstractmethod
    def main(self):
        pass
