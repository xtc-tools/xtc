#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from AbsImplementer import AbsImplementer
import transform


class PerfectlyNestedImplementer(AbsImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
    ):
        super().__init__(mlir_install_dir)
        #
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.reduction_dims = reduction_dims
        self.tiles = {k: {k: 1} for k, v in self.dims.items()}
        self.tile_sizes_cache = {}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling = dict([])

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
        for d in vectorization:
            vector_size = self.tile_sizes_cache[d]
            vector_index = self.permutation.index(d)
            assert vector_index > 0
            containing_dim = self.permutation[vector_index - 1]
            for ad, cd in self.tiles.items():
                if containing_dim in cd:
                    cd[containing_dim] = vector_size
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self.parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling

    @abstractmethod
    def payload(self, m, elt_type):
        pass

    @abstractmethod
    def uniquely_match(self):
        pass

    @abstractmethod
    def main(self):
        pass
