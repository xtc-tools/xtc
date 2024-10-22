#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.ir import Operation as xdslOperation
from xdsl.dialects import func as xdslfunc

from xdsl_aux import xdsl_operator_to_function
from MlirNodeImplementer import MlirNodeImplementer
from MlirImplementer import MlirImplementer
import transform


class MlirGraphImplementer(MlirImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        xdsl_func: xdslfunc.FuncOp,
        nodes: list[MlirNodeImplementer],
        vectors_size: int,
    ):
        self.nodes: dict[str, MlirNodeImplementer] = {}
        for impl in nodes:
            assert impl.source_op in xdsl_func.body.first_block.ops
            self.nodes[impl.payload_name] = impl
        #
        super().__init__(mlir_install_dir, xdsl_func, vectors_size)

    def schedule_kernel(
        self,
        signature: str,
        input_var: str,
    ) -> list[str]:
        # Tiling
        tiling_block = []
        last_tiled_loop = None
        for _, impl in self.nodes.items():
            tiling_instrs, last_tiled_loop = impl.materialize_tiling(input_var)
            tiling_block += tiling_instrs
        # Vectorization. Very ad-hoc.
        first_impl = list(self.nodes.values())[0]
        vect_instrs, vectorized = first_impl.normalize_and_vectorize(last_tiled_loop)
        # Unrolling
        unroll_block = []
        to_unroll = vectorized
        for _, impl in self.nodes.items():
            unroll_instrs, to_unroll = impl.materialize_unrolling(to_unroll)
            unroll_block += unroll_instrs
        #
        body = tiling_block + vect_instrs + unroll_block
        kernel = [signature, "{"] + body + [transform.get_empty_terminator(), "}"]
        return kernel

    def np_inputs_spec(self) -> list[dict[str, list[int]]]:
        assert False, "Implementation missing"

    def np_outputs_spec(self) -> list[dict[str, list[int]]]:
        assert False, "Implementation missing"
        pass

    def reference_impl(self, *operands):
        assert False, "Implementation missing"
