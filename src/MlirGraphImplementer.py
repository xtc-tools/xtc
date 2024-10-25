#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast
from typing_extensions import override
from xdsl.dialects import func as xdslfunc
from xdsl.ir import Block as xdslBlock
from mlir.dialects.transform.structured import structured_match
from mlir.dialects import transform
from mlir.ir import UnitAttr

from MlirNodeImplementer import MlirNodeImplementer
from MlirImplementer import MlirImplementer


class MlirGraphImplementer(MlirImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        xdsl_func: xdslfunc.FuncOp,
        nodes: list[MlirNodeImplementer],
        vectors_size: int,
        concluding_passes: list[str] = [],
    ):
        self.nodes: dict[str, MlirNodeImplementer] = {}
        for impl in nodes:
            first_block = cast(xdslBlock, xdsl_func.body.first_block)
            assert impl.source_op in first_block.ops
            self.nodes[impl.payload_name] = impl
        #
        super().__init__(
            mlir_install_dir,
            xdsl_func,
            vectors_size,
            concluding_passes,
        )

    @override
    def generate_unroll(self, handle):
        for _, impl in self.nodes.items():
            impl.generate_node_unroll(handle)

    @override
    def generate_tiling(self):
        handle = None
        for _, impl in self.nodes.items():
            match0 = structured_match(
                results_=transform.AnyOpType.get(),
                target=self.named_sequence.bodyTarget,
                op_attrs={impl.op_id_attribute: UnitAttr.get()},
            )
            handle = impl.generate_node_tiling(match0)
        assert handle, "At least 1 operation should have been processed"
        return handle

    @override
    def np_inputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        assert False, "Implementation missing"

    @override
    def np_outputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        assert False, "Implementation missing"
        pass

    @override
    def reference_impl(self, *operands):
        assert False, "Implementation missing"
