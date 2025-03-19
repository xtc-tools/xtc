#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast, Any
from typing_extensions import override

from xdsl.dialects import func as xdslfunc
from xdsl.dialects.builtin import AnyMemRefType
from xdsl.ir import Block as xdslBlock

from .MlirNodeImplementer import MlirNodeImplementer
from .MlirBackend import MlirBackend


class MlirGraphImplementer(MlirBackend):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        nodes: list[MlirNodeImplementer],
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
        no_alias: bool = False,
    ):
        self.nodes: dict[str, MlirNodeImplementer] = {}
        for impl in nodes:
            first_block = cast(xdslBlock, xdsl_func.body.first_block)
            assert impl.source_op in first_block.ops
            self.nodes[impl.payload_name] = impl
        super().__init__(
            xdsl_func=xdsl_func,
            always_vectorize=always_vectorize,
            concluding_passes=concluding_passes,
            no_alias=no_alias,
        )

    def _np_types_spec(
        self, types: list[AnyMemRefType]
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
    def np_inputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        inputs_args_types = [arg.type for arg in self.xdsl_func.args[:-1]]
        list_memref_tys = cast(list[AnyMemRefType], inputs_args_types)
        return self._np_types_spec(list_memref_tys)

    @override
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        # Assume inputs are first, and output is single last param
        outputs_args_types = [arg.type for arg in self.xdsl_func.args[-1:]]
        list_memref_tys = cast(list[AnyMemRefType], outputs_args_types)
        return self._np_types_spec(list_memref_tys)

    @override
    def reference_impl(self, *args: Any) -> None:
        assert False, "Implementation missing"
