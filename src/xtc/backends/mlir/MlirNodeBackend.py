#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import cast, Any
from typing_extensions import override
import numpy as np

from xdsl.ir import Operation as xdslOperation
from xdsl.dialects.builtin import AnyMemRefType as xdslAnyMemRefType
from xdsl.dialects.builtin import UnitAttr as xdslUnitAttr
from xtc.utils.xdsl_aux import xdsl_operator_to_function

from .MlirBackend import MlirBackend


class MlirNodeBackend(MlirBackend):
    count = 0

    def __init__(
        self,
        source_op: xdslOperation,
        dims: list[str],
        payload_name: str = "f",
        concluding_passes: list[str] = [],
        loop_stamps: list[str] = [],
        always_vectorize: bool = True,
        no_alias: bool = False,
        id: str | None = None,
    ):
        self._graph = None
        if id is None:
            self.op_id_attribute = f"__id{MlirNodeBackend.count}__"
            MlirNodeBackend.count += 1
        else:
            self.op_id_attribute = id
        source_op.attributes[self.op_id_attribute] = xdslUnitAttr()
        xdsl_func = xdsl_operator_to_function(source_op, payload_name)
        # Call the parent constructor
        super().__init__(
            xdsl_func=xdsl_func,
            concluding_passes=concluding_passes,
            always_vectorize=always_vectorize,
            no_alias=no_alias,
        )
        self.dims = dims
        self.source_op = source_op
        # Specification of transformations
        self.loop_stamps = loop_stamps

    def _np_types_spec(
        self, types: list[xdslAnyMemRefType]
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
        list_attr_tys = [i.type for i in self.source_op.inputs]  # type: ignore
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)

    @override
    def np_outputs_spec(self) -> list[dict[str, Any]]:
        list_attr_tys = [i.type for i in self.source_op.outputs]  # type: ignore
        list_memref_tys = cast(list[xdslAnyMemRefType], list_attr_tys)
        return self._np_types_spec(list_memref_tys)
