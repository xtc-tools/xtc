#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
from mlir.ir import (
    InsertionPoint,
    UnitAttr,
)
from mlir.dialects import transform
from xdsl.dialects import func as xdslfunc
from mlir.dialects.transform import (
    structured,
    vector,
    get_parent_op,
)

from xdsl_aux import brand_inputs_with_noalias
from MlirModule import MlirModule


class MlirImplementer(MlirModule, ABC):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        always_vectorize: bool,
        no_alias: bool,
        concluding_passes: list[str],
    ):
        if no_alias:
            brand_inputs_with_noalias(xdsl_func)
        self.payload_name = str(xdsl_func.sym_name).replace('"', "")
        #
        super().__init__([xdsl_func])
        #
        self.always_vectorize = always_vectorize
        self.concluding_passes = concluding_passes
        #
        self.schedule_injected = False
        #

    @property
    def mlir_payload(self):
        return self.payload_name

    def generate_vectorization(self, handle):
        handle = get_parent_op(
            transform.AnyOpType.get(),
            handle,
            isolated_from_above=True,
        )
        if self.always_vectorize or self.vectorize:
            handle = structured.VectorizeChildrenAndApplyPatternsOp(handle)
            with InsertionPoint(transform.ApplyPatternsOp(handle).patterns):
                vector.ApplyLowerOuterProductPatternsOp()
                vector.ApplyLowerContractionPatternsOp()
        return handle

    @abstractmethod
    def generate_tiling(self):
        pass

    @abstractmethod
    def generate_unroll(self, handle):
        pass

    @override
    def implement(self, measure=True):
        #
        if measure:
            self.measure_execution_time(
                entry_function_name="entry",
                measured_function_name=self.payload_name,
            )
        #
        with InsertionPoint(self.mlir_module.body), self.mlir_context, self.loc:
            self.mlir_module.operation.attributes["transform.with_named_sequence"] = (
                UnitAttr.get()
            )
            self.named_sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        with (
            InsertionPoint.at_block_begin(self.named_sequence.body),
            self.mlir_context,
            self.loc,
        ):
            handle = self.generate_tiling()
            self.generate_unroll(handle)
            handle = self.generate_vectorization(handle)
            for p in self.concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])
            self.schedule = True
