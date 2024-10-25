#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
import numpy as np
from mlir.ir import (
    Type,
    IntegerType,
    F64Type,
    FunctionType,
    Context,
    Location,
    InsertionPoint,
    UnitAttr,
)
from mlir.dialects import (
    arith,
    memref,
    linalg,
    builtin,
    func,
)
from xdsl.dialects import func as xdslfunc
from mlir.dialects import transform
from mlir.dialects.transform import structured, vector, get_parent_op

from xdsl_aux import brand_inputs_with_noalias


class MlirModule(ABC):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        vectors_size: int,
        concluding_passes: list[str],
    ):
        self.vectors_size = vectors_size
        self.concluding_passes = concluding_passes
        #
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)
        self.schedule_injected = False
        #
        f64 = F64Type.get(context=self.ctx)
        self.ext_rtclock = self.add_external_function(
            name="rtclock",
            input_types=[],
            output_types=[f64],
        )
        self.ext_printF64 = self.add_external_function(
            name="printF64",
            input_types=[f64],
            output_types=[],
        )
        #
        self.local_functions = {}
        #
        brand_inputs_with_noalias(xdsl_func)
        payload_func = self.parse_and_add_function(str(xdsl_func))
        self.payload_name = str(payload_func.name).replace('"', "")
        self.measure_execution_time(
            new_function_name="entry",
            measured_function_name=self.payload_name,
        )
        #
        with InsertionPoint(self.module.body), self.ctx, self.loc:
            self.module.operation.attributes["transform.with_named_sequence"] = (
                UnitAttr.get()
            )
            self.named_sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        with InsertionPoint(self.named_sequence.body), self.ctx, self.loc:
            transform.YieldOp([])

    def add_external_function(
        self,
        name: str,
        input_types: list[Type],
        output_types: list[Type],
    ):
        with InsertionPoint.at_block_begin(self.module.body):
            myfunc = func.FuncOp(
                name=name,
                type=FunctionType.get(
                    inputs=input_types,
                    results=output_types,
                ),
                visibility="private",
                loc=self.loc,
            )
            return myfunc

    def parse_and_add_function(
        self,
        function: str,
    ) -> func.FuncOp:
        payload_func: func.FuncOp = func.FuncOp.parse(function, context=self.ctx)
        ip = InsertionPoint.at_block_begin(self.module.body)
        ip.insert(payload_func)
        name = str(payload_func.name).replace('"', "")
        self.local_functions[str(name)] = payload_func
        return payload_func

    def measure_execution_time(
        self,
        new_function_name: str,
        measured_function_name: str,
    ):
        measured_function = self.local_functions[measured_function_name]
        #
        with InsertionPoint.at_block_begin(self.module.body):
            fmain = func.FuncOp(
                name=new_function_name,
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
            self.local_functions[new_function_name] = fmain
        #
        with InsertionPoint(fmain.add_entry_block()), self.loc as loc:
            function_type = measured_function.type
            #
            inputs = []
            for ity in function_type.inputs:
                if IntegerType.isinstance(ity.element_type):
                    v = int(np.random.random())
                else:
                    v = np.random.random()
                scal = arith.ConstantOp(ity.element_type, v)
                mem = memref.AllocOp(ity, [], [])
                linalg.fill(scal, outs=[mem])
                inputs.append(mem)
            #
            callrtclock1 = func.CallOp(self.ext_rtclock, [], loc=self.loc)
            #
            for oty in function_type.results:
                v = 0 if IntegerType.isinstance(oty.element_type) else 0.0
                scal = arith.ConstantOp(oty.element_type, v)
                mem = memref.AllocOp(oty, [], [])
                linalg.fill(scal, outs=[mem])
                inputs.append(mem)
            #
            func.CallOp(measured_function, inputs, loc=self.loc)
            #
            callrtclock2 = func.CallOp(self.ext_rtclock, [], loc=self.loc)
            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(self.ext_printF64, [time], loc=self.loc)
            #
            for i in inputs:
                memref.DeallocOp(i)

            func.ReturnOp([], loc=self.loc)
        return fmain

    def generate_vectorization(self, handle):
        handle = get_parent_op(
            transform.AnyOpType.get(),
            handle,
            isolated_from_above=True,
        )
        if self.vectors_size > 0:
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

    def inject_schedule(self):
        with (
            InsertionPoint.at_block_begin(self.named_sequence.body),
            self.ctx,
            self.loc,
        ):
            handle = self.generate_tiling()
            handle = self.generate_vectorization(handle)
            self.generate_unroll(handle)
            for p in self.concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            self.schedule = True
