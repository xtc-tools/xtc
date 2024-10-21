#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
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
    Module,
)

from mlir.dialects import (
    arith,
    memref,
    linalg,
    builtin,
    func,
)


class MlirModule:
    def __init__(
        self,
    ):
        #
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)
        self.schedule_injected = False
        #
        f64 = F64Type.get(context=self.ctx)
        self.ext_rtclock = self.add_external_function(
            name="rtClock",
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
        payload_func = func.FuncOp.parse(function, context=self.ctx)
        ip = InsertionPoint.at_block_begin(self.module.body)
        ip.insert(payload_func)
        self.local_functions[str(payload_func.name)] = payload_func
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

    def inject_schedule(self, schedule_kernel: list[str]):
        if self.schedule_injected:
            return
        attr_name = "transform.with_named_sequence"
        self.module.operation.attributes[attr_name] = UnitAttr.get(context=self.ctx)
        trans_script = (
            "module attributes {transform.with_named_sequence} {"
            + "\n"
            + "\n".join(schedule_kernel)
            + "\n"
            + "}"
        )
        trans_match = Module.parse(trans_script, context=self.ctx)
        with InsertionPoint(self.module.body):
            for o in trans_match.body.operations:
                o.operation.clone()
        self.schedule = True
