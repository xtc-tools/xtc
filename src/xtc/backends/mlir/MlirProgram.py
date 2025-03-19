#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.dialects import func as xdslfunc
from mlir.dialects import func, builtin, arith, memref, linalg
from mlir.ir import (
    Type,
    Context,
    Location,
    InsertionPoint,
    FunctionType,
    F64Type,
    IntegerType,
)
import numpy as np


class RawMlirProgram:
    def __init__(self, source: str):
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp.parse(source, context=self.ctx)

    @property
    def mlir_context(self):
        return self.ctx

    @property
    def mlir_module(self):
        return self.module


class MlirProgram(RawMlirProgram):
    def __init__(self, xdsl_func: xdslfunc.FuncOp) -> None:
        super().__init__("module{}")
        self.local_functions: dict[str, func.FuncOp] = {}
        self.parse_and_add_function(str(xdsl_func))
        self.payload_name = str(xdsl_func.sym_name).replace('"', "")

    def parse_and_add_function(
        self,
        function: str,
    ) -> func.FuncOp:
        payload_func: func.FuncOp = func.FuncOp.parse(
            function, context=self.mlir_context
        )
        ip = InsertionPoint.at_block_begin(self.mlir_module.body)
        ip.insert(payload_func)
        name = str(payload_func.name).replace('"', "")
        self.local_functions[str(name)] = payload_func
        return payload_func

    def add_external_function(
        self,
        name: str,
        input_types: list[Type],
        output_types: list[Type],
    ):
        with InsertionPoint.at_block_begin(self.mlir_module.body):
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

    def measure_execution_time(self):
        entry_function_name = "entry"
        measured_function_name = self.payload_name
        measured_function = self.local_functions[measured_function_name]
        # Create the external helpers (these ones are provided by MLIR library)
        f64 = F64Type.get(context=self.mlir_context)
        ext_rtclock = self.add_external_function(
            name="rtclock",
            input_types=[],
            output_types=[f64],
        )
        ext_printF64 = self.add_external_function(
            name="printF64",
            input_types=[f64],
            output_types=[],
        )
        # Create the entry point
        with InsertionPoint.at_block_begin(self.mlir_module.body):
            fmain = func.FuncOp(
                name=entry_function_name,
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
            self.local_functions[entry_function_name] = fmain
        # Populate the entry point
        with InsertionPoint(fmain.add_entry_block()), self.loc as loc:
            function_type = measured_function.type
            # Fill the parameters with random values
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
            # Fill the output with random values
            for oty in function_type.results:
                v = 0 if IntegerType.isinstance(oty.element_type) else 0.0
                scal = arith.ConstantOp(oty.element_type, v)
                mem = memref.AllocOp(oty, [], [])
                linalg.fill(scal, outs=[mem])
                inputs.append(mem)
            # Execute and print the execution time
            callrtclock1 = func.CallOp(ext_rtclock, [])
            func.CallOp(measured_function, inputs)
            callrtclock2 = func.CallOp(ext_rtclock, [])
            time = arith.SubFOp(callrtclock2, callrtclock1)
            func.CallOp(ext_printF64, [time])
            # Dealloc
            for i in inputs:
                memref.DeallocOp(i)

            func.ReturnOp([], loc=self.loc)
        return fmain
