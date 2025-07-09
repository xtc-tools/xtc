#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.dialects import func as xdslfunc
from mlir.dialects import func, builtin
from mlir.ir import ArrayAttr, DictAttr, UnitAttr, Context, Location, InsertionPoint


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
    def __init__(self, xdsl_func: xdslfunc.FuncOp, no_alias: bool) -> None:
        super().__init__("module{}")
        self.local_functions: dict[str, func.FuncOp] = {}
        self.parse_and_add_function(str(xdsl_func), no_alias)
        self.payload_name = str(xdsl_func.sym_name).replace('"', "")

    def parse_and_add_function(
        self,
        function: str,
        no_alias: bool,
    ) -> func.FuncOp:
        # Parse the function to MLIR AST
        payload_func: func.FuncOp = func.FuncOp.parse(
            function, context=self.mlir_context
        )

        # Insert (or not) the noalias attributes
        arg_attrs = []
        if no_alias:
            for _ in payload_func.arguments:
                dict_attr = DictAttr.get(
                    {
                        "llvm.noalias": UnitAttr.get(context=self.mlir_context),
                    },
                    context=self.mlir_context,
                )
                arg_attrs.append(dict_attr)
            payload_func.arg_attrs = ArrayAttr.get(arg_attrs, context=self.mlir_context)

        # Insert the function in the MLIR program
        ip = InsertionPoint.at_block_begin(self.mlir_module.body)
        ip.insert(payload_func)
        name = str(payload_func.name).replace('"', "")
        self.local_functions[str(name)] = payload_func

        return payload_func
