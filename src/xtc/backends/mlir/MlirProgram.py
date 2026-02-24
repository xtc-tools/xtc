#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.dialects import func as xdslfunc
from mlir.dialects import func
from mlir.ir import (
    ArrayAttr,
    Diagnostic,
    DictAttr,
    UnitAttr,
    Context,
    Location,
    InsertionPoint,
    Module,
)


class RawMlirProgram:
    def __init__(self, source: str):
        self.ctx = Context()
        self.ctx.attach_diagnostic_handler(self._diagnostic_handler)
        self.ctx.emit_error_diagnostics = True  # type: ignore
        self.loc = Location.unknown(self.ctx)
        self.module = Module.parse(source, context=self.ctx)
        self.mlir_extensions: list[str] = []

    @property
    def mlir_context(self):
        return self.ctx

    @property
    def mlir_module(self):
        return self.module

    @classmethod
    def _diagnostic_handler(cls, diag: Diagnostic) -> bool:
        raise RuntimeError(f"MLIR Error: {diag}")

    def require_extension(self, extension: str, weak: bool = False):
        if extension in self.mlir_extensions:
            return
        # Register the extension
        if extension == "sdist":
            try:
                from mlir_sdist.extras.utils import (
                    register_dialects as sdist_register_dialects,
                )

                sdist_register_dialects(self.ctx)
                self.mlir_extensions.append(extension)
            except ImportError:
                if not weak:
                    raise ImportError("mlir_sdist is not installed but is required")
        else:
            raise ValueError(f"Unknown extension: {extension}")


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

        with self.mlir_context:
            # Insert (or not) the noalias attributes
            new_arg_attrs = []
            if no_alias:
                for arg_attrs in payload_func.arg_attrs:
                    new_dict = {}
                    for i in range(len(arg_attrs)):
                        new_dict[arg_attrs[i].name] = arg_attrs[i].attr
                    new_dict["llvm.noalias"] = UnitAttr.get(context=self.mlir_context)
                    new_arg_attrs.append(
                        DictAttr.get(new_dict, context=self.mlir_context)
                    )
                payload_func.arg_attrs = ArrayAttr.get(
                    new_arg_attrs, context=self.mlir_context
                )

            # Insert the function in the MLIR program
            ip = InsertionPoint.at_block_begin(self.mlir_module.body)
            ip.insert(payload_func)
            name = str(payload_func.name).replace('"', "")
            self.local_functions[str(name)] = payload_func

        return payload_func
