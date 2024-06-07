#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import numpy as np
from typing import Any
from functools import partial
from typing import Optional

import utils
from evaluator import Evaluator, Executor
from ndarray import NDArray

from JIROps import Operation, Operators
from JIRScheduler import JIRSchedulerAdaptor

from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp, StringAttr
from jir.environment import get_host_target_triple
from jir.backend.util.annotate_fastmath import annotate_fastmath
from jir.parser import JIRParser, JIRFormatter
from jir.transform.util.index import JIRFunctionDimensionIndex
from jir.context import JIRFunctionContext
from jir.backend.target import JIRBackendTargetProperties
from jir.backend.xdsl.translator import JIR2XDSLFunctionTranslator
from jir.backend.xdsl.computation import JIRComputationFunctionCallProviderForXDSL
from jir.backend.xdsl.benchmark import jir2xdsl_generate_benchmark_routine
from jir.backend.xdsl.compiler import (
    MLIRLowering,
    MLIR2LLVMConversion,
    LLVMSharedLibraryCompiler,
    PolygeistCompiler,
)
from jir.backend.util.merge_mlir_modules import merge_mlir_modules_by_content
from jir.transform.primitives.canonicalize import canonicalize
from jir.transform.command import (
    JIRTransformCommand,
    JIRWritebackBufferCommandClass,
    JIRDistributeCommandClass,
    JIRFuseCommandClass,
    JIRInterchangeCommandClass,
    JIRSplitLoopIterationDimensionCommandClass,
    JIRSubdimCommandClass,
    JIRTileCommandClass,
    JIRUpdateLoopPropsCommandClass,
    JIRWrapLoopCommandClass,
    JIRDropLoopCommandClass,
    JIRComplementaryCommandClass,
    JIRCanonicalizeCommandClass,
)


__all__ = [
    "Implementer",
]

COMMANDS = [
    JIRWritebackBufferCommandClass,
    JIRComplementaryCommandClass,
    JIRDistributeCommandClass,
    JIRFuseCommandClass,
    JIRInterchangeCommandClass,
    JIRSplitLoopIterationDimensionCommandClass,
    JIRSubdimCommandClass,
    JIRTileCommandClass,
    JIRUpdateLoopPropsCommandClass,
    JIRWrapLoopCommandClass,
    JIRDropLoopCommandClass,
    JIRCanonicalizeCommandClass,
]

COMMAND_INDEX = {cmd.command: cmd for cmd in COMMANDS}


class Implementer:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
        jir_install_dir: str,
        geist_install_dir: str,
        **kwargs,
    ) -> None:
        self.source_op = source_op
        self.args = self.source_op.args
        self.jir_install_dir = jir_install_dir
        self.geist_install_dir = geist_install_dir
        self.dims = dims
        self.jir_dims = {
            k: v
            for k, v in zip(
                source_op.args_names[: len(dims)], source_op.args[: len(dims)]
            )
        }
        self._op_function_mlir = None
        self._transformed_jir_function = None
        self._transformed_jir_dims = None
        self._transformed_jir_cmds = None
        self.op_function, self.jir_function, self.payload_name = (
            self.source_op.generate()
        )
        self.transformer = JIRSchedulerAdaptor(source_op, self.dims)
        self._jir_llvm_config = f"{jir_install_dir}/bin/llvm-config"
        self._target_triple = kwargs.get("target_triple") or get_host_target_triple(
            self._jir_llvm_config
        )
        self._target_arch = kwargs.get("target_arch") or "native"

    def _get_op_function_mlir(self) -> str:
        if self._op_function_mlir is not None:
            return self._op_function_mlir
        polygeist_compiler = PolygeistCompiler(f"{self.geist_install_dir}/bin/cgeist")
        self._op_function_mlir = annotate_fastmath(polygeist_compiler(self.op_function))
        return self._op_function_mlir

    def _generate_module_for(self, ctx: JIRFunctionContext) -> ModuleOp:
        computations = JIRComputationFunctionCallProviderForXDSL()
        function_translator = JIR2XDSLFunctionTranslator(
            computations, JIRBackendTargetProperties(vector_size=4)
        )
        fn = function_translator(ctx.function, function_ctx=ctx)
        module_attr = dict()
        module_attr["llvm.target_triple"] = StringAttr(self._target_triple)
        return ModuleOp(
            [fn, *computations.function_declarations], attributes=module_attr
        )

    @classmethod
    def _save_temp(
        cls, save_temps: bool, save_temps_dir: Optional[str], fname: str, content: str
    ) -> None:
        if not save_temps:
            return
        os.makedirs(save_temps_dir, exist_ok=True)
        with open(f"{save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    def _transform(
        self, transform_sequence: str, transform_dims: dict[str, int]
    ) -> None:
        dims = {**self.jir_dims, **transform_dims}
        parser = JIRParser()
        formatter = JIRFormatter()
        function = parser(self.jir_function)
        transform_seq = parser.parse_transform_sequence(transform_sequence)
        for cmd in transform_seq:
            if cmd.command not in COMMAND_INDEX:
                raise RuntimeError(f"Unknown command {cmd.command}")
            function = COMMAND_INDEX[cmd.command].run(cmd, function)
        function = canonicalize(function)
        self._transformed_jir_function = function
        self._transformed_jir_dims = dims
        self._transformed_jir_cmds = transform_sequence

    def reset_schedule(self) -> None:
        self._transformed_jir_function = None
        self._transformed_jir_dims = None
        self._transformed_jir_cmds = None
        self.transformer.reset()

    def tile(self, axis: str, tiles: dict[str, int]) -> None:
        self.transformer.tile(axis, tiles)

    def vectorize(self, axes: list[str]) -> None:
        self.transformer.vectorize(axes)

    def parallelize(self, axes: list[str]) -> None:
        self.transformer.parallelize(axes)

    def unroll(self, axes_unroll: dict[str, int]) -> None:
        self.transformer.unroll(axes_unroll)

    def interchange(self, axes_order: list[str]) -> None:
        self.transformer.interchange(axes_order)

    def _compile_jir_module(self, dump_file, **kwargs) -> Any:
        save_temps = kwargs.get("save_temps", False)
        save_temps_dir = kwargs.get("save_temps_dir") or "./save_temps_dir"
        save_temp = partial(self._save_temp, save_temps, save_temps_dir)
        if self._transformed_jir_function is None:
            save_temp(f"{dump_file}.jir", str(self.jir_function))
            transform_cmds, transform_dims = self.transformer.generate_transform()
            transform_dims_str = "".join(
                [f"{k}={v}\n" for k, v in transform_dims.items()]
            )
            transform_cmds_str = "".join([f"{t};\n" for t in transform_cmds])
            save_temp(f"{dump_file}.dims", transform_dims_str)
            save_temp(f"{dump_file}.tjir", transform_cmds_str)
            self._transform(transform_cmds_str, transform_dims)
            save_temp(
                f"{dump_file}.transformed.jir", str(self._transformed_jir_function)
            )
        fn = self._transformed_jir_function
        dims = self._transformed_jir_dims
        index = JIRFunctionDimensionIndex()
        ctx = JIRFunctionContext(fn)
        index(fn)
        for dimension, size in self._transformed_jir_dims.items():
            ctx.define_dimension(index[dimension], int(size))
        if not ctx.well_defined:
            raise RuntimeError("Some ctx dimensions are missing")
        module = self._generate_module_for(ctx)
        save_temp(f"{dump_file}.module.mlir", str(module))
        return module

    def compile(
        self, dump_file=None, debug=False, shared_lib=False, executable=False, **kwargs
    ) -> None:
        assert dump_file is not None, "dump_file must be passes"
        assert not executable, "TODO: executable output not implemented"
        assert shared_lib, "TODO: shared_lib mandatory"
        save_temps = kwargs.get("save_temps", False)
        save_temps_dir = kwargs.get("save_temps_dir") or "./save_temps_dir"
        save_temp = partial(self._save_temp, save_temps, save_temps_dir)
        mlir_lowering = MLIRLowering(f"{self.jir_install_dir}/bin/mlir-opt")
        mlir2llvm = MLIR2LLVMConversion(f"{self.jir_install_dir}/bin/mlir-translate")
        llvm_compiler = LLVMSharedLibraryCompiler(
            f"{self.jir_install_dir}/bin/clang",
            f"{self.jir_install_dir}/lib",
            self._target_triple,
            self._target_arch,
        )
        module = self._compile_jir_module(dump_file=dump_file, **kwargs)
        save_temp(f"{dump_file}.polygeist.c", self.op_function)
        computation_primitives = self._get_op_function_mlir()
        save_temp(f"{dump_file}.op.mlir", str(computation_primitives))
        computation_module = str(
            merge_mlir_modules_by_content(str(module), str(computation_primitives))
        )
        save_temp(f"{dump_file}.merged.mlir", computation_module)
        lowered_computation_module = mlir_lowering(computation_module)
        save_temp(f"{dump_file}.lowered.mlir", computation_module)
        llvm_computation_module = mlir2llvm(lowered_computation_module)
        save_temp(f"{dump_file}.lowered.ll", computation_module)
        compiled_computation_module = llvm_compiler(llvm_computation_module)
        if dump_file is not None:
            library_path = f"{dump_file}.so"
            with open(library_path, "wb") as out:
                out.write(compiled_computation_module)

    def load_and_evaluate(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        parameters=None,
    ):
        results, code, error = self.load_and_eval(
            dll,
            sym,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            validate=validate,
            parameters=parameters,
        )
        if code == 0:
            return min(results)
        else:
            return error

    def load_and_eval(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        init_zero=False,
        parameters=None,
    ):
        libpath = os.path.abspath(dll)
        with utils.LibLoader(libpath) as lib:
            func = getattr(lib, sym)
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            inputs_spec = self.np_inputs_spec()
            outputs_spec = self.np_outputs_spec()
            out_init = np.zeros if init_zero else np.empty
            if parameters is None:
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [out_init(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [np.empty(**spec) for spec in outputs_spec]
                self.reference_impl(*ref_inputs, *ref_outputs)
                exec_func = Executor(func)
                test_outputs = [NDArray(out_init(**spec)) for spec in outputs_spec]
                exec_func(*parameters[0], *test_outputs)
                for out_ref, out in zip(
                    ref_outputs, [out.numpy() for out in test_outputs]
                ):
                    if not np.allclose(out_ref, out):
                        return [], 1, "Error in validation: outputs differ"
            eval_func = Evaluator(
                func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
            )
            results = eval_func(*parameters[0], *parameters[1])
        return np.array(results), 0, ""

    def np_inputs_spec(self):
        operator = self.source_op.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                operator.inputs_dims(*self.args), operator.inputs_types(*self.args)
            )
        ]

    def np_outputs_spec(self):
        operator = self.source_op.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                operator.outputs_dims(*self.args), operator.outputs_types(*self.args)
            )
        ]

    def reference_impl(self, *operands):
        self.source_op.operator.reference_impl(*operands)
