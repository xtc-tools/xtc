#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import numpy as np
from typing import Any
from functools import partial
from typing import Optional
from copy import deepcopy

import utils
from evaluator import Evaluator, Executor
from ndarray import NDArray

from JIROps import Operation, Operators
from JIRScheduler import JIRSchedulerAdaptor

from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp, StringAttr
from jir.node import JIROp
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
    "Scheduler",
    "Schedule",
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


class Schedule:
    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = deepcopy(scheduler)

    def get_schedule_impl(self) -> Any:
        return self.scheduler.transformer.generate_transform()

    def __str__(self) -> str:
        return str(self.get_schedule_impl())


class Scheduler:
    def __init__(self, impl: "Implementer") -> None:
        self.transformer = JIRSchedulerAdaptor(impl.source_op, impl.dims)

    def implement(self) -> Schedule:
        return Schedule(scheduler=self)

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


class Implementer:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
        jir_install_dir: str,
        geist_install_dir: str,
        vectors_size: int = 16,
        **kwargs,
    ) -> None:
        self.payload_name = source_op.name
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
        self.op_function, self.jir_function = self.source_op.generate()
        self._vectors_size = vectors_size
        self._jir_llvm_config = f"{jir_install_dir}/bin/llvm-config"
        self._target_triple = kwargs.get("target_triple") or get_host_target_triple(
            self._jir_llvm_config
        )
        self._target_arch = kwargs.get("target_arch") or "native"
        self._jir_function_op = None
        self._op_function_mlir = None
        self._jir_parser = JIRParser()
        self._parse_function()
        self._parse_primitives()

    def _parse_function(self) -> None:
        assert self._jir_function_op is None
        self._jir_function_op = self._jir_parser.parse_function(self.jir_function)

    def _parse_primitives(self) -> None:
        assert self._op_function_mlir is None
        polygeist_compiler = PolygeistCompiler(f"{self.geist_install_dir}/bin/cgeist")
        self._op_function_mlir = annotate_fastmath(polygeist_compiler(self.op_function))

    def _generate_module_for(self, ctx: JIRFunctionContext) -> ModuleOp:
        computations = JIRComputationFunctionCallProviderForXDSL()
        function_translator = JIR2XDSLFunctionTranslator(
            computations, JIRBackendTargetProperties(vector_size=self._vectors_size)
        )
        fn = function_translator(ctx.function, function_ctx=ctx)
        module_attr = dict()
        module_attr["llvm.target_triple"] = StringAttr(self._target_triple)
        return ModuleOp(
            [fn, *computations.function_declarations], attributes=module_attr
        )

    @classmethod
    def _save_temp(
        cls, save_temps: bool, save_temps_dir: str, fname: str, content: str
    ) -> None:
        if not save_temps:
            return
        os.makedirs(save_temps_dir, exist_ok=True)
        with open(f"{save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    def _transform_function(
        self,
        jir_function_op: JIROp,
        transform_sequence: str,
        transform_dims: dict[str, int],
    ) -> JIROp:
        transform_seq = self._jir_parser.parse_transform_sequence(transform_sequence)
        transformed_function = deepcopy(jir_function_op)
        for cmd in transform_seq:
            if cmd.command not in COMMAND_INDEX:
                raise RuntimeError(f"Unknown command {cmd.command}")
            transformed_function = COMMAND_INDEX[cmd.command].run(
                cmd, transformed_function
            )
        transformed_function = canonicalize(transformed_function)
        return transformed_function

    def get_scheduler(self) -> Scheduler:
        return Scheduler(self)

    def _compile_jir_module(self, schedule: Schedule, dump_file: str, **kwargs) -> Any:
        save_temps = kwargs.get("save_temps", False)
        save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        save_temp = partial(self._save_temp, save_temps, save_temps_dir)
        dump_base = os.path.basename(dump_file)

        save_temp(f"{dump_base}.jir", str(self.jir_function))
        transform_cmds, transform_dims = schedule.get_schedule_impl()
        transform_dims_str = "".join([f"{k}={v}\n" for k, v in transform_dims.items()])
        transform_cmds_str = "".join([f"{t};\n" for t in transform_cmds])
        save_temp(f"{dump_base}.dims", transform_dims_str)
        save_temp(f"{dump_base}.tjir", transform_cmds_str)
        transformed_function_op = self._transform_function(
            self._jir_function_op, transform_cmds_str, transform_dims
        )
        save_temp(f"{dump_base}.transformed.jir", str(transformed_function_op))
        index = JIRFunctionDimensionIndex()
        ctx = JIRFunctionContext(transformed_function_op)
        index(transformed_function_op)
        for dimension, size in transform_dims.items():
            ctx.define_dimension(index[dimension], int(size))
        if not ctx.well_defined:
            raise RuntimeError("Some ctx dimensions are missing")
        module = self._generate_module_for(ctx)
        save_temp(f"{dump_base}.module.mlir", str(module))
        return module

    def compile(
        self,
        schedule,
        dump_file=None,
        debug=False,
        shared_lib=False,
        executable=False,
        **kwargs,
    ) -> None:
        assert dump_file is not None, "dump_file must be passes"
        assert not executable, "TODO: executable output not implemented"
        assert shared_lib, "TODO: shared_lib mandatory"
        save_temps = kwargs.get("save_temps", False)
        save_temps_dir = kwargs.get("save_temps_dir") or "./save_temps_dir"
        save_temp = partial(self._save_temp, save_temps, save_temps_dir)
        dump_base = os.path.basename(dump_file)

        mlir_lowering = MLIRLowering(f"{self.jir_install_dir}/bin/mlir-opt")
        mlir2llvm = MLIR2LLVMConversion(f"{self.jir_install_dir}/bin/mlir-translate")
        llvm_compiler = LLVMSharedLibraryCompiler(
            f"{self.jir_install_dir}/bin/clang",
            f"{self.jir_install_dir}/lib",
            self._target_triple,
            self._target_arch,
        )
        module = self._compile_jir_module(schedule, dump_file=dump_file, **kwargs)
        save_temp(f"{dump_base}.polygeist.c", self.op_function)
        computation_primitives = self._op_function_mlir
        save_temp(f"{dump_base}.op.mlir", str(computation_primitives))
        computation_module = str(
            merge_mlir_modules_by_content(str(module), str(computation_primitives))
        )
        save_temp(f"{dump_base}.merged.mlir", computation_module)
        lowered_computation_module = mlir_lowering(computation_module)
        save_temp(f"{dump_base}.lowered.mlir", computation_module)
        llvm_computation_module = mlir2llvm(lowered_computation_module)
        save_temp(f"{dump_base}.lowered.ll", computation_module)
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
        init_zero=False,
        parameters=None,
        reference=None,
        **kwargs,
    ):
        results, code, error = self.load_and_eval(
            dll,
            sym,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            validate=validate,
            init_zero=init_zero,
            parameters=parameters,
            reference=reference,
            **kwargs,
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
        reference=None,
        **kwargs,
    ):
        libpath = os.path.abspath(dll)
        with utils.LibLoader(libpath) as lib:
            func = getattr(lib, sym)
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            if parameters is None:
                inputs_spec = self.np_inputs_spec()
                outputs_spec = self.np_outputs_spec()
                out_init = np.zeros if init_zero else np.empty
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [out_init(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [
                    np.empty(shape=out.shape, dtype=out.dtype) for out in parameters[1]
                ]
                if reference is None:
                    reference = self.mlir_module.reference_impl
                reference(*ref_inputs, *ref_outputs)
                exec_func = Executor(func)
                exec_func(*parameters[0], *parameters[1])
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
