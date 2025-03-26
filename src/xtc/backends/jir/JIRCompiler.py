#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
from pathlib import Path
from copy import deepcopy
import tempfile
import subprocess

from xdsl.dialects.builtin import ModuleOp, StringAttr
from jir.environment import get_host_target_triple
from jir.node import JIROp, JIRFunction
from jir.parser import JIRParser
from jir.transform.util.index import JIRFunctionDimensionIndex
from jir.context import JIRFunctionContext
from jir.backend.target import JIRBackendTargetProperties
from jir.backend.xdsl.translator import JIR2XDSLFunctionTranslator
from jir.backend.xdsl.computation import JIRComputationFunctionCallProviderForXDSL
from jir.backend.xdsl.compiler import (
    MLIRLowering,
    MLIR2LLVMConversion,
    LLVMSharedLibraryCompiler,
)
from jir.backend.util.merge_mlir_modules import merge_mlir_modules_by_content
from jir.transform.primitives.canonicalize import canonicalize
import jir.transform.command as command

from xtc.targets.host import HostModule

import xtc.backends.jir as backend
import xtc.itf as itf

from xtc.utils.tools import (
    get_mlir_prefix,
)
from xtc.utils.ext_tools import (
    cc_bin,
    llc_opts,
    opt_opts,
    runtime_libs,
    shared_lib_opts,
)

from .JIRScheduler import JIRSchedule

__all__ = [
    "JIRCompiler",
]


COMMANDS = [
    command.JIRWritebackBufferCommandClass,
    command.JIRComplementaryCommandClass,
    command.JIRDistributeCommandClass,
    command.JIRFuseCommandClass,
    command.JIRInterchangeCommandClass,
    command.JIRSplitLoopIterationDimensionCommandClass,
    command.JIRSubdimCommandClass,
    command.JIRTileCommandClass,
    command.JIRUpdateLoopPropsCommandClass,
    command.JIRWrapLoopCommandClass,
    command.JIRDropLoopCommandClass,
    command.JIRCanonicalizeCommandClass,
]

COMMAND_INDEX = {cmd.command: cmd for cmd in COMMANDS}


class JIRCompiler(itf.comp.Compiler):
    def __init__(
        self,
        backend: "backend.JIRBackend",
        **kwargs: Any,
    ) -> None:
        self._backend = backend
        self.source_op = self._backend.source_op
        self.dims = self._backend.dims
        self.payload_name = self._backend.payload_name
        self._op_function_str = self._backend._op_function_str
        self._jir_function_str = self._backend._jir_function_str
        # TODO? deepcopy?
        self._op_function_mlir = self._backend._op_function_mlir
        self._jir_function_op = self._backend._jir_function_op
        self.save_temps = kwargs.get("save_temps", False)
        self.save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        self.bare_ptr = True
        self.dump_file = kwargs.get("dump_file")
        assert self.dump_file is not None, f"must pass the dump_file name"
        self.print_source_ir = kwargs.get("print_source_ir", False)
        self.print_transformed_ir = kwargs.get("print_transformed_ir", False)
        self.print_assembly = kwargs.get("print_assembly", False)
        self.color = kwargs.get("color", False)
        self.shared_lib = kwargs.get("shared_lib", False)
        self.executable = kwargs.get("executable", False)
        assert not self.executable, f"executable generation not supported yet for TVM"
        assert self.shared_lib, f"shared_lib generation is mandatory for TVM"
        self.mlir_install_dir = get_mlir_prefix()
        self._jir_llvm_config = f"{self.mlir_install_dir}/bin/llvm-config"
        self._target_triple = kwargs.get(
            "target_triple", get_host_target_triple(self._jir_llvm_config)
        )
        self._target_arch = kwargs.get("target_arch", "native")
        self._target_cpu = kwargs.get("target_cpu", "native")
        self.jir_dims = {
            k: v
            for k, v in zip(
                self.source_op.args_names[: len(self.dims)],
                self.source_op.args[: len(self.dims)],
            )
        }
        self._vectors_size = kwargs.get("vector_size", 16)

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def compile(self, schedule: itf.schd.Schedule) -> itf.comp.Module:
        assert isinstance(schedule, JIRSchedule)
        assert self.dump_file is not None
        save_temp = self._save_temp
        source_op = self._backend.source_op
        func_name = source_op.name

        dump_base = Path(self.dump_file).name
        lib_path = self.dump_file

        mlir_lowering = MLIRLowering(f"{self.mlir_install_dir}/bin/mlir-opt")
        mlir2llvm = MLIR2LLVMConversion(f"{self.mlir_install_dir}/bin/mlir-translate")
        llvm_compiler = LLVMSharedLibraryCompiler(
            f"{self.mlir_install_dir}/bin/clang",
            f"{self.mlir_install_dir}/lib",
            None,
            self._target_triple,
            self._target_arch,
        )
        save_temp(f"{dump_base}.jir", str(self._jir_function_str))
        module = self._compile_jir_module(
            self._jir_function_op,
            schedule,
            dump_file=lib_path,
        )
        save_temp(f"{dump_base}.polygeist.c", self._op_function_str)
        computation_primitives = self._op_function_mlir
        save_temp(f"{dump_base}.op.mlir", str(computation_primitives))
        computation_module = str(
            merge_mlir_modules_by_content(str(module), str(computation_primitives))
        )
        save_temp(f"{dump_base}.merged.mlir", computation_module)
        lowered_computation_module = mlir_lowering(computation_module)
        save_temp(f"{dump_base}.lowered.mlir", lowered_computation_module)
        llvm_computation_module = mlir2llvm(lowered_computation_module)
        save_temp(f"{dump_base}.lowered.ll", llvm_computation_module)
        compiled_bc = self._opt_compiler(llvm_computation_module, pic=self.shared_lib)
        compiled_obj = self._llc_compiler(compiled_bc, pic=self.shared_lib)

        compiled_so = self._shlib_compiler(compiled_obj)
        library_path = f"{lib_path}.so"
        with open(library_path, "wb") as out:
            out.write(compiled_so)

        return HostModule(
            dump_base,
            func_name,
            f"{lib_path}.so",
            "shlib",
            bare_ptr=self.bare_ptr,
            np_inputs_spec=source_op.np_inputs_spec,
            np_outputs_spec=source_op.np_outputs_spec,
            reference_impl=source_op.reference_impl,
        )

    def _compile_jir_module(
        self,
        jir_function_op: JIRFunction,
        schedule: JIRSchedule,
        dump_file: str,
    ) -> ModuleOp:
        dump_base = Path(dump_file).name
        save_temp = self._save_temp

        transform_cmds, transform_dims = schedule.get_schedule_impl()
        transform_dims_str = "".join([f"{k}={v}\n" for k, v in transform_dims.items()])
        transform_cmds_str = "".join([f"{t};\n" for t in transform_cmds])
        save_temp(f"{dump_base}.dims", transform_dims_str)
        save_temp(f"{dump_base}.tjir", transform_cmds_str)
        transformed_function_op = self._transform_function(
            jir_function_op, transform_cmds_str, transform_dims
        )
        save_temp(f"{dump_base}.transformed.jir", str(transformed_function_op))
        index = JIRFunctionDimensionIndex()
        ctx = JIRFunctionContext(transformed_function_op)
        index(transformed_function_op)
        for dimension, size in transform_dims.items():
            ctx.define_dimension(index.index[dimension], int(size))
        if not ctx.well_defined:
            raise RuntimeError("Some ctx dimensions are missing")
        module = self._generate_module_for(ctx)
        save_temp(f"{dump_base}.module.mlir", str(module))
        return module

    @property
    def _cmd_cc(self):
        return [cc_bin]

    @property
    def _shared_libs(self):
        return [f"{self.mlir_install_dir}/lib/{lib}" for lib in runtime_libs]

    @property
    def _shared_path(self):
        return [f"-Wl,--rpath={self.mlir_install_dir}/lib/"]

    @property
    def _cmd_opt(self):
        opt = [f"{self.mlir_install_dir}/bin/opt"]
        arch_opts = [f"-march={self._target_arch}", f"--mcpu={self._target_cpu}"]
        return opt + opt_opts + arch_opts

    @property
    def _cmd_llc(self):
        llc = [f"{self.mlir_install_dir}/bin/llc"]
        if self._target_arch == "native":
            arch_opts = [f"--mcpu={self._target_cpu}"]
        else:
            arch_opts = [f"-march={self._target_arch}", f"--mcpu={self._target_cpu}"]
        return llc + llc_opts + arch_opts

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

    def _save_temp(self, fname: str, content: str) -> None:
        if not self.save_temps:
            return
        Path(self.save_temps_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    def _transform_function(
        self,
        jir_function_op: JIROp,
        transform_sequence: str,
        transform_dims: dict[str, int],
    ) -> JIRFunction:
        # We modify the input jir function: copy first
        transformed_function = deepcopy(jir_function_op)
        transform_seq = JIRParser().parse_transform_sequence(transform_sequence)
        for cmd in transform_seq:
            if cmd.command not in COMMAND_INDEX:
                raise RuntimeError(f"Unknown command {cmd.command}")
            transformed_function = COMMAND_INDEX[cmd.command].run(
                cmd, transformed_function
            )
        transformed_function = canonicalize(transformed_function)
        return transformed_function

    def _opt_compiler(self, llvm_compilation_module: str, **kwargs: Any) -> bytes:
        pic = kwargs.get("pic", False)
        opt_pic = ["--relocation-model=pic"] if pic else []
        with tempfile.TemporaryDirectory() as tdir:
            input_ll = f"{tdir}/input.ll"
            with open(input_ll, "w") as outf:
                outf.write(llvm_compilation_module)
            temp_bc = f"{tdir}/output.bc"
            opt_cmd = self._cmd_opt + opt_pic + [input_ll, "-o", temp_bc]
            proc = subprocess.run(
                args=opt_cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Failed to compile LLVM IR with opt: "
                    f"{' '.join(opt_cmd)}\n{proc.stderr}"
                )
            with open(temp_bc, "rb") as inf_bc:
                result_bc = inf_bc.read()
        return result_bc

    def _llc_compiler(self, llvm_bc_module: bytes, **kwargs: Any) -> bytes:
        pic = kwargs.get("pic", False)
        opt_pic = ["--relocation-model=pic"] if pic else []
        with tempfile.TemporaryDirectory() as tdir:
            input_bc = f"{tdir}/input.bc"
            with open(input_bc, "wb") as outf:
                outf.write(llvm_bc_module)
            temp_obj = f"{tdir}/output.o"
            llc_cmd = self._cmd_llc + opt_pic + [input_bc, "-o", temp_obj]
            proc = subprocess.run(
                args=llc_cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Failed to compile LLVM IR with opt: "
                    f"{' '.join(llc_cmd)}\n{proc.stderr}"
                )
            with open(temp_obj, "rb") as inf_obj:
                result_obj = inf_obj.read()
        return result_obj

    def _shlib_compiler(self, obj_module: bytes, **kwargs: Any) -> bytes:
        with tempfile.TemporaryDirectory() as tdir:
            input_obj = f"{tdir}/input.o"
            with open(input_obj, "wb") as outf:
                outf.write(obj_module)
            temp_so = f"{tdir}/output.so"
            shlib_cmd = [
                *self._cmd_cc,
                *shared_lib_opts,
                input_obj,
                "-o",
                temp_so,
                *self._shared_libs,
                *self._shared_path,
            ]
            proc = subprocess.run(
                args=shlib_cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Failed to compile LLVM IR with opt: "
                    f"{' '.join(shlib_cmd)}\n{proc.stderr}"
                )
            with open(temp_so, "rb") as inf_so:
                result_so = inf_so.read()
        return result_so
