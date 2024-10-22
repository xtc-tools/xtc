#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
import subprocess
import sys
import os
import tempfile
from pathlib import Path
import numpy as np

from xdsl.dialects import func as xdslfunc

from mlir.dialects.transform import NamedSequenceOp
from mlir.passmanager import PassManager

import utils
from evaluator import Evaluator, Executor
from MlirModule import MlirModule
from ext_tools import (
    transform_opts,
    lowering_opts,
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    dump_file,
    mlirrunner_opts,
    objdump_bin,
    cc_bin,
    objdump_opts,
    objdump_color_opts,
)
import transform


# class MlirImplementer:
class MlirImplementer(ABC):
    def __init__(
        self,
        mlir_install_dir: str,
        xdsl_func: xdslfunc.FuncOp,
        vectors_size: int,
    ):
        # Compilation information
        self.vectors_size = vectors_size
        self.shared_libs = [f"{mlir_install_dir}/lib/{lib}" for lib in runtime_libs]
        self.shared_path = [f"-Wl,--rpath={mlir_install_dir}/lib/"]
        self.cmd_run_mlir = [
            f"{mlir_install_dir}/bin/mlir-cpu-runner",
            *[f"-shared-libs={lib}" for lib in self.shared_libs],
        ] + mlirrunner_opts
        self.cmd_mlirtranslate = [
            f"{mlir_install_dir}/bin/mlir-translate"
        ] + mlirtranslate_opts
        self.cmd_llc = [f"{mlir_install_dir}/bin/llc"] + llc_opts
        self.cmd_opt = [f"{mlir_install_dir}/bin/opt"] + opt_opts
        self.cmd_cc = [cc_bin]
        # Module definition
        self._mlir_module = MlirModule()
        payload_func = self._mlir_module.parse_and_add_function(str(xdsl_func))
        self.payload_name = str(payload_func.name).replace('"', "")
        self._mlir_module.measure_execution_time(
            new_function_name="entry",
            measured_function_name=self.payload_name,
        )

    @property
    def ctx(self):
        return self._mlir_module.ctx

    @property
    def module(self):
        return self._mlir_module.module

    def inject_schedule(self):
        sym_name = "@__transform_main"
        myvar = transform.get_new_var()
        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_var=myvar, sym_name=sym_name
        )
        kernel = self.schedule_kernel(signature=seq_sig, input_var=input_var)
        self._mlir_module.inject_schedule(kernel)

    @abstractmethod
    def schedule_kernel(
        self,
        signature: str,
        input_var: str,
    ) -> list[str]:
        pass

    def build_disassemble_extra_opts(
        self,
        obj_file: str,
        color: bool,
    ) -> list[str]:
        disassemble_extra_opts = [obj_file]
        if color:
            disassemble_extra_opts += objdump_color_opts
        return disassemble_extra_opts

    def build_run_extra_opts(
        self, exe_file: str, print_assembly: bool, color: bool
    ) -> list[str]:
        run_extra_opts: list[str] = []
        if print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={exe_file}",
            ]
        return run_extra_opts

    def dump_ir(self, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(self.module), file=sys.stderr)

    def mlir_compile(
        self,
        print_source_ir: bool,
        print_transformed_ir: bool,
        color: bool,
        debug: bool,
        print_lowered_ir: bool,
    ):
        if print_source_ir:
            self.dump_ir("IR Dump Before transform")
        pm = PassManager("builtin.module", context=self.ctx)
        for opt in transform_opts:
            pm.add(opt)
        pm.run(self.module)
        lop = [o for o in self.module.body.operations][-1]
        assert isinstance(lop, NamedSequenceOp)
        lop.erase()
        if print_transformed_ir:
            self.dump_ir("IR Dump After transform")
        #
        pm = PassManager("builtin.module", context=self.ctx)
        for opt in lowering_opts:
            pm.add(opt)
        pm.run(self.module)

        if print_lowered_ir:
            self.dump_ir("IR Dump After MLIR Opt")

    def disassemble(
        self,
        obj_file: str,
        color: bool,
        debug: bool,
    ):
        disassemble_extra_opts = self.build_disassemble_extra_opts(
            obj_file=obj_file, color=color
        )
        symbol = f"--disassemble={self.payload_name}" if self.payload_name else None
        disassemble_cmd = (
            [objdump_bin] + objdump_opts + [symbol] + disassemble_extra_opts
        )
        print(" ".join(disassemble_cmd))
        bc_process = self.execute_command(
            cmd=disassemble_cmd, pipe_stdoutput=False, debug=debug
        )

    def execute_command(
        self,
        cmd: str,
        debug: bool,
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        if debug:
            print(f"> exec: {pretty_cmd}", file=sys.stderr)

        if input_pipe and pipe_stdoutput:
            result = subprocess.run(
                cmd, input=input_pipe, stdout=subprocess.PIPE, text=True
            )
        elif input_pipe and not pipe_stdoutput:
            result = subprocess.run(cmd, input=input_pipe, text=True)
        elif not input_pipe and pipe_stdoutput:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(cmd, text=True)
        return result

    def evaluate(
        self,
        print_source_ir=False,
        print_transformed_ir=False,
        print_assembly=False,
        color=True,
        dump_file=dump_file,
        debug=False,
        print_lowered_ir=False,
    ):
        exe_dump_file = f"{dump_file}.o"

        self.inject_schedule()
        self.mlir_compile(
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            color=color,
            debug=debug,
            print_lowered_ir=print_lowered_ir,
        )

        run_extra_opts = self.build_run_extra_opts(
            exe_file=exe_dump_file, print_assembly=print_assembly, color=color
        )
        cmd_run = self.cmd_run_mlir + run_extra_opts
        result = self.execute_command(
            cmd=cmd_run, input_pipe=str(self.module), debug=debug
        )

        if print_assembly:
            disassemble_process = self.disassemble(
                obj_file=exe_dump_file,
                color=color,
                debug=debug,
            )

        return result.stdout

    def generate_without_compilation(
        self,
        color=True,
    ):
        return str(self.module)

    def compile(
        self,
        print_source_ir=False,
        print_transformed_ir=False,
        print_assembly=False,
        color=True,
        dump_file=dump_file,
        debug=False,
        print_lowered_ir=False,
        shared_lib=False,
        executable=False,
        **kwargs,
    ):
        save_temps = kwargs.get("save_temps", False)
        ir_dump_file = f"{dump_file}.ir"
        bc_dump_file = f"{dump_file}.bc"
        obj_dump_file = f"{dump_file}.o"
        so_dump_file = f"{dump_file}.so"
        exe_c_file = f"{dump_file}.main.c"
        exe_dump_file = f"{dump_file}.out"

        self.inject_schedule()

        self.mlir_compile(
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            color=color,
            debug=debug,
            print_lowered_ir=print_lowered_ir,
        )

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd, input_pipe=str(self.module), debug=debug
        )

        opt_pic = ["--relocation-model=pic"] if shared_lib else []
        opt_cmd = self.cmd_opt + opt_pic + [ir_dump_file, "-o", bc_dump_file]
        bc_process = self.execute_command(cmd=opt_cmd, debug=debug)

        llc_cmd = self.cmd_llc + opt_pic + [bc_dump_file, "-o", obj_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd, debug=debug)

        # clang_cmd = self.cmd_clang + ['-c', ir_dump_file, '-o', exe_dump_file]
        # bc_process = self.execute_command(cmd=clang_cmd, debug = debug)

        if print_assembly:
            disassemble_process = self.disassemble(
                obj_file=obj_dump_file, color=color, debug=debug
            )

        payload_objs = [obj_dump_file, *self.shared_libs]
        payload_path = [*self.shared_path]
        if shared_lib:
            shared_cmd = [
                *self.cmd_cc,
                *shared_lib_opts,
                obj_dump_file,
                "-o",
                so_dump_file,
                *self.shared_libs,
                *self.shared_path,
            ]
            self.execute_command(cmd=shared_cmd, debug=debug)
            payload_objs = [so_dump_file]
            payload_path = ["-Wl,--rpath=${ORIGIN}"]

        if executable:
            exe_cmd = [
                *self.cmd_cc,
                *exe_opts,
                exe_c_file,
                "-o",
                exe_dump_file,
                *payload_objs,
                *payload_path,
            ]
            with open(exe_c_file, "w") as outf:
                outf.write("extern void entry(void); int main() { entry(); return 0; }")
            self.execute_command(cmd=exe_cmd, debug=debug)

        if not save_temps:
            Path(ir_dump_file).unlink(missing_ok=True)
            Path(bc_dump_file).unlink(missing_ok=True)
            Path(obj_dump_file).unlink(missing_ok=True)
            Path(exe_c_file).unlink(missing_ok=True)

    def compile_and_evaluate(
        self,
        print_source_ir=False,
        print_transformed_ir=False,
        print_ir_after=[],
        print_ir_before=[],
        print_assembly=False,
        color=True,
        debug=False,
        print_lowered_ir=False,
        dump_file=None,
    ):
        with tempfile.TemporaryDirectory() as tdir:
            if dump_file is None:
                dump_file = f"{tdir}/payload"
            self.compile(
                print_source_ir=print_source_ir,
                print_transformed_ir=print_transformed_ir,
                print_ir_after=print_ir_after,
                print_ir_before=print_ir_before,
                print_assembly=print_assembly,
                color=color,
                debug=debug,
                print_lowered_ir=print_lowered_ir,
                dump_file=dump_file,
                shared_lib=True,
                executable=True,
            )
            exe_file = os.path.abspath(f"{dump_file}.out")
            result = self.execute_command(
                cmd=[f"{exe_file}"],
                debug=debug,
            )
            return result.stdout

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
        parameters=None,
    ):
        libpath = os.path.abspath(dll)
        with utils.LibLoader(libpath) as lib:
            func = getattr(lib, sym)
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            inputs_spec = self.np_inputs_spec()
            outputs_spec = self.np_outputs_spec()
            if parameters is None:
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [np.empty(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [np.empty(**spec) for spec in outputs_spec]
                self.reference_impl(*ref_inputs, *ref_outputs)
                exec_func = Executor(func)
                exec_func(*parameters[0], *parameters[1])
                for out_ref, out in zip(
                    ref_outputs, [out.numpy() for out in parameters[1]]
                ):
                    if not np.allclose(out_ref, out):
                        return [], 1, "Error in validation: outputs differ"
            eval_func = Evaluator(
                func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
            )
            results = eval_func(*parameters[0], *parameters[1])
        return np.array(results), 0, ""

    @abstractmethod
    def np_inputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        pass

    @abstractmethod
    def np_outputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        pass

    @abstractmethod
    def reference_impl(self, *operands):
        pass
