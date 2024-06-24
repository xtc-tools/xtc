#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from functools import partial
import numpy as np

from xdsl.dialects import func as xdslfunc

from mlir.dialects import arith, transform
from mlir.dialects.transform import NamedSequenceOp
from mlir.passmanager import PassManager

import utils
from evaluator import Evaluator, Executor
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

from MlirModule import RawMlirModule


class MlirCompiler:
    def __init__(
        self,
        mlir_module: RawMlirModule,
        mlir_install_dir: str | None = None,
        to_disassemble: str | None = None,
    ):
        self.mlir_module = mlir_module
        self.mlir_install_dir = utils.get_mlir_prefix(mlir_install_dir)
        self.to_disassemble = to_disassemble

    @property
    def cmd_cc(self):
        return [cc_bin]

    @property
    def cmd_opt(self):
        return [f"{self.mlir_install_dir}/bin/opt"] + opt_opts

    @property
    def cmd_llc(self):
        return [f"{self.mlir_install_dir}/bin/llc"] + llc_opts

    @property
    def cmd_mlirtranslate(self):
        return [f"{self.mlir_install_dir}/bin/mlir-translate"] + mlirtranslate_opts

    @property
    def cmd_run_mlir(self):
        return [
            f"{self.mlir_install_dir}/bin/mlir-cpu-runner",
            *[f"-shared-libs={lib}" for lib in self.shared_libs],
        ] + mlirrunner_opts

    @property
    def shared_libs(self):
        return [f"{self.mlir_install_dir}/lib/{lib}" for lib in runtime_libs]

    @property
    def shared_path(self):
        return [f"-Wl,--rpath={self.mlir_install_dir}/lib/"]

    @property
    def payload_name(self):
        return self.mlir_module.payload_name

    @property
    def disassemble_option(self):
        if self.to_disassemble is None:
            return "--disassemble"
        else:
            return f"--disassemble={self.to_disassemble}"

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
        print(str(self.mlir_module.mlir_module), file=sys.stderr)

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
        pm = PassManager("builtin.module", context=self.mlir_module.mlir_context)
        for opt in transform_opts:
            pm.add(opt)
        pm.run(self.mlir_module.mlir_module)
        lop = [o for o in self.mlir_module.mlir_module.body.operations][-1]
        assert isinstance(lop, NamedSequenceOp)
        lop.erase()
        if print_transformed_ir:
            self.dump_ir("IR Dump After transform")
        pm = PassManager("builtin.module", context=self.mlir_module.mlir_context)
        for opt in lowering_opts:
            pm.add(opt)
        pm.run(self.mlir_module.mlir_module)

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
        symbol = [f"{self.disassemble_option}"]
        disassemble_cmd = [objdump_bin] + objdump_opts + symbol + disassemble_extra_opts
        print(" ".join(disassemble_cmd))
        dis_process = self.execute_command(
            cmd=disassemble_cmd, pipe_stdoutput=False, debug=debug
        )

    def execute_command(
        self,
        cmd: list[str],
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
        print_source_ir: bool = False,
        print_transformed_ir: bool = False,
        print_assembly: bool = False,
        color: bool = True,
        dump_file: str = dump_file,
        debug: bool = False,
        print_lowered_ir: bool = False,
    ):
        exe_dump_file = f"{dump_file}.o"
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
            cmd=cmd_run, input_pipe=str(self.mlir_module.mlir_module), debug=debug
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
        color: bool = True,
    ):
        return str(self.mlir_module)

    @classmethod
    def _save_temp(
        cls, save_temps: bool, save_temps_dir: str, fname: str, content: str
    ) -> None:
        if not save_temps:
            return
        os.makedirs(save_temps_dir, exist_ok=True)
        with open(f"{save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    def compile(
        self,
        print_source_ir: bool = False,
        print_transformed_ir: bool = False,
        print_assembly: bool = False,
        color: bool = True,
        dump_file: str = dump_file,
        debug: bool = False,
        print_lowered_ir: bool = False,
        shared_lib: bool = False,
        executable: bool = False,
        **kwargs,
    ):
        save_temps = kwargs.get("save_temps", False)
        save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        save_temp = partial(self._save_temp, save_temps, save_temps_dir)

        dump_base = os.path.basename(dump_file)
        dump_tmp_base = f"{save_temps_dir}/{dump_base}"
        ir_dump_file = f"{dump_tmp_base}.ir"
        bc_dump_file = f"{dump_tmp_base}.bc"
        obj_dump_file = f"{dump_tmp_base}.o"
        exe_c_file = f"{dump_tmp_base}.main.c"
        so_dump_file = f"{dump_file}.so"
        exe_dump_file = f"{dump_file}.out"
        src_ir_dump_file = f"{dump_base}.mlir"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        source_ir_str = str(self.mlir_module.mlir_module)
        save_temp(src_ir_dump_file, source_ir_str)

        self.mlir_compile(
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            color=color,
            debug=debug,
            print_lowered_ir=print_lowered_ir,
        )

        mlir_llvm_ir_str = str(self.mlir_module.mlir_module)
        save_temp(mlir_llvm_dump_file, mlir_llvm_ir_str)

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd, input_pipe=str(self.mlir_module.mlir_module), debug=debug
        )
        opt_pic = ["--relocation-model=pic"] if shared_lib else []
        opt_cmd = self.cmd_opt + opt_pic + [ir_dump_file, "-o", bc_dump_file]
        opt_process = self.execute_command(cmd=opt_cmd, debug=debug)

        llc_cmd = self.cmd_llc + opt_pic + [bc_dump_file, "-o", obj_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd, debug=debug)

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
        print_source_ir: bool = False,
        print_transformed_ir: bool = False,
        print_ir_after: list[str] = [],
        print_ir_before: list[str] = [],
        print_assembly: bool = False,
        color: bool = True,
        debug: bool = False,
        print_lowered_ir: bool = False,
        dump_file: str | None = None,
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
        reference=None,
    ):
        results, code, error = self.load_and_eval(
            dll,
            sym,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            validate=validate,
            parameters=parameters,
            reference=reference,
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
        reference=None,
    ):
        libpath = os.path.abspath(dll)
        with utils.LibLoader(libpath) as lib:
            func = getattr(lib, sym)
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            if parameters is None:
                inputs_spec = self.mlir_module.np_inputs_spec()
                outputs_spec = self.mlir_module.np_outputs_spec()
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [np.empty(**spec) for spec in outputs_spec]
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
                    ref_outputs, [out.numpy() for out in parameters[1]]
                ):
                    if not np.allclose(out_ref, out):
                        return [], 1, "Error in validation: outputs differ"
            eval_func = Evaluator(
                func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
            )
            results = eval_func(*parameters[0], *parameters[1])
        return np.array(results), 0, ""
