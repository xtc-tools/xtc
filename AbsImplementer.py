#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
from abc import abstractmethod
import subprocess
import sys
import os
import tempfile
from pathlib import Path

import utils
from evaluator import Evaluator, Executor
import xdsl
from xdsl.dialects.builtin import (
    f64,
)

from mlir.ir import *
from mlir.dialects import builtin, func

import transform

transform_opt = "transform-interpreter"
erase_schedule_opt = "test-transform-dialect-erase-schedule"
transform_opts = [
    f"--{transform_opt}",
    "--func-bufferize",
    "--canonicalize",
]

lowering_opts = [
    f"--{erase_schedule_opt}",
    "--lower-affine",
    "--convert-vector-to-scf",
    "--convert-linalg-to-loops",
    "--loop-invariant-code-motion",
    "--buffer-loop-hoisting",
    "--cse",
    "--sccp",
    "--canonicalize",
    "--convert-scf-to-cf",
    "--canonicalize",
    "--cse",
    "--convert-vector-to-llvm=enable-x86vector",
    "--convert-math-to-llvm",
    "--expand-strided-metadata",
    "--lower-affine",
    "--buffer-results-to-out-params",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm=use-bare-ptr-memref-call-conv",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]

mliropt_opts = transform_opts + lowering_opts

mlirtranslate_opts = ["--mlir-to-llvmir"]

llc_opts = ["-O3", "-filetype=obj", "--mcpu=native"]

opt_opts = ["-O3", "--march=native"]

cc_opts = ["-O3", "-march=native"]

shared_lib_opts = ["--shared", *cc_opts]

exe_opts = [*cc_opts]

runtime_libs = [
    "libmlir_runner_utils.so",
    "libmlir_c_runner_utils.so",
]

dump_file = "/tmp/dump"

mlirrunner_opts = [
    "-e",
    "entry",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = "objdump"

cc_bin = "cc"

objdump_opts = ["-dr", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]


class AbsImplementer(ABC):
    count = 0

    def __init__(
        self,
        mlir_install_dir: str,
        vectors_size: int,
        payload_name=None,
    ):
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)
        self.module.operation.attributes["transform.with_named_sequence"] = (
            UnitAttr.get(context=self.ctx)
        )
        #
        self.vectors_size = vectors_size
        #
        self.payload_name = (
            payload_name if payload_name else f"payload{AbsImplementer.count}"
        )
        self.init_payload_name = f"init_{self.payload_name}"
        AbsImplementer.count += 1
        #
        self.mliropt = [f"{mlir_install_dir}/bin/mlir-opt"]
        self.cmd_mliropt = self.mliropt + mliropt_opts
        #
        self.cmd_run_mlir = [
            f"{mlir_install_dir}/bin/mlir-cpu-runner",
            *[f"-shared-libs={mlir_install_dir}/lib/{lib}" for lib in runtime_libs],
        ] + mlirrunner_opts
        #
        self.cmd_mlirtranslate = [
            f"{mlir_install_dir}/bin/mlir-translate"
        ] + mlirtranslate_opts
        #
        self.cmd_llc = [f"{mlir_install_dir}/bin/llc"] + llc_opts
        #
        self.cmd_opt = [f"{mlir_install_dir}/bin/opt"] + opt_opts
        #
        self.cmd_cc = [cc_bin]
        #
        self.shared_libs = [f"{mlir_install_dir}/lib/{lib}" for lib in runtime_libs]
        self.shared_path = list(
            dict.fromkeys(
                [f"-Wl,--rpath={os.path.dirname(lib)}" for lib in self.shared_libs]
            )
        )
        #
        self.cmd_disassembler = (
            [objdump_bin] + objdump_opts + [f"--disassemble={self.payload_name}"]
        )
        #
        self.cmds_history = []

    def build_compile_extra_opts(
        self,
        print_source_ir,
        print_transformed_ir,
        print_ir_after,
        print_ir_before,
        color,
    ):
        compile_extra_opts = []
        if print_source_ir:
            zero_opt = mliropt_opts[0].replace("--", "")
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_opt}")
        if print_transformed_ir:
            compile_extra_opts.append(f"--mlir-print-ir-after={erase_schedule_opt}")
        compile_extra_opts += [f"--mlir-print-ir-after={p}" for p in print_ir_after]
        compile_extra_opts += [f"--mlir-print-ir-before={p}" for p in print_ir_before]
        return compile_extra_opts

    def build_disassemble_extra_opts(self, obj_file, color):
        disassemble_extra_opts = [obj_file]
        if color:
            disassemble_extra_opts += objdump_color_opts
        return disassemble_extra_opts

    def build_run_extra_opts(self, exe_file, print_assembly, color):
        run_extra_opts = []
        if print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={exe_file}",
            ]
        return run_extra_opts

    def mlir_compile(
        self,
        code,
        print_source_ir,
        print_transformed_ir,
        print_ir_after,
        print_ir_before,
        color,
        debug,
        print_lowered_ir,
    ):
        compile_extra_opts = self.build_compile_extra_opts(
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
        )
        compile_cmd = self.cmd_mliropt + compile_extra_opts
        module_llvm = self.execute_command(
            cmd=compile_cmd, input_pipe=code, debug=debug
        )
        if print_lowered_ir:
            print(f"// -----// IR Dump After MLIR Opt //----- //", file=sys.stderr)
            print(module_llvm.stdout, file=sys.stderr)
        return str(module_llvm.stdout)

    def disassemble(self, obj_file, color, debug):
        disassemble_extra_opts = self.build_disassemble_extra_opts(
            obj_file=obj_file, color=color
        )
        disassemble_cmd = self.cmd_disassembler + disassemble_extra_opts
        bc_process = self.execute_command(
            cmd=disassemble_cmd, pipe_stdoutput=False, debug=debug
        )
        return bc_process

    def execute_command(
        self,
        cmd,
        debug,
        input_pipe=None,
        pipe_stdoutput=True,
    ):
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        self.cmds_history.append(pretty_cmd)
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
        print_ir_after=[],
        print_ir_before=[],
        print_assembly=False,
        color=True,
        dump_file=dump_file,
        debug=False,
        print_lowered_ir=False,
    ):
        exe_dump_file = f"{dump_file}.o"

        str_module = self.glue()

        str_module_llvm = self.mlir_compile(
            code=str_module,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
            debug=debug,
            print_lowered_ir=print_lowered_ir,
        )

        run_extra_opts = self.build_run_extra_opts(
            exe_file=exe_dump_file, print_assembly=print_assembly, color=color
        )
        cmd_run = self.cmd_run_mlir + run_extra_opts
        result = self.execute_command(
            cmd=cmd_run, input_pipe=str_module_llvm, debug=debug
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
        str_module = self.glue()
        mlir_process = self.execute_command(
            cmd=self.mliropt, input_pipe=str_module, debug=False
        )
        return str(mlir_process.stdout)

    def compile(
        self,
        print_source_ir=False,
        print_transformed_ir=False,
        print_ir_after=[],
        print_ir_before=[],
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

        source_ir = self.glue()

        str_module_llvm = self.mlir_compile(
            code=source_ir,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
            debug=debug,
            print_lowered_ir=print_lowered_ir,
        )

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd, input_pipe=str_module_llvm, debug=debug
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

    def glue(self):
        # Generate the payload
        ext_rtclock = self.build_rtclock()
        ext_printF64 = self.build_printF64()
        payload_func = self.payload()
        main_func = self.main(ext_rtclock, ext_printF64, payload_func)

        # Generate the schedule
        match_sym_name, str_trans_match = self.uniquely_match()
        sched_sym_name, str_trans_sched = self.materialize_schedule()
        main_name, str_trans_main = transform.build_main(
            matchers_transformers=[(match_sym_name, sched_sym_name)],
            vectors_size=self.vectors_size,
        )
        trans_script = (
            "module attributes {transform.with_named_sequence} {"
            + "\n"
            + str_trans_sched
            + "\n"
            + str_trans_match
            + "\n"
            + str_trans_main
            + "\n"
            + "}"
        )
        trans_match = Module.parse(trans_script, context=self.ctx)
        with InsertionPoint(self.module.body):
            for o in trans_match.body.operations:
                o.operation.clone()

        # Glue
        str_glued = str(self.module)
        return str_glued

    @abstractmethod
    def np_inputs_spec(self):
        pass

    @abstractmethod
    def np_outputs_spec(self):
        pass

    @abstractmethod
    def reference_impl(self, *operands):
        pass

    @abstractmethod
    def payload(self):
        pass

    @abstractmethod
    def uniquely_match(self):
        pass

    @abstractmethod
    def materialize_schedule(self):
        pass

    @abstractmethod
    def main(self):
        pass

    @abstractmethod
    def build_rtclock(self):
        pass

    @abstractmethod
    def build_printF64(self):
        pass
