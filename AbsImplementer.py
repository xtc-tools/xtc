#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
from abc import abstractmethod
import subprocess

import xdsl
from xdsl.dialects.builtin import (
    f64,
)

from mlir.ir import *
from mlir.dialects import builtin, func

import transform

transform_opt = "transform-interpreter"
transform_opts = [
    f"--{transform_opt}",
    "--canonicalize",
    "--cse",
    "--sccp",
]

lowering_opts = [
    "--test-transform-dialect-erase-schedule",
    "--func-bufferize",
    "--convert-vector-to-scf",
    "--convert-linalg-to-loops",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--canonicalize",
    "--cse",
    "--convert-vector-to-llvm",
    "--convert-math-to-llvm",
    "--expand-strided-metadata",
    "--lower-affine",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]

mliropt_opts = transform_opts + lowering_opts

mlirtranslate_opts = ["--mlir-to-llvmir"]

llc_opts = ["-O3", "-filetype=obj"]

opt_opts = ["-O3"]

dump_file = "/tmp/dump"

mlirrunner_opts = [
    "-e",
    "main",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = "objdump"

objdump_opts = ["-d", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]


class AbsImplementer(ABC):
    count = 0

    def __init__(
        self,
        mlir_install_dir: str,
    ):
        #
        self.payload_name = f"payload{AbsImplementer.count}"
        self.init_payload_name = f"init_{self.payload_name}"
        AbsImplementer.count += 1
        #
        self.mliropt = [f"{mlir_install_dir}/bin/mlir-opt"]
        self.cmd_mliropt = self.mliropt + mliropt_opts
        #
        self.cmd_run_mlir = [
            f"{mlir_install_dir}/bin/mlir-cpu-runner",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_runner_utils.so",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_c_runner_utils.so",
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
            zero_lowering_opt = lowering_opts[0].replace("--", "")
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_lowering_opt}")
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
        print_source_ir=False,
        print_transformed_ir=False,
        print_ir_after=[],
        print_ir_before=[],
        color=True,
    ):
        compile_extra_opts = self.build_compile_extra_opts(
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
        )
        compile_cmd = self.cmd_mliropt + compile_extra_opts
        module_llvm = self.execute_command(cmd=compile_cmd, input_pipe=code)

        return str(module_llvm.stdout)

    def disassemble(self, obj_file, color):
        disassemble_extra_opts = self.build_disassemble_extra_opts(
            obj_file=obj_file, color=color
        )
        disassemble_cmd = self.cmd_disassembler + disassemble_extra_opts
        bc_process = self.execute_command(cmd=disassemble_cmd, pipe_stdoutput=False)
        return bc_process

    def execute_command(self, cmd, input_pipe=None, pipe_stdoutput=True):
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
        self.cmds_history.append(" ".join(cmd))
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
    ):
        exe_dump_file = f"{dump_file}.out"

        str_module = self.glue()

        str_module_llvm = self.mlir_compile(
            code=str_module,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
        )

        run_extra_opts = self.build_run_extra_opts(
            exe_file=dump_file, print_assembly=print_assembly, color=color
        )
        cmd_run = self.cmd_run_mlir + run_extra_opts
        result = self.execute_command(cmd=cmd_run, input_pipe=str_module_llvm)

        if print_assembly:
            disassemble_process = self.disassemble(obj_file=dump_file, color=color)

        return result.stdout

    def generate_without_compilation(
        self,
        color=True,
    ):
        str_module = self.glue()
        mlir_process = self.execute_command(cmd=self.mliropt, input_pipe=str_module)
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
    ):
        ir_dump_file = f"{dump_file}.ir"
        bc_dump_file = f"{dump_file}.bc"
        exe_dump_file = f"{dump_file}.out"

        str_module = self.glue()

        str_module_llvm = self.mlir_compile(
            code=str_module,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_ir_after=print_ir_after,
            print_ir_before=print_ir_before,
            color=color,
        )

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd, input_pipe=str_module_llvm
        )

        opt_cmd = self.cmd_opt + [ir_dump_file, "-o", bc_dump_file]
        bc_process = self.execute_command(cmd=opt_cmd)

        llc_cmd = self.cmd_llc + [bc_dump_file, "-o", exe_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd)

        if print_assembly:
            disassemble_process = self.disassemble(exe_file=exe_dump_file, color=color)

    def glue(self):
        ext_rtclock = self.build_rtclock()
        ext_printF64 = self.build_printF64()
        payload_func = self.payload()
        init_func = self.init_payload()
        main_func = self.main(ext_rtclock, ext_printF64, payload_func, init_func)
        # Glue the module
        # mod = ModuleOp([ext_rtclock,ext_printF64,payload_func,main_func])
        # str_mod = str(mod)
        str_mod = "\n".join(
            [
                str(tl)
                for tl in [
                    ext_rtclock,
                    ext_printF64,
                    init_func,
                    payload_func,
                    main_func,
                ]
            ]
        )
        match_sym_name, str_trans_match = self.uniquely_match()

        sched_sym_name, str_trans_sched = self.materialize_schedule()

        main_name, str_trans_main = transform.build_main(
            [(match_sym_name, sched_sym_name)]
        )
        str_glued = (
            "module attributes {transform.with_named_sequence} {"
            + "\n"
            + str_mod
            + "\n"
            + str_trans_sched
            + "\n"
            + str_trans_match
            + "\n"
            + str_trans_main
            + "\n"
            + "}"
        )

        return str_glued

    @abstractmethod
    def payload(self, m, elt_type):
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
