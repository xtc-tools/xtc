#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any, cast
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path


from xtc.utils.tools import (
    get_mlir_prefix,
)

from xtc.utils.ext_tools import (
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    system_libs,
    mlirrunner_opts,
    objdump_bin,
    objdump_arm_bin,
    cc_bin,
    objdump_opts,
    objdump_color_opts,
)

from xtc.targets.host import HostModule

import xtc.backends.mlir as backend
import xtc.itf as itf

from .MlirProgram import MlirProgram, RawMlirProgram
from .MlirScheduler import MlirSchedule

from .MlirCompilerPasses import (
    MlirProgramInsertTransformPass,
    MlirProgramApplyTransformPass,
    MlirProgramToLLVMDialectPass,
)


class MlirCompiler(itf.comp.Compiler):
    def __init__(
        self,
        backend: "backend.MlirBackend",
        **kwargs: Any,
    ):
        self._backend = backend
        self._compiler_kwargs = kwargs

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def compile(
        self,
        schedule: itf.schd.Schedule,
    ) -> itf.comp.Module:
        shared_lib = self._compiler_kwargs.get("shared_lib", False)
        executable = self._compiler_kwargs.get("executable", False)
        dump_file = self._compiler_kwargs.get("dump_file")
        temp_dir = None
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/{self._backend.payload_name}"
            self._compiler_kwargs["dump_file"] = dump_file
        program = self.generate_program()
        compiler = MlirProgramCompiler(
            mlir_program=program,
            mlir_schedule=cast(MlirSchedule, schedule),
            concluding_passes=self._backend.concluding_passes,
            always_vectorize=self._backend.always_vectorize,
            **self._compiler_kwargs,
        )
        assert compiler.dump_file is not None
        compiler.compile()
        io_specs_args = {}
        if self._backend._graph is None:
            # Pass backend defined inputs/outputs specs when not a Graph
            io_specs_args.update(
                {
                    "np_inputs_spec": self._backend.np_inputs_spec,
                    "np_outputs_spec": self._backend.np_outputs_spec,
                }
            )
        executable = HostModule(
            Path(compiler.dump_file).name,
            self._backend.payload_name,
            f"{compiler.dump_file}.so",
            "shlib",
            bare_ptr=compiler.bare_ptr,
            graph=self._backend._graph,
            **io_specs_args,
        )
        if temp_dir is not None:
            shutil.rmtree(temp_dir)
        return executable

    def generate_program(self) -> RawMlirProgram:
        # xdsl_func input must be read only
        return MlirProgram(self._backend.xdsl_func, self._backend.no_alias)


class MlirProgramCompiler:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
        mlir_schedule: MlirSchedule | None = None,
        **kwargs: Any,
    ):
        self._mlir_program = mlir_program
        self._mlir_schedule = mlir_schedule
        self.mlir_install_dir = get_mlir_prefix(kwargs.get("mlir_install_dir", None))
        self.to_disassemble = kwargs.get("to_disassemble", "")
        self.save_temps = kwargs.get("save_temps", False)
        self.save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        self.bare_ptr = True
        self.print_source_ir = kwargs.get("print_source_ir", False)
        self.print_transformed_ir = kwargs.get("print_transformed_ir", False)
        self.print_assembly = kwargs.get("print_assembly", False)
        self.visualize_jumps = kwargs.get("visualize_jumps", True)
        self.print_lowered_ir = kwargs.get("print_lowered_ir", False)
        self.debug = kwargs.get("debug", False)
        self.color = kwargs.get("color", False)
        self.shared_lib = kwargs.get("shared_lib", False)
        self.executable = kwargs.get("executable", False)
        self.dump_file = kwargs.get("dump_file")
        self.concluding_passes = kwargs.get("concluding_passes", [])
        self.always_vectorize = kwargs.get("always_vectorize", False)
        self.vectors_size = kwargs.get("vectors_size", None)
        self.arch = kwargs.get("arch", "native")
        self.cpu = kwargs.get("cpu", "native")

    @property
    def cmd_cc(self):
        return [cc_bin]

    @property
    def cmd_opt(self):
        opt = [f"{self.mlir_install_dir}/bin/opt"]
        return opt + opt_opts + [f"-march={self.arch}", f"--mcpu={self.cpu}"]

    @property
    def cmd_llc(self):
        llc = [f"{self.mlir_install_dir}/bin/llc"]
        if self.arch == "native":
            llc_arch = [f"--mcpu={self.cpu}"]
        else:
            llc_arch = [f"-march={self.arch}", f"--mcpu={self.cpu}"]
        return llc + llc_opts + llc_arch

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
        return system_libs + [
            f"{self.mlir_install_dir}/lib/{lib}" for lib in runtime_libs
        ]

    @property
    def shared_path(self):
        return [f"-Wl,--rpath={self.mlir_install_dir}/lib/"]

    @property
    def disassemble_option(self):
        if not self.to_disassemble:
            return "--disassemble"
        else:
            return f"--disassemble={self.to_disassemble}"

    def build_disassemble_extra_opts(
        self,
        obj_file: str,
    ) -> list[str]:
        disassemble_extra_opts = [obj_file]
        if self.visualize_jumps:
            disassemble_extra_opts += ["--visualize-jumps"]
        if self.color:
            disassemble_extra_opts += objdump_color_opts
        return disassemble_extra_opts

    def build_run_extra_opts(self, obj_file: str) -> list[str]:
        run_extra_opts: list[str] = []
        if self.print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={obj_file}",
            ]
        return run_extra_opts

    def dump_ir(self, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(self._mlir_program.mlir_module), file=sys.stderr)

    def mlir_insert_transform_pass(self) -> None:
        insert_transform_pass = MlirProgramInsertTransformPass(
            mlir_program=self._mlir_program,
            mlir_schedule=self._mlir_schedule,
            concluding_passes=self.concluding_passes,
            always_vectorize=self.always_vectorize,
            vectors_size=self.vectors_size,
        )
        insert_transform_pass.run()
        if self.print_source_ir:
            self.dump_ir("IR Dump Before transform")

    def mlir_apply_transform_pass(self) -> None:
        apply_transform_pass = MlirProgramApplyTransformPass(
            mlir_program=self._mlir_program,
        )
        apply_transform_pass.run()
        if self.print_transformed_ir:
            self.dump_ir("IR Dump After transform")

    def mlir_to_llvm_pass(self) -> None:
        to_llvm_pass = MlirProgramToLLVMDialectPass(
            mlir_program=self._mlir_program,
        )
        to_llvm_pass.run()
        if self.print_lowered_ir:
            self.dump_ir("IR Dump After MLIR Opt")

    def mlir_compile(self) -> None:
        self.mlir_insert_transform_pass()
        self.mlir_apply_transform_pass()
        self.mlir_to_llvm_pass()

    def disassemble(
        self,
        obj_file: str,
    ) -> subprocess.CompletedProcess:
        disassemble_extra_opts = self.build_disassemble_extra_opts(obj_file=obj_file)
        symbol = [f"{self.disassemble_option}"]
        objdump = objdump_arm_bin if self.arch == "aarch64" else objdump_bin
        disassemble_cmd = [objdump] + objdump_opts + symbol + disassemble_extra_opts
        dis_process = self.execute_command(cmd=disassemble_cmd, pipe_stdoutput=False)
        return dis_process

    def execute_command(
        self,
        cmd: list[str],
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        if self.debug:
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

    def evaluate(self) -> str:
        self.mlir_compile()

        obj_dump_file = f"{self.dump_file}.o"
        run_extra_opts = self.build_run_extra_opts(
            obj_file=obj_dump_file,
        )
        cmd_run = self.cmd_run_mlir + run_extra_opts
        result = self.execute_command(
            cmd=cmd_run, input_pipe=str(self._mlir_program.mlir_module)
        )
        if self.print_assembly:
            disassemble_process = self.disassemble(
                obj_file=obj_dump_file,
            )
            assert disassemble_process.returncode == 0
        return result.stdout

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self.save_temps:
            return
        os.makedirs(self.save_temps_dir, exist_ok=True)
        with open(f"{self.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))

    def compile(self) -> None:
        save_temp = self._save_temp
        save_temps_dir = self.save_temps_dir
        dump_file = self.dump_file
        temp_dir = None
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self.save_temps:
            assert self.dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = save_temps_dir
            os.makedirs(save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent

        dump_base = Path(dump_file).name
        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        ir_dump_file = f"{dump_tmp_file}.ir"
        bc_dump_file = f"{dump_tmp_file}.bc"
        obj_dump_file = f"{dump_tmp_file}.o"
        exe_c_file = f"{dump_tmp_file}.main.c"
        so_dump_file = f"{dump_file}.so"
        exe_dump_file = f"{dump_file}.out"
        src_ir_dump_file = f"{dump_base}.mlir"
        mlir_btrn_dump_file = f"{dump_base}.before_trn.mlir"
        mlir_atrn_dump_file = f"{dump_base}.after_trn.mlir"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        save_temp(src_ir_dump_file, self._mlir_program.mlir_module)

        self.mlir_insert_transform_pass()
        save_temp(mlir_btrn_dump_file, self._mlir_program.mlir_module)

        self.mlir_apply_transform_pass()
        save_temp(mlir_atrn_dump_file, self._mlir_program.mlir_module)

        self.mlir_to_llvm_pass()
        save_temp(mlir_llvm_dump_file, self._mlir_program.mlir_module)

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd,
            input_pipe=str(self._mlir_program.mlir_module),
        )
        assert llvmir_process.returncode == 0

        opt_pic = ["--relocation-model=pic"] if self.shared_lib else []
        opt_cmd = self.cmd_opt + opt_pic + [ir_dump_file, "-o", bc_dump_file]
        opt_process = self.execute_command(cmd=opt_cmd)
        assert opt_process.returncode == 0

        llc_cmd = self.cmd_llc + opt_pic + [bc_dump_file, "-o", obj_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd)
        assert bc_process.returncode == 0

        if self.print_assembly:
            disassemble_process = self.disassemble(obj_file=obj_dump_file)
            assert disassemble_process.returncode == 0

        payload_objs = [obj_dump_file, *self.shared_libs]
        payload_path = [*self.shared_path]
        if self.shared_lib:
            shared_cmd = [
                *self.cmd_cc,
                *shared_lib_opts,
                obj_dump_file,
                "-o",
                so_dump_file,
                *self.shared_libs,
                *self.shared_path,
            ]
            shlib_process = self.execute_command(cmd=shared_cmd)
            assert shlib_process.returncode == 0

            payload_objs = [so_dump_file]
            payload_path = ["-Wl,--rpath=${ORIGIN}"]

        if self.executable:
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
            exe_process = self.execute_command(cmd=exe_cmd)
            assert exe_process.returncode == 0

        if not self.save_temps:
            Path(ir_dump_file).unlink(missing_ok=True)
            Path(bc_dump_file).unlink(missing_ok=True)
            Path(obj_dump_file).unlink(missing_ok=True)
            Path(exe_c_file).unlink(missing_ok=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir)
