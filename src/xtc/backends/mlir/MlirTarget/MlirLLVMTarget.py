#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any
import subprocess
import os
import sys
import tempfile
import shutil
from pathlib import Path

from xtc.utils.host_tools import disassemble

from xtc.utils.ext_tools import (
    get_shlib_extension,
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    system_libs,
    cc_bin,
)
from xtc.utils.host_tools import (
    target_triple,
)

from xtc.targets.host import HostModule
import xtc.itf as itf
from xtc.itf.graph import Graph

from .MlirTarget import MlirTarget
from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

from mlir.passmanager import PassManager
from mlir.ir import OpResult

__all__ = ["MlirLLVMTarget"]


class MlirLLVMTarget(MlirTarget):
    """The default CPU target using llvmir

    This target implements the lowering to llvmir, and run llvm toolchain to generate the     final shared lib or executable for CPU.
    """

    def __init__(self, config: MlirConfig):
        super().__init__(config)

    @override
    def name(self) -> str:
        return "llvm-cpu"

    @override
    def arch(self) -> str:
        return "cpu"

    @override
    def generate_code_for_target(
        self,
        mlir_program: RawMlirProgram,  # Will be modified in place
        **kwargs: Any,
    ) -> None:
        save_temp = self._save_temp
        save_temps_dir = self._config.save_temps_dir
        temp_dir = None
        dump_file = kwargs.get("dump_file", None)
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self._config.save_temps:
            assert dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = Path(save_temps_dir)
            os.makedirs(save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent
        dump_base = Path(dump_file).name

        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        ir_dump_file = f"{dump_tmp_file}.ir"
        bc_dump_file = f"{dump_tmp_file}.bc"
        obj_dump_file = f"{dump_tmp_file}.o"
        exe_c_file = f"{dump_tmp_file}.main.c"
        so_dump_file = f"{dump_file}.{get_shlib_extension()}"
        exe_dump_file = f"{dump_file}.out"
        src_ir_dump_file = f"{dump_base}.mlir"
        mlir_btrn_dump_file = f"{dump_base}.before_trn.mlir"
        mlir_atrn_dump_file = f"{dump_base}.after_trn.mlir"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        # Lower to MLIR LLVM dialect
        self._mlir_to_llvm_pass(mlir_program)

        # Do the rest
        save_temp(mlir_llvm_dump_file, mlir_program.mlir_module)

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd,
            input_pipe=str(mlir_program.mlir_module),
        )
        assert llvmir_process.returncode == 0

        opt_pic = ["--relocation-model=pic"] if self._config.shared_lib else []
        opt_cmd = self.cmd_opt + opt_pic + [ir_dump_file, "-o", bc_dump_file]
        opt_process = self.execute_command(cmd=opt_cmd)
        assert opt_process.returncode == 0

        llc_cmd = self.cmd_llc + opt_pic + [bc_dump_file, "-o", obj_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd)
        assert bc_process.returncode == 0

        if self._config.print_assembly:
            disassembly = disassemble(
                obj_dump_file,
                function=self._config.to_disassemble,
                arch=self._config.arch,
                color=self._config.color,
                visualize_jumps=self._config.visualize_jumps,
            )
            print(disassembly, file=sys.stderr)

        payload_objs = [obj_dump_file, *self.shared_libs]
        payload_path = [*self.shared_path]
        if self._config.shared_lib:
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
            payload_path = ["-Wl,-rpath,$ORIGIN"]

        if self._config.executable:
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

        if not self._config.save_temps:
            Path(ir_dump_file).unlink(missing_ok=True)
            Path(bc_dump_file).unlink(missing_ok=True)
            Path(obj_dump_file).unlink(missing_ok=True)
            Path(exe_c_file).unlink(missing_ok=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

    @override
    def create_module(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> itf.comp.Module:
        return HostModule(name, payload_name, file_name, file_type, graph, **kwargs)

    @override
    def custom_vectorize(self) -> bool:
        return False

    @override
    def apply_custom_vectorize(self, handle: OpResult) -> None:
        return

    def dump_ir(self, mlir_program: RawMlirProgram, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(mlir_program.mlir_module), file=sys.stderr)

    def _mlir_to_llvm_pass(self, mlir_program: RawMlirProgram):
        to_llvm_pass = MlirProgramToLLVMDialectPass(
            mlir_program=mlir_program,
        )
        to_llvm_pass.run()
        if self._config.print_lowered_ir:
            self.dump_ir(mlir_program, "IR Dump After MLIR Opt")

    @property
    def cmd_cc(self):
        return [cc_bin]

    @property
    def cmd_opt(self):
        opt = [f"{self._config.mlir_install_dir}/bin/opt"]
        return (
            opt
            + opt_opts
            + [f"-march={self._config.arch}", f"--mcpu={self._config.cpu}"]
        )

    @property
    def cmd_llc(self):
        llc = [f"{self._config.mlir_install_dir}/bin/llc"]
        if self._config.arch == "native":
            llc_arch = [f"--mcpu={self._config.cpu}"]
        else:
            llc_arch = [f"-march={self._config.arch}", f"--mcpu={self._config.cpu}"]
            triple = target_triple(self._config.arch)
            if triple:
                llc_arch += [f"--mtriple={triple}"]
        return llc + llc_opts + llc_arch

    @property
    def cmd_mlirtranslate(self):
        return [
            f"{self._config.mlir_install_dir}/bin/mlir-translate"
        ] + mlirtranslate_opts

    @property
    def shared_libs(self):
        return system_libs + [
            f"{self._config.mlir_install_dir}/lib/{lib}" for lib in runtime_libs
        ]

    @property
    def shared_path(self):
        return [f"-Wl,-rpath,{self._config.mlir_install_dir}/lib/"]

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self._config.save_temps:
            return
        os.makedirs(self._config.save_temps_dir, exist_ok=True)
        with open(f"{self._config.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))

    def execute_command(
        self,
        cmd: list[str],
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        if self._config.debug:
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


class MlirProgramToLLVMDialectPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def _lowering_pipeline(self) -> list[str]:
        pipeline = [
            "canonicalize",
            "cse",
            "sccp",
        ]
        if "sdist" in self._mlir_program.mlir_extensions:
            pipeline += [
                "sdist-lower-distribution",
                "convert-sdist-to-std",
                "cse",
                "canonicalize",
                "convert-sdist-utils-to-std",
            ]
        pipeline += [
            # From complex control to the soup of basic blocks
            "expand-strided-metadata",
            "convert-linalg-to-loops",
            "lower-affine",
            "convert-vector-to-scf{full-unroll=true}",
            "scf-forall-to-parallel",
            "convert-scf-to-openmp",
            "canonicalize",
            "cse",
            "sccp",
            "convert-scf-to-cf",
            "canonicalize",
            "cse",
            "sccp",
            # Memory accesses to LLVM
            "buffer-results-to-out-params",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}",
            "finalize-memref-to-llvm",
            "canonicalize",
            "cse",
            "sccp",
            # Data flow to LLVM
            "convert-vector-to-llvm{enable-x86vector=true}",
            "convert-index-to-llvm",
            "convert-arith-to-llvm",
            "canonicalize",
            "cse",
            "sccp",
            # Control flow to LLVM
            "convert-cf-to-llvm",
            "convert-openmp-to-llvm",
            "canonicalize",
            "cse",
            "sccp",
        ]
        return pipeline

    def run(self) -> None:
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in self._lowering_pipeline():
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
