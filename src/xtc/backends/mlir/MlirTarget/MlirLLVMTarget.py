#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override

from xtc.utils.host_tools import target_triple
from xtc.utils.ext_tools import (
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
)
from mlir.passmanager import PassManager

from .MlirCpuTarget import MlirCpuTarget
from .cpu_lowering import cpu_frontend_lowering
from ..MlirProgram import RawMlirProgram

__all__ = ["MlirLLVMTarget"]


class MlirLLVMTarget(MlirCpuTarget):
    """The default CPU target using llvmir

    This target implements the lowering to llvmir, and run llvm toolchain to generate the     final shared lib or executable for CPU.
    """

    @override
    def name(self) -> str:
        return "llvm-cpu"

    @override
    def _lower_and_compile_object(
        self,
        mlir_program: RawMlirProgram,  # Modified in place
        dump_tmp_file: str,
        dump_base: str,
        obj_dump_file: str,
    ) -> list[str]:
        ir_dump_file = f"{dump_tmp_file}.ir"
        bc_dump_file = f"{dump_tmp_file}.bc"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        # Lower to MLIR LLVM dialect
        self._mlir_to_llvm_pass(mlir_program)
        self._save_temp(mlir_llvm_dump_file, mlir_program.mlir_module)

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

        return [ir_dump_file, bc_dump_file]

    def _mlir_to_llvm_pass(self, mlir_program: RawMlirProgram):
        to_llvm_pass = MlirProgramToLLVMDialectPass(
            mlir_program=mlir_program,
        )
        to_llvm_pass.run()
        if self._config.print_lowered_ir:
            self.dump_ir(mlir_program, "IR Dump After MLIR Opt")

    @property
    def cmd_opt(self):
        opt = [f"{self._config.llvm_install_dir}/bin/opt"]
        return (
            opt
            + opt_opts
            + [f"-march={self._config.arch}", f"--mcpu={self._config.cpu}"]
        )

    @property
    def cmd_llc(self):
        llc = [f"{self._config.llvm_install_dir}/bin/llc"]
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


class MlirProgramToLLVMDialectPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def _lowering_pipeline(self) -> list[str]:
        return cpu_frontend_lowering(
            self._mlir_program.mlir_extensions, uplift_fma=True
        ) + [
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
            "convert-math-to-llvm",
            "convert-vector-to-llvm",
            "convert-index-to-llvm",
            "convert-arith-to-llvm",
            "convert-ub-to-llvm",
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

    def run(self) -> None:
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in self._lowering_pipeline():
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
