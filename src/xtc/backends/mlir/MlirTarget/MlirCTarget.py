#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override

from xtc.utils.ext_tools import (
    get_openmp_cflags,
    xtctranslate_opts,
    target_cc_bin,
    target_cc_opts,
)

from mlir.passmanager import PassManager

from .MlirCpuTarget import MlirCpuTarget
from .cpu_lowering import cpu_frontend_lowering
from ..MlirProgram import RawMlirProgram

__all__ = ["MlirCTarget"]


class MlirCTarget(MlirCpuTarget):
    """A generic C Target for CPU

    This target implements the lowering and code generation to C, and use
    the default C compiler to generate the final shared lib or executable for CPU.
    """

    @override
    def name(self) -> str:
        return "c-cpu"

    @override
    def _lower_and_compile_object(
        self,
        mlir_program: RawMlirProgram,  # Modified in place
        dump_tmp_file: str,
        dump_base: str,
        obj_dump_file: str,
    ) -> list[str]:
        c_dump_file = f"{dump_tmp_file}.c"
        mlir_std_dump_file = f"{dump_base}.std.mlir"

        # Lower to the MLIR std dialects
        self._mlir_to_std_pass(mlir_program)
        self._save_temp(mlir_std_dump_file, mlir_program.mlir_module)

        codegen_c_cmd = self.cmd_xtc_translate_c + ["-o", c_dump_file]
        codegen_c_process = self.execute_command(
            cmd=codegen_c_cmd,
            input_pipe=str(mlir_program.mlir_module),
        )
        assert codegen_c_process.returncode == 0

        cc_pic = ["-fPIC"] if self._config.shared_lib else []
        cc_crt_inc = [f"-I{self._config.mlir_install_dir}/runtime/include"]
        cc_arch = [f"-march={self._config.cpu}", f"-mtune={self._config.cpu}"]
        cc_omp = get_openmp_cflags(target_cc_bin)
        # FIXME Some options may change between a GCC and a LLVM based compiler
        # This works for GCC 12 for x86
        cc_cmd = (
            self.cmd_target_cc
            + target_cc_opts
            + cc_arch
            + cc_pic
            + cc_crt_inc
            + cc_omp
            + ["-c", c_dump_file, "-o", obj_dump_file]
        )
        cc_process = self.execute_command(cmd=cc_cmd)
        assert cc_process.returncode == 0

        return [c_dump_file]

    def _mlir_to_std_pass(self, mlir_program: RawMlirProgram):
        to_std_pass = MlirProgramToStdDialectsPass(
            mlir_program=mlir_program,
        )
        to_std_pass.run()
        if self._config.print_lowered_ir:
            self.dump_ir(mlir_program, "IR Dump After MLIR Opt")

    @property
    def cmd_target_cc(self):
        return [target_cc_bin]

    @property
    def cmd_xtc_translate_c(self):
        return [
            f"{self._config.mlir_install_dir}/bin/xtc-translate"
        ] + xtctranslate_opts


class MlirProgramToStdDialectsPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def _lowering_pipeline(self) -> list[str]:
        return cpu_frontend_lowering(
            self._mlir_program.mlir_extensions, uplift_fma=False
        ) + [
            # convert-scf-to-cf is intentionally skipped: the C emitter keeps scf
            "canonicalize",
            "cse",
            "sccp",
            # Memory accesses
            "buffer-results-to-out-params",
            "canonicalize",
            "cse",
            "sccp",
        ]

    def run(self) -> None:
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in self._lowering_pipeline():
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
