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
from pathlib import Path

from xtc.utils.ext_tools import (
    get_shlib_extension,
    runtime_libs,
    system_libs,
    cc_bin,
)

from xtc.runtimes.accelerator.mppa import MppaConfig

from xtc.targets.accelerator.mppa import MppaModule
import xtc.itf as itf
from xtc.itf.graph import Graph

from .MlirTarget import MlirTarget
from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

from mlir.passmanager import PassManager
from mlir.ir import OpResult
from mlir.dialects import transform

__all__ = ["MlirMppaTarget"]


class MlirMppaTarget(MlirTarget):
    """Kalray MPPA Target

    This target implements the lowering and code generation to C
    for the Kalray MPPA architecture, using the Mlir-Mppa backend.
    """

    def __init__(self, config: MlirConfig):
        super().__init__(config)
        # config.required_extensions.append("sdist")
        self._mlir_mppa_backend = MlirMppaBackend(config)

    @override
    def name(self) -> str:
        return "mppa"

    @override
    def arch(self) -> str:
        return "kv3-2"

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
        mlir_atrn_dump_file = f"{dump_base}.after_trn.mlir"
        mlir_bmppa_dump_file = f"{dump_base}.before_mppa.mlir"
        mlir_amppa_dump_file = f"{dump_base}.after_mppa.mlir"
        c_host_dump_file = f"{dump_base}.host.c"
        c_accelerator_dump_file = f"{dump_base}.accelerator.c"
        obj_host_dump_file = f"{dump_base}.host.o"
        obj_accelerator_dump_file = f"{dump_base}.accelerator.o"
        so_dump_file = f"{dump_file}.{get_shlib_extension()}"
        kvx_so_dump_file = f"{dump_file}.kvx.so"

        # Lower to MLIR with MPPA dialect
        save_temp(mlir_atrn_dump_file, mlir_program.mlir_module)
        self._mlir_to_mppa_pass(mlir_program)

        # Run MLIR MPPA backend
        with open(mlir_bmppa_dump_file, "w") as outf:
            outf.write(str(mlir_program.mlir_module))
        self._mlir_mppa_backend.run_lowering(
            mlir_before_mppa_dump_file=mlir_bmppa_dump_file,
            mlir_after_mppa_dump_file=mlir_amppa_dump_file,
        )
        if self._config.print_lowered_ir:
            print(f"// -----// IR Dump After MPPA Opt //----- //", file=sys.stderr)
            with open(mlir_amppa_dump_file, "r") as inf:
                print(inf.read(), file=sys.stderr)

        # Generate C code for host and accelerator
        self._mlir_mppa_backend.generate_c_host(
            mlir_after_mppa_dump_file=mlir_amppa_dump_file,
            c_host_dump_file=c_host_dump_file,
        )
        self._mlir_mppa_backend.generate_c_accelerator(
            mlir_after_mppa_dump_file=mlir_amppa_dump_file,
            c_accelerator_dump_file=c_accelerator_dump_file,
        )

        # Compile C code for accelerator
        self._mlir_mppa_backend.compile_c_accelerator(
            c_accelerator_dump_file=c_accelerator_dump_file,
            obj_accelerator_dump_file=obj_accelerator_dump_file,
        )
        # Link KVX library
        self._mlir_mppa_backend.link_kvx_library(
            obj_accelerator_dump_file=obj_accelerator_dump_file,
            kvx_so_dump_file=kvx_so_dump_file,
        )

        # Compile C code for host
        self._mlir_mppa_backend.compile_c_host(
            c_host_dump_file=c_host_dump_file,
            obj_host_dump_file=obj_host_dump_file,
            kvx_so_dump_file=kvx_so_dump_file,
        )

        # Link final shared library
        self._mlir_mppa_backend.link_shared_library(
            obj_host_dump_file=obj_host_dump_file,
            obj_accelerator_dump_file=obj_accelerator_dump_file,
            so_dump_file=so_dump_file,
        )

        # Remove intermediate files if needed
        if not self._config.save_temps:
            os.remove(mlir_bmppa_dump_file)
            os.remove(mlir_amppa_dump_file)
            os.remove(c_host_dump_file)
            os.remove(c_accelerator_dump_file)
            os.remove(obj_host_dump_file)
            os.remove(obj_accelerator_dump_file)

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
        mppa_config = MppaConfig(self._config)
        return MppaModule(
            name, payload_name, file_name, file_type, mppa_config, graph, **kwargs
        )

    @override
    def custom_vectorize(self) -> bool:
        return True

    @override
    def apply_custom_vectorize(self, handle: OpResult) -> None:
        transform.AnnotateOp(handle, "xtc.request_vectorization")

    def dump_ir(self, mlir_program: RawMlirProgram, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(mlir_program.mlir_module), file=sys.stderr)

    def _mlir_to_mppa_pass(self, mlir_program: RawMlirProgram):
        to_mppa_pass = MlirProgramToMlirMppaPass(
            mlir_program=mlir_program,
        )
        to_mppa_pass.run()
        if self._config.print_lowered_ir:
            self.dump_ir(mlir_program, "IR Dump After MLIR Opt")

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


class MlirProgramToMlirMppaPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def _lowering_pipeline(self) -> list[str]:
        pipeline = [
            "cse",
            "sccp",
        ]
        if "sdist" in self._mlir_program.mlir_extensions:
            pipeline += [
                "sdist-lower-distribution",
                "cse",
                "convert-sdist-to-mppa",
                "cse",
                "convert-sdist-utils-to-mppa",
                "cse",
                "canonicalize",
                "cse",
            ]
        return pipeline

    def run(self) -> None:
        self._mlir_program.mlir_context.allow_unregistered_dialects = True
        pm = PassManager(context=self._mlir_program.mlir_context)
        pm.enable_verifier(False)
        for opt in self._lowering_pipeline():
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
        self._mlir_program.mlir_context.allow_unregistered_dialects = False


class MlirMppaBackend:
    def __init__(self, config: MlirConfig):
        self._config = config
        try:
            import mlir_mppa
        except ImportError:
            raise ImportError(
                "mlir_mppa is not installed but is required for MPPA target"
            )
        try:
            self._csw_path = os.environ["KALRAY_TOOLCHAIN_DIR"]
        except KeyError:
            raise KeyError(
                "Please source the Kalray Accesscore Toolchain: https://www.kalrayinc.com/products/software/"
            )
        self._mlir_mppa_path = mlir_mppa.__path__[0]

    @property
    def cmd_mppa_opt(self):
        return [f"{self._mlir_mppa_path}/bin/mppa-opt"]

    @property
    def cmd_mppa_translate(self):
        return [f"{self._mlir_mppa_path}/bin/mppa-translate"]

    @property
    def cmd_kvx_cc(self):
        return [f"{self._csw_path}/bin/kvx-cos-gcc"]

    @property
    def cmd_host_cc(self):
        return [cc_bin]

    def _execute_command(
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

    def _lowering_pipeline(self) -> str:
        passes = []
        # TODO run these only if sdist is not present
        # passes.append("func.func(mppa-launch{device=k300})")
        # passes.append("func.func(kvxcluster-scf-forall-distribute{num-clusters=1})")
        # passes.append("func.func(kvxcluster-launch)")
        passes.append("canonicalize")
        passes.append("func.func(mppa-load-weights)")
        passes.append("func.func(mppa-copy-buffers)")
        passes.append("canonicalize")
        passes.append("func.func(kalray-lift-strided-memref-copy-to-linalg)")
        passes.append("canonicalize")
        passes.append("func.func(kvxcluster-lower-promoted-memory)")
        passes.append(
            "func.func(kvxcluster-optimize-dma-transfers{bundle=true pipeline=false})"
        )
        passes.append("canonicalize")
        passes.append("func.func(kvxcluster-basic-static-allocation)")
        passes.append("canonicalize")
        passes.append("func.func(kalray-remove-useless-initializations)")
        passes.append("canonicalize")
        passes.append("func.func(kvxpe-scf-forall-distribute{num-pes=1})")
        passes.append("func.func(kvxpe-launch)")
        passes.append(
            "func.func(kvxuks-catch{request-attribute=xtc.request_vectorization})"
        )
        passes.append("canonicalize")
        passes.append("convert-linalg-to-loops")
        passes.append("func.func(lower-affine)")
        passes.append("func.func(expand-strided-metadata)")
        passes.append("func.func(kvx-non-canonical-vectorize)")
        passes.append("func.func(kvx-vectorize)")
        passes.append("func.func(scf-forall-to-for)")
        passes.append("convert-math-to-kvxisa")
        passes.append("convert-math-to-libm")
        passes.append("func.func(lower-affine)")
        passes.append("cse")
        # TODO Enable Mppa traces
        ##if config.mppa_trace_enable:
        ##    passes.append("func.func(kalray-request-benchmarks{target-op=kvxcluster.launch})")
        ##    passes.append("kalray-apply-instrumentation{use-traces=" + str(config.mppa_trace_enable) + "}")
        passes.append("func.func(kvxcluster-outline-kernels{specialize=true})")
        passes.append("func.func(canonicalize)")

        new_passes = []
        for p in passes:
            new_passes.append(p)
            new_passes.append("cse")
            # new_passes.append("canonicalize") # FIXME bug with kvxcluster.launch

        # No cse or canonicalize must run after
        new_passes.append(
            "func.func(kalray-clone-crossing-constants)"
        )  # TODO remove remaining useless
        passes = new_passes

        return "builtin.module(" + ",".join(passes) + ")"

    def run_lowering(
        self, mlir_before_mppa_dump_file: str, mlir_after_mppa_dump_file: str
    ) -> None:
        cmd = self.cmd_mppa_opt + [
            "-pass-pipeline=" + self._lowering_pipeline(),
            mlir_before_mppa_dump_file,
            "-o",
            mlir_after_mppa_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def generate_c_host(
        self, mlir_after_mppa_dump_file: str, c_host_dump_file: str
    ) -> None:
        cmd = self.cmd_mppa_translate + [
            "--mlir-to-c-host",
            mlir_after_mppa_dump_file,
            "-o",
            c_host_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def generate_c_accelerator(
        self, mlir_after_mppa_dump_file: str, c_accelerator_dump_file: str
    ) -> None:
        cmd = self.cmd_mppa_translate + [
            "--mlir-to-c-accelerator",
            mlir_after_mppa_dump_file,
            "-o",
            c_accelerator_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def compile_c_accelerator(
        self, c_accelerator_dump_file: str, obj_accelerator_dump_file: str
    ) -> None:
        cmd = self.cmd_kvx_cc + [
            "-O2",
            "-fPIC",
            f"-I{self._mlir_mppa_path}/include",
            "-march=kv3-2",
            "-DBUILD_ID=0",
            "-fvect-cost-model=cheap",
            "-fstack-limit-register=sr",
            "-c",
            c_accelerator_dump_file,
            "-o",
            obj_accelerator_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def link_kvx_library(
        self, obj_accelerator_dump_file: str, kvx_so_dump_file: str
    ) -> None:
        cmd = self.cmd_kvx_cc + [
            "-shared",
            "-fPIC",
            "-march=kv3-2",
            "-Wl,-soname=libkvx.so",
            obj_accelerator_dump_file,
            "-o",
            kvx_so_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def compile_c_host(
        self, c_host_dump_file: str, obj_host_dump_file: str, kvx_so_dump_file: str
    ) -> None:
        cmd = self.cmd_host_cc + [
            "-O2",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-I" + self._mlir_mppa_path + "/include",
            "-I" + self._csw_path + "/include",
            "-DTARGET_KV3_2",
            '-DKERNEL_PATHNAME="' + kvx_so_dump_file + '"',
            "-c",
            c_host_dump_file,
            "-o",
            obj_host_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    def link_shared_library(
        self, obj_host_dump_file: str, obj_accelerator_dump_file: str, so_dump_file: str
    ) -> None:
        cmd = self.cmd_host_cc + [
            "-shared",
            "-fPIC",
            "-O2",
            obj_host_dump_file,
            "-o",
            so_dump_file,
            "-Wl,-rpath,$ORIGIN/../lib",
            "-L" + self._csw_path + "/lib",
            "-lmppa_offload_host",
            "-lmopd",
            "-lmppa_rproc_host",
            "-lpthread",
            "-L" + self._mlir_mppa_path + "/_mlir_libs",
            "-lmlir_c_runner_utils",
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0
