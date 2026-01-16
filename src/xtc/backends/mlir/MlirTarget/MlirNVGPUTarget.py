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

from xtc.utils.ext_tools import (
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    cuda_runtime_lib,
    system_libs,
    cc_bin,
)
from xtc.utils.tools import get_cuda_prefix
from xtc.targets.gpu import GPUModule
import xtc.itf as itf
from xtc.itf.graph import Graph

from mlir.dialects import func
from mlir.ir import UnitAttr
from mlir.passmanager import PassManager

from .MlirTarget import MlirTarget
from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

__all__ = ["MlirNVGPUTarget"]


class MlirNVGPUTarget(MlirTarget):
    """The default NVNVGPU target using llvmir

    This target implements the lowering to nvptx and llvmir dialects. The resulting module can be used by the execution engine.
    """

    def __init__(self, config: MlirConfig):
        super().__init__(config)
        self._available_gpus = self._query_available_gpus()
        self._selected_gpu = self._select_gpu()
        self._ptx = self._query_ptx_version()

    @override
    def name(self) -> str:
        return "llvm-nvgpu"

    @override
    def arch(self) -> str:
        return "sm_" + "".join(self._selected_gpu["compute_cap"].split("."))

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
        so_dump_file = f"{dump_file}.so"
        exe_dump_file = f"{dump_file}.out"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        if self._config.debug:
            print(
                f"> Selected GPU: {self._selected_gpu['name']} (Compute Capability: {self._selected_gpu['compute_cap']})"
            )
            print(f"> Arch: {self.arch()}")
            print(f"> PTX Version: {self._ptx}")

        # Lower to llvm dialect and nvptx
        self._mlir_to_llvm_and_nvptx_pass(mlir_program, self.arch(), self._ptx)

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
            payload_path = ["-Wl,--rpath=${ORIGIN}"]

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

    def _query_available_gpus(self) -> list[dict[str, str]]:
        """
        Query available NVIDIA GPUs using `nvidia-smi`.

        Returns:
            list[dict[str, str]]: List of dictionaries, each with 'name' key for GPU model.
            If no GPUs are found or errors occur, the returned list will contain a single dict
            with a 'name' key describing the error.
        """
        try:
            # Execute nvidia-smi to get the GPU names.
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                gpu_infos = [
                    line.strip().split(",")
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                if gpu_infos:
                    return [
                        {
                            "name": fields[0].strip(),
                            "compute_cap": fields[1].strip()
                            if len(fields) > 1
                            else "Unknown",
                        }
                        for fields in gpu_infos
                    ]
                else:
                    return [
                        {
                            "name": "GPU detected but name unavailable",
                            "compute_cap": "Unknown",
                        }
                    ]
            else:
                return [
                    {
                        "name": f"nvidia-smi failed with return code {result.returncode}: {result.stderr.strip()}"
                    }
                ]
        except FileNotFoundError:
            return [
                {
                    "name": "nvidia-smi command not found - NVIDIA drivers may not be installed"
                }
            ]
        except subprocess.TimeoutExpired:
            return [{"name": "nvidia-smi command timed out"}]
        except Exception as e:
            return [{"name": f"Error running nvidia-smi: {str(e)}"}]

    def _select_gpu(self) -> dict[str, str]:
        """
        Select the best GPU based on the available GPUs.
        """
        # Assert that CUDA_VISIBLE_DEVICES and config.selected_device are not both set at the same time
        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)
        config_selected_device = self._config.selected_device
        if cuda_visible_devices_env is not None and config_selected_device is not None:
            raise AssertionError(
                "Both CUDA_VISIBLE_DEVICES environment variable and config.selected_device are set; "
                "please set only one to specify the GPU."
            )
        device_index = None
        if config_selected_device is not None:
            device_index = config_selected_device
        else:
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible_devices is not None:
                try:
                    device_index = int(cuda_visible_devices)
                except ValueError:
                    raise ValueError(
                        f"Invalid CUDA_VISIBLE_DEVICES value: {cuda_visible_devices}"
                    )
            else:
                device_index = 0
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        if not (0 <= device_index < len(self._available_gpus)):
            raise ValueError(f"Selected device index is out of range: {device_index}")

        # Force use of the same enumeration order than nvidia-smi to get available GPUs
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return self._available_gpus[device_index]

    def _query_ptx_version(self) -> str:
        """
        Query the PTX version of the selected GPU.
        """
        cuda_install_dir = get_cuda_prefix()
        try:
            compiled_cmd = subprocess.run(
                [
                    f"{cuda_install_dir}/bin/nvcc",
                    "-c",
                    "-arch=" + self.arch(),
                    "-x",
                    "cu",
                    "-ptx",
                    "-o",
                    "-",
                    "-",
                ],
                input='extern "C" __global__ void k(){}\n',
                stdout=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            result = subprocess.run(
                ["grep", "-m1", "^\\.version"],
                input=compiled_cmd.stdout,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                ptx_version = result.stdout.strip()
                if not ptx_version.startswith(".version"):
                    raise ValueError(f"Failed to query PTX version: {ptx_version}")
                arch_ptx = "".join(self._selected_gpu["compute_cap"].split("."))
                cc_ptx = "".join(ptx_version.split(" ")[1].strip().split("."))
                # FIXME seems to not work for ptx version above 80
                if int(arch_ptx) > 80 and int(cc_ptx) > 80:
                    return "80"
                # Take the minimum between compiler ptx and compute capability of the device
                if int(arch_ptx) < int(cc_ptx):
                    return arch_ptx
                else:
                    return cc_ptx
            else:
                raise ValueError(
                    f"Failed to query PTX version: {result.stderr.strip()}"
                )
        except FileNotFoundError:
            raise ValueError(
                "nvcc command not found - NVIDIA drivers may not be installed"
            )
        except subprocess.TimeoutExpired:
            raise ValueError("nvcc command timed out")
        except Exception as e:
            raise ValueError(f"Error running nvcc: {str(e)}")

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
        return GPUModule(name, payload_name, file_name, file_type, graph, **kwargs)

    @property
    def disassemble_option(self):
        if not self._config.to_disassemble:
            return "--disassemble"
        else:
            return f"--disassemble={self._config.to_disassemble}"

    def dump_ir(self, mlir_program: RawMlirProgram, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(mlir_program.mlir_module), file=sys.stderr)

    def _apply_emit_c_interface(self, mlir_program: RawMlirProgram):
        with mlir_program.ctx:
            for op in mlir_program.mlir_module.body.operations:
                if isinstance(op, func.FuncOp):
                    op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    def _mlir_to_llvm_and_nvptx_pass(
        self, mlir_program: RawMlirProgram, sm_arch: str, ptx_version: str
    ):
        to_llvm_pass = MlirProgramToLLVMDialectPass(
            mlir_program=mlir_program,
        )
        to_llvm_pass.run(sm_arch, ptx_version)
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
        return llc + llc_opts + llc_arch

    @property
    def cmd_mlirtranslate(self):
        return [
            f"{self._config.mlir_install_dir}/bin/mlir-translate"
        ] + mlirtranslate_opts

    @property
    def shared_libs(self):
        return (
            system_libs
            + [f"{self._config.mlir_install_dir}/lib/{lib}" for lib in runtime_libs]
            + [f"{self._config.mlir_install_dir}/lib/{cuda_runtime_lib}"]
        )

    @property
    def shared_path(self):
        return [f"-Wl,--rpath={self._config.mlir_install_dir}/lib/"]

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

    def _lowering_pipeline(self, sm_arch: str, ptx_version: str) -> list[str]:
        return [
            "canonicalize",
            "cse",
            "sccp",
            # From complex control to the soup of basic blocks
            "expand-strided-metadata",
            "scf-forall-to-parallel",
            "canonicalize",
            "cse",
            "sccp",
            "func.func(gpu-map-parallel-loops)",
            "convert-parallel-loops-to-gpu",
            "convert-linalg-to-loops",
            "canonicalize",
            "cse",
            "sccp",
            "buffer-results-to-out-params",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}",
            "gpu-lower-to-nvvm-pipeline{cubin-chip="
            + sm_arch
            + " cubin-features=+ptx"
            + "".join(ptx_version.split("."))
            + " opt-level=3}",
        ]

    def run(self, sm_arch: str, ptx_version: str) -> None:
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in self._lowering_pipeline(sm_arch, ptx_version):
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
