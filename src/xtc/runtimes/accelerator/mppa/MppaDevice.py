#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import subprocess
import logging
import os
import subprocess
import ctypes
import sys
from pathlib import Path
from typing import Any, Callable
from typing_extensions import override

from xtc.itf.runtime.accelerator import AcceleratorDevice
from xtc.itf.comp.module import Module
from xtc.utils.cfunc import CFunc

__all__ = ["MppaDevice"]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

from xtc.runtimes.types.dlpack import DLDevice, DLDataType

from xtc.utils.ext_tools import cc_bin

from .config import MppaConfig
from xtc.utils.loader import LibLoader
from xtc.runtimes.host.HostRuntime import HostRuntime

MAX_NB_LOADED_KERNELS = 10


def _get_csrcs_dir():
    return Path(__file__).parents[3] / "csrcs" / "runtimes" / "accelerator" / "mppa"


def _execute_command(
    cmd: list[str],
    input_pipe: str | None = None,
    pipe_stdoutput: bool = True,
    debug: bool = False,
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


def _compile_kvx_object(device: "MppaDevice", src_file: str, obj_file: str):
    cmd_kvx_cc = [f"{device._csw_path}/bin/kvx-cos-gcc"]
    cmd = cmd_kvx_cc + [
        "-O2",
        "-fPIC",
        f"-I{device._mlir_mppa_path}/include",
        "-march=kv3-2",
        "-c",
        src_file,
        "-o",
        obj_file,
    ]
    return _execute_command(cmd=cmd, debug=device.config.mlir_config.debug)


def _compile_host_object(device: "MppaDevice", src_file: str, obj_file: str):
    cmd_host_cc = [cc_bin]
    cmd = cmd_host_cc + [
        "-O2",
        "-fPIC",
        "-Wall",
        "-Wextra",
        f"-I{device._mlir_mppa_path}/include",
        f"-I{device._csw_path}/include",
        f"-I{_get_csrcs_dir()}",
        "-DNB_CC=5",
        "-DTARGET_KV3_2",
        '-DKERNEL_PATHNAME="' + device.config.work_dir + "/mppa_runtime_acc.so" + '"',
        "-c",
        src_file,
        "-o",
        obj_file,
    ]
    return _execute_command(cmd=cmd, debug=device.config.mlir_config.debug)


def _compile_runtime_lib(device: "MppaDevice") -> LibLoader:
    kvx_src_files = [
        device._mlir_mppa_path + "/src/runtime/mppa_management_accelerator.c",
    ]
    host_src_files = [
        device._mlir_mppa_path + "/src/runtime/mppa_management_host.c",
        str(_get_csrcs_dir() / "host.c"),
    ]

    # Compile KVX objects
    kvx_obj_files = [
        f"{device.config.work_dir}/{Path(file).stem}.o" for file in kvx_src_files
    ]
    for src_file, obj_file in zip(kvx_src_files, kvx_obj_files):
        _compile_kvx_object(device, src_file, obj_file)
    # Link KVX objects
    cmd_kvx_cc = [f"{device._csw_path}/bin/kvx-cos-gcc"]
    cmd_kvx_link = cmd_kvx_cc + [
        "-shared",
        "-fPIC",
        "-march=kv3-2",
        "-Wl,-soname=mppa_runtime_acc.so",
        *kvx_obj_files,
        "-o",
        device.config.work_dir + "/mppa_runtime_acc.so",
    ]
    exe_process = _execute_command(
        cmd=cmd_kvx_link, debug=device.config.mlir_config.debug
    )
    assert exe_process.returncode == 0

    # Compile host objects
    host_obj_files = [
        f"{device.config.work_dir}/{Path(file).stem}.o" for file in host_src_files
    ]
    for src_file, obj_file in zip(host_src_files, host_obj_files):
        _compile_host_object(device, src_file, obj_file)
    # Link host objects
    cmd_host_cc = [cc_bin]
    cmd_host_link = cmd_host_cc + [
        "-shared",
        "-fPIC",
        "-O2",
        *host_obj_files,
        "-o",
        device.config.work_dir + "/mppa_runtime_host.so",
        "-Wl,-rpath,$ORIGIN/../lib",
        "-L" + device._csw_path + "/lib",
        "-lmppa_offload_host",
        "-lmopd",
        "-lmppa_rproc_host",
        "-lpthread",
    ]
    exe_process = _execute_command(
        cmd=cmd_host_link, debug=device.config.mlir_config.debug
    )
    assert exe_process.returncode == 0

    return LibLoader(device.config.work_dir + "/mppa_runtime_host.so")


class MppaDevice(AcceleratorDevice):
    """A class for Mppa device"""

    # This is a singleton class; only one instance of MppaDevice will ever be created.
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "MppaDevice":
        if cls._instance is None:
            cls._instance = super(MppaDevice, cls).__new__(cls)
            cls._instance.__init_once__(*args)
        return cls._instance

    def __init__(self, config: MppaConfig | None = None):
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
        if (config is not None) and (config != self.config):
            raise ValueError(
                "MppaDevice already initialized with a different configuration"
            )

    def __init_once__(self, config: MppaConfig | None = None):
        if config is None:
            config = MppaConfig()
        self.config: MppaConfig = config
        self.lib_loader: LibLoader | None = None
        self.mppa_initialized: bool = False
        self.loaded_kernels: dict[Module, LibLoader] = {}
        self.calls_counter: int = 0
        self.need_rebuild: bool = False

    def __build_runtime_lib(self) -> LibLoader:
        os.system("mkdir -p " + self.config.work_dir)
        build_subdir = self.config.work_dir + "/mppa_management"
        os.system("mkdir -p " + build_subdir)
        if self.config.platform in ["iss", "qemu"]:
            os.environ["OMP_MPPA_FIRMWARE_NAME"] = self.config.firmware
            os.environ["MPPA_RPROC_PLATFORM_MODE"] = "sim"
            os.environ["MPPA_RPROC_SIM_PATH"] = self.config.work_dir + "/mymppa"
        if self.need_rebuild:
            os.system("rm -r " + build_subdir + "/*")
            self.need_rebuild = False
        return _compile_runtime_lib(self)

    def _insert_mock_tracepoints(self):
        assert self.lib_loader is not None
        kernel_fn = getattr(self.lib_loader.lib, "mppa_insert_mock_tracepoints")
        kernel_fn()

    def __del__(self):
        if self.mppa_initialized:
            self.deinit_device()
        if self.lib_loader is not None:
            self.lib_loader.close()
            self.lib_loader = None
        self._instance = None

    @override
    def detect_accelerator(self) -> bool:
        o = subprocess.run(
            ["kvx-board-diag", "--list-board"], capture_output=True, text=True
        )
        if "No Available board" in o.stdout:
            return False
        return True

    @override
    def target_name(self) -> str:
        return "mppa"

    @override
    def device_name(self) -> str:
        return "k300"

    @override
    def device_arch(self) -> str:
        return "kv3-2"

    @override
    def device_id(self) -> int:
        return 0  # TODO: Allow multiple mppa per machine (e.g. TC4)

    @override
    def init_device(self) -> None:
        """Pre-Init Mppa-Offload, which takes around 3 secondes"""
        if self.mppa_initialized:
            return
        if self.lib_loader is None:
            self.lib_loader = self.__build_runtime_lib()
        assert self.lib_loader is not None
        if self.config.verbose:
            print("(Mppa Pre-Init)")
        os.environ["MLIR_MPPA_FIRMWARE_NAME"] = self.config.firmware
        # prepare qemu/iss
        if self.config.platform in ["iss", "qemu"]:
            os.system("mkdir -p " + self.config.work_dir + "/mymppa")
            os.environ["OMP_MPPA_FIRMWARE_NAME"] = self.config.firmware
            os.environ["MPPA_RPROC_PLATFORM_MODE"] = "sim"
            os.environ["MPPA_RPROC_SIM_PATH"] = self.config.work_dir + "/mymppa"
        if self.config.platform == "iss":
            if self.config.verbose:
                print("(Launching ISS)")
            subprocess.Popen(
                "kvx-cluster --disable-cache --march="
                + self.config.arch
                + " --no-load-elf --sim-server=SOCKET --mmap --mppa-wdir="
                + self.config.work_dir
                + "/mymppa",
                shell=True,
            )
        elif self.config.platform == "qemu":
            if self.config.verbose:
                print("(Launching Qemu)")
            subprocess.Popen(
                "kvx-qemu-offload-bridge --arch "
                + self.config.arch
                + " --work-dir "
                + self.config.work_dir
                + "/mymppa",
                shell=True,
            )
        # set env variables for traces
        if self.config.mppa_trace_enable:
            if self.config.verbose:
                print("(Using Mppa traces)")
                if self.config.platform in ["iss", "qemu"]:
                    print(
                        "[Warning: Mppa traces are enabled, ISS/Qemu cannot handle them]"
                    )
            if self.config.mppa_trace_use_syscall:
                os.environ["MPPA_ENVP"] = (
                    "MPPA_TRACE_ENABLE_META=1 MPPA_TRACE_USE_SYSCALL=1"
                )
                if self.config.verbose:
                    print(
                        "[Warning: Mppa traces are enabled using syscalls, please consider using hardware acquisition if the overhead is too high]"
                    )
            else:
                os.environ["MPPA_ENVP"] = "MPPA_TRACE_ENABLE_META=1"
        preinit_fn = getattr(self.lib_loader.lib, "mppa_init_device")
        preinit_fn.restype = ctypes.c_bool
        if not preinit_fn():
            raise Exception("Failed to pre-init Mppa-Offload")
        self.mppa_initialized = True

    @override
    def deinit_device(self) -> None:
        """De-Init Mppa-Offload"""
        assert self.lib_loader is not None
        if not self.mppa_initialized:
            return
        remaining_modules = list(self.loaded_kernels.keys())
        for module in remaining_modules:
            self.unload_module(module)
        if self.config.verbose:
            print("(Mppa De-Init)")
        deinit_fn = getattr(self.lib_loader.lib, "mppa_deinit_device")
        deinit_fn.restype = ctypes.c_bool
        if not deinit_fn():
            raise Exception("Failed to de-init Mppa-Offload")
        self.mppa_initialized = False

    @override
    def load_module(self, module: Module) -> None:
        """Add a new loaded kernel in the cache"""
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        if len(self.loaded_kernels) == MAX_NB_LOADED_KERNELS:
            if self.config.verbose:
                print(
                    "(maximum number of loaded kernels exceeded, removing the last recently used)"
                )
            assert False, "Maximum number of loaded kernels exceeded"
            # FIXME
        # Init if not already done
        self.init_device()
        # Add new kernel
        if self.config.verbose:
            print("(Loading kernel: " + module.name + ")")
        libloader = LibLoader(str(Path(module.file_name).absolute()))
        # Pass the context created during pre-init
        get_mppa_common_structures_fn = getattr(
            self.lib_loader.lib, "mppa_get_common_structures"
        )
        get_mppa_common_structures_fn.restype = ctypes.c_void_p
        mppa_common_structures = get_mppa_common_structures_fn()
        set_mppa_common_structures_fn = getattr(
            libloader.lib, "set_mppa_common_structures"
        )
        set_mppa_common_structures_fn.argtypes = [ctypes.c_void_p]
        set_mppa_common_structures_fn(mppa_common_structures)
        # Load kernel
        load_kernel_fn = getattr(libloader.lib, "load_kernel")
        load_kernel_fn()
        self.loaded_kernels[module] = libloader

    @override
    def get_module_function(self, module: Module, function_name: str) -> Callable:
        if not self.mppa_initialized:
            self.init_device()
        if module not in self.loaded_kernels.keys():
            raise Exception("Kernel is not loaded")
        func = getattr(self.loaded_kernels[module].lib, function_name)
        assert func is not None, (
            f"Cannot find symbol {function_name} in lib {module.file_name}"
        )
        return func

    @override
    def unload_module(self, module: Module) -> None:
        """Remove a loaded kernel from the cache"""
        if not self.mppa_initialized:
            self.init_device()
        if module not in self.loaded_kernels.keys():
            raise Exception("Kernel is not loaded")
        self.loaded_kernels[module].close()
        self.loaded_kernels.pop(module)

    @override
    def memory_allocate(self, size_bytes: int) -> Any:
        assert self.lib_loader is not None
        # Create a memory handle
        create_memory_handle_fn = getattr(
            self.lib_loader.lib, "mppa_create_memory_handle"
        )
        create_memory_handle_fn.restype = ctypes.c_void_p
        memory_handle = create_memory_handle_fn()
        # Allocate memory
        allocate_memory_fn = getattr(self.lib_loader.lib, "mppa_memory_allocate")
        allocate_memory_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        allocate_memory_fn.restype = ctypes.c_bool
        if not allocate_memory_fn(memory_handle, size_bytes):
            raise Exception("Failed to allocate memory")
        return memory_handle

    @override
    def memory_free(self, handle: Any) -> None:
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        # Free memory
        free_memory_fn = getattr(self.lib_loader.lib, "mppa_memory_free")
        free_memory_fn.argtypes = [ctypes.c_void_p]
        free_memory_fn.restype = ctypes.c_bool
        if not free_memory_fn(handle):
            raise Exception("Failed to free memory")
        # Destroy memory handle
        destroy_memory_handle_fn = getattr(
            self.lib_loader.lib, "mppa_destroy_memory_handle"
        )
        destroy_memory_handle_fn.argtypes = [ctypes.c_void_p]
        destroy_memory_handle_fn.restype = ctypes.c_bool
        if not destroy_memory_handle_fn(handle):
            raise Exception("Failed to destroy memory handle")

    @override
    def memory_copy_to(
        self, acc_handle: Any, src: ctypes.c_void_p, size_bytes: int
    ) -> None:
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        # Copy memory to accelerator device
        copy_to_memory_fn = getattr(self.lib_loader.lib, "mppa_memory_copy_to")
        copy_to_memory_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        copy_to_memory_fn.restype = ctypes.c_bool
        if not copy_to_memory_fn(acc_handle, src, size_bytes):
            raise Exception("Failed to copy memory to accelerator device")

    @override
    def memory_copy_from(
        self, acc_handle: Any, dst: ctypes.c_void_p, size_bytes: int
    ) -> None:
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        # Copy memory from accelerator device to host
        copy_from_memory_fn = getattr(self.lib_loader.lib, "mppa_memory_copy_from")
        copy_from_memory_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        copy_from_memory_fn.restype = ctypes.c_bool
        if not copy_from_memory_fn(acc_handle, dst, size_bytes):
            raise Exception("Failed to copy memory from accelerator device to host")

    @override
    def memory_fill_zero(self, acc_handle: Any, size_bytes: int) -> None:
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        fill_zero_memory_fn = getattr(self.lib_loader.lib, "mppa_memory_fill_zero")
        fill_zero_memory_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fill_zero_memory_fn.restype = ctypes.c_bool
        if not fill_zero_memory_fn(acc_handle, size_bytes):
            raise Exception("Failed to fill memory with zeros")

    @override
    def memory_data_pointer(self, acc_handle: Any) -> ctypes.c_void_p:
        if not self.mppa_initialized:
            self.init_device()
        assert self.lib_loader is not None
        # Get data pointer
        get_data_pointer_fn = getattr(self.lib_loader.lib, "mppa_memory_data_pointer")
        get_data_pointer_fn.argtypes = [ctypes.c_void_p]
        get_data_pointer_fn.restype = ctypes.c_void_p
        return get_data_pointer_fn(acc_handle)

    @override
    def evaluate(
        self,
        results: Any,
        repeat: int,
        number: int,
        nargs: int,
        cfunc: CFunc,
        args: Any,
    ) -> None:
        HostRuntime.get().evaluate(
            results,
            repeat,
            number,
            nargs,
            cfunc,
            args,
        )

    @override
    def evaluate_perf(
        self,
        pmu_events: list[str],
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args_tuples: list[Any],
    ) -> list[float]:
        return HostRuntime.get().evaluate_perf(
            pmu_events,
            repeat,
            number,
            min_repeat_ms,
            cfunc,
            args_tuples,
        )

    @override
    def evaluate_packed(
        self,
        results: Any,
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args: Any,
        codes: Any,
        nargs: int,
    ) -> None:
        raise NotImplementedError("evaluate_packed is not implemented for MPPA device")

    @override
    def evaluate_packed_perf(
        self,
        results: Any,
        pmu_events: list[str],
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args: Any,
        codes: Any,
        nargs: int,
    ) -> None:
        raise NotImplementedError(
            "evaluate_packed_perf is not implemented for MPPA device"
        )

    @override
    def cndarray_new(
        self,
        ndim: int,
        shape: Any,
        dtype: DLDataType,
        device: DLDevice,
    ) -> Any:
        return HostRuntime.get().cndarray_new(ndim, shape, dtype, device)

    @override
    def cndarray_del(self, handle: Any) -> None:
        HostRuntime.get().cndarray_del(handle)

    @override
    def cndarray_copy_from_data(self, handle: Any, data_handle: Any) -> None:
        HostRuntime.get().cndarray_copy_from_data(handle, data_handle)

    @override
    def cndarray_copy_to_data(self, handle: Any, data_handle: Any) -> None:
        HostRuntime.get().cndarray_copy_to_data(handle, data_handle)

    @override
    def evaluate_flops(self, dtype_name: str | bytes) -> float:
        return HostRuntime.get().evaluate_flops(dtype_name)
