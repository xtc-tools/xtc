#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import logging
import ctypes
import logging
import ctypes
from pathlib import Path
from typing import Any, Callable
from typing_extensions import override

from xtc.itf.runtime.accelerator import AcceleratorDevice
from xtc.itf.comp.module import Module
from xtc.utils.cfunc import CFunc, _str_list_to_c, _c_ascii_str

from ...host.runtime import resolve_runtime, RuntimeType, runtime_funcs

__all__ = ["GPUDevice"]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

from xtc.runtimes.types.dlpack import DLDevice, DLDataType

from xtc.utils.loader import LibLoader
from xtc.utils.tools import get_mlir_prefix
from xtc.utils.ext_tools import cuda_runtime_lib


class GPUDevice(AcceleratorDevice):
    """A class for GPU device"""

    # This is a singleton class; only one instance of GPUDevice will ever be created.
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "GPUDevice":
        if cls._instance is None:
            cls._instance = super(GPUDevice, cls).__new__(cls)
            cls._instance.__init_once__(*args)
        return cls._instance

    def __init__(self):
        # TODO check installation of cuda
        pass

    def __init_once__(self):
        self._mlir_runtime_lib = LibLoader(
            f"{get_mlir_prefix()}/lib/{cuda_runtime_lib}"
        )
        self.loaded_kernels: dict[Module, LibLoader] = {}

    def __get_runtime_func(self, name: str) -> Callable:
        if name in runtime_funcs:
            entries = resolve_runtime(RuntimeType.GPU)
            assert entries is not None
            return entries[name]
        raise AttributeError(f"undefined runtime function: {name}")

    def __del__(self):
        remaining_modules = list(self.loaded_kernels.keys())
        for module in remaining_modules:
            self.unload_module(module)
        self._mlir_runtime_lib.close()
        self._instance = None

    @override
    def detect_accelerator(self) -> bool:
        raise NotImplementedError("GPUDevice.detect_accelerator is not implemented")

    @override
    def target_name(self) -> str:
        return "nvgpu"

    @override
    def device_name(self) -> str:
        return "nvgpu"

    @override
    def device_arch(self) -> str:
        return "cuda"

    @override
    def device_id(self) -> int:
        return 0  # TODO: Handle multiple GPUs

    @override
    def init_device(self) -> None:
        # Not necessary for now
        pass

    @override
    def deinit_device(self) -> None:
        # Not necessary for now
        pass

    @override
    def load_module(self, module: Module) -> None:
        libloader = LibLoader(str(Path(module.file_name).absolute()))
        self.loaded_kernels[module] = libloader

    @override
    def get_module_function(self, module: Module, function_name: str) -> Callable:
        if module not in self.loaded_kernels.keys():
            raise Exception("Kernel is not loaded")
        func = getattr(self.loaded_kernels[module].lib, function_name)
        assert func is not None, (
            f"Cannot find symbol {function_name} in lib {module.file_name}"
        )
        return func

    @override
    def unload_module(self, module: Module) -> None:
        if module not in self.loaded_kernels.keys():
            raise Exception("Kernel is not loaded")
        self.loaded_kernels[module].close()
        self.loaded_kernels.pop(module)

    @override
    def memory_allocate(self, size_bytes: int) -> Any:
        raise NotImplementedError("memory_allocate is not implemented for GPU device")

    @override
    def memory_free(self, handle: Any) -> None:
        raise NotImplementedError("memory_free is not implemented for GPU device")

    @override
    def memory_copy_to(
        self, acc_handle: Any, src: ctypes.c_void_p, size_bytes: int
    ) -> None:
        raise NotImplementedError("memory_copy_to is not implemented for GPU device")

    @override
    def memory_copy_from(
        self, acc_handle: Any, dst: ctypes.c_void_p, size_bytes: int
    ) -> None:
        raise NotImplementedError("memory_copy_from is not implemented for GPU device")

    @override
    def memory_fill_zero(self, acc_handle: Any, size_bytes: int) -> None:
        raise NotImplementedError("memory_fill_zero is not implemented for GPU device")

    @override
    def memory_data_pointer(self, acc_handle: Any) -> ctypes.c_void_p:
        raise NotImplementedError(
            "memory_data_pointer is not implemented for GPU device"
        )

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
        self.__get_runtime_func("evaluate")(
            ctypes.cast(results, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(repeat),
            ctypes.c_int(number),
            ctypes.c_int(nargs),
            ctypes.cast(cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
            ctypes.cast(args, ctypes.POINTER(ctypes.c_voidp)),
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
        args_array = (ctypes.c_voidp * len(args_tuples))(
            *[arg[0] for arg in args_tuples]
        )
        values_num = 1
        if len(pmu_events) > 0:
            values_num = len(pmu_events)
            # FIXME check if the PMU events are supported by the target
        results_array = (ctypes.c_double * (repeat * values_num))()
        self.__get_runtime_func("evaluate_perf")(
            ctypes.cast(results_array, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(pmu_events)),
            _str_list_to_c(pmu_events),
            ctypes.c_int(repeat),
            ctypes.c_int(number),
            ctypes.c_int(min_repeat_ms),
            ctypes.cast(cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
            ctypes.cast(args_array, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.c_int(len(args_tuples)),
        )
        return [float(x) for x in results_array]

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
        raise NotImplementedError("evaluate_packed is not implemented for GPU device")

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
            "evaluate_packed_perf is not implemented for GPU device"
        )

    @override
    def cndarray_new(
        self,
        ndim: int,
        shape: Any,
        dtype: DLDataType,
        device: DLDevice,
    ) -> Any:
        # Convert shape if it's a list/tuple to ctypes array
        if isinstance(shape, (list, tuple)):
            shape_array = (ctypes.c_int64 * len(shape))(*shape)
            shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
        return self.__get_runtime_func("cndarray_new")(
            ctypes.c_int32(ndim),
            shape,
            dtype,
            device,
        )

    @override
    def cndarray_del(self, handle: Any) -> None:
        self.__get_runtime_func("cndarray_del")(handle)

    @override
    def cndarray_copy_from_data(self, handle: Any, data_handle: Any) -> None:
        self.__get_runtime_func("cndarray_copy_from_data")(handle, data_handle)

    @override
    def cndarray_copy_to_data(self, handle: Any, data_handle: Any) -> None:
        self.__get_runtime_func("cndarray_copy_to_data")(handle, data_handle)

    @override
    def evaluate_flops(self, dtype_name: str | bytes) -> float:
        return float(
            self.__get_runtime_func("evaluate_flops")(
                _c_ascii_str.from_param(dtype_name)
            )
        )

    # Extra methods
    def _register_buffer(self, handle: Any, size_bytes: int) -> None:
        nb_bytes_c = ctypes.c_int64(size_bytes)
        buffer_ptr = ctypes.cast(handle, ctypes.c_void_p)
        func_name = "mgpuMemHostRegister"
        func = getattr(self._mlir_runtime_lib.lib, func_name)
        assert func is not None, (
            f"Cannot find symbol {func_name} in lib {self._mlir_runtime_lib.lib}"
        )
        func(buffer_ptr, nb_bytes_c)

    def _unregister_buffer(self, handle: Any) -> None:
        buffer_ptr = ctypes.cast(handle, ctypes.c_void_p)
        func_name = "mgpuMemHostUnregister"
        func = getattr(self._mlir_runtime_lib.lib, func_name)
        assert func is not None, (
            f"Cannot find symbol {func_name} in lib {self._mlir_runtime_lib.lib}"
        )
        func(buffer_ptr)
