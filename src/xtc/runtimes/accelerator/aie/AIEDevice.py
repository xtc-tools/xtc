#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import logging
import ctypes
from typing import Any, Callable
from typing_extensions import override

from xtc.itf.runtime.accelerator import AcceleratorDevice
from xtc.itf.comp.module import Module
from xtc.utils.cfunc import CFunc

__all__ = ["AIEDevice"]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

from xtc.runtimes.types.dlpack import DLDevice, DLDataType
from xtc.runtimes.host.HostRuntime import HostRuntime


class AIEDevice(AcceleratorDevice):
    """A class for AIE device"""

    # This is a singleton class; only one instance of AIEDevice will ever be created.
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AIEDevice":
        if cls._instance is None:
            cls._instance = super(AIEDevice, cls).__new__(cls)
            cls._instance.__init_once__(*args)
        return cls._instance

    def __init__(self):
        try:
            import aie
        except ImportError:
            raise ImportError(
                "mlir_aie is not installed but is required for aie target"
            )
        self._aie_path = aie.__path__[0]

    def __init_once__(self):
        self.aie_initialized: bool = False

    def __del__(self):
        if self.aie_initialized:
            self.deinit_device()
        self._instance = None

    @override
    def detect_accelerator(self) -> bool:
        raise NotImplementedError(
            "detect_accelerator is not implemented for aie device"
        )

    @override
    def target_name(self) -> str:
        return "aie"

    @override
    def device_name(self) -> str:
        return "NPU"

    @override
    def device_arch(self) -> str:
        return "aie2"

    @override
    def device_id(self) -> int:
        return 0

    @override
    def init_device(self) -> None:
        if self.aie_initialized:
            return
        self.aie_initialized = True

    @override
    def deinit_device(self) -> None:
        if not self.aie_initialized:
            return
        self.aie_initialized = False

    @override
    def load_module(self, module: Module) -> None:
        if not self.aie_initialized:
            self.init_device()

    @override
    def get_module_function(self, module: Module, function_name: str) -> Callable:
        if not self.aie_initialized:
            self.init_device()
        assert function_name == module.payload_name, (
            "function name must be the same as the payload name"
        )
        assert hasattr(module, "wrapper"), "Cannot find the AIE module wrapper"
        return getattr(module, "wrapper")

    @override
    def unload_module(self, module: Module) -> None:
        if not self.aie_initialized:
            self.init_device()

    @override
    def memory_allocate(self, size_bytes: int) -> Any:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError("memory_allocate is not implemented for aie device")

    @override
    def memory_free(self, handle: Any) -> None:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError("memory_free is not implemented for aie device")

    @override
    def memory_copy_to(
        self, acc_handle: Any, src: ctypes.c_void_p, size_bytes: int
    ) -> None:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError("memory_copy_to is not implemented for aie device")

    @override
    def memory_copy_from(
        self, acc_handle: Any, dst: ctypes.c_void_p, size_bytes: int
    ) -> None:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError("memory_copy_from is not implemented for aie device")

    @override
    def memory_fill_zero(self, acc_handle: Any, size_bytes: int) -> None:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError("memory_fill_zero is not implemented for aie device")

    @override
    def memory_data_pointer(self, acc_handle: Any) -> ctypes.c_void_p:
        if not self.aie_initialized:
            self.init_device()
        raise NotImplementedError(
            "memory_data_pointer is not implemented for aie device"
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
        raise NotImplementedError("evaluate_packed is not implemented for aie device")

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
            "evaluate_packed_perf is not implemented for aie device"
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
