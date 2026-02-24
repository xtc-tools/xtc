#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import logging
from typing import Any, Callable
from typing_extensions import override

from xtc.runtimes.types.dlpack import DLDevice, DLDataType

from xtc.utils.cfunc import CFunc, _str_list_to_c, _c_ascii_str
from xtc.itf.runtime.common import CommonRuntimeInterface

from .runtime import runtime_funcs, resolve_runtime, RuntimeType

__all__ = ["HostRuntime"]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False


class HostRuntime(CommonRuntimeInterface):
    """A class for Host runtime"""

    # This is a singleton class; only one instance of HostRuntime will ever be created.
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "HostRuntime":
        if cls._instance is None:
            cls._instance = super(HostRuntime, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def __del__(self):
        self._instance = None

    def __get_runtime_func(self, name: str) -> Callable:
        if name in runtime_funcs:
            entries = resolve_runtime(RuntimeType.HOST)
            assert entries is not None
            return entries[name]
        raise AttributeError(f"undefined runtime function: {name}")

    def __getattr__(self, name: str) -> Callable:
        return self.__get_runtime_func(name)

    @classmethod
    def get(cls) -> "HostRuntime":
        if cls._instance is None:
            cls._instance = HostRuntime()
        return cls._instance

    @override
    def target_name(self) -> str:
        return "host"

    @override
    def device_name(self) -> str:
        return "host"

    @override
    def device_arch(self) -> str:
        return "host"

    @override
    def device_id(self) -> int:
        return 0

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
        self.__get_runtime_func("evaluate_packed")(
            ctypes.cast(results, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(repeat),
            ctypes.c_int(number),
            ctypes.c_int(min_repeat_ms),
            ctypes.cast(cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
            ctypes.cast(args, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.cast(codes, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(nargs),
        )

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
        self.__get_runtime_func("evaluate_packed_perf")(
            ctypes.cast(results, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(pmu_events)),
            _str_list_to_c(pmu_events),
            ctypes.c_int(repeat),
            ctypes.c_int(number),
            ctypes.c_int(min_repeat_ms),
            ctypes.cast(cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
            ctypes.cast(args, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.cast(codes, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(nargs),
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
