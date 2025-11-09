#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from types import ModuleType
import ctypes

from xtc.runtimes.host.runtime import RuntimeType

__all__ = [
    "Evaluator",
    "Executor",
]


class ArgTypeCode:
    INT = 0
    HANDLE = 3
    NDARRAY_HANDLE = 13


CArgCode = ctypes.c_int


class CArgValue(ctypes.Union):
    _fields_ = [
        ("v_int64", ctypes.c_int64),
        ("v_float64", ctypes.c_double),
        ("v_handle", ctypes.c_void_p),
        ("v_str", ctypes.c_char_p),
    ]


class CRetValue(CArgValue):
    pass


CPackedFunc = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(CArgValue),
    ctypes.POINTER(CArgCode),
    ctypes.c_int,
    ctypes.POINTER(CRetValue),
    ctypes.POINTER(CArgCode),
)


class CFunc:
    def __init__(self, f: Any, packed: bool = False) -> None:
        self.handle = f
        self.is_packed = packed or (
            hasattr(self.handle, "packed") and self.handle.packed
        )

    def arg_tuple(self, arg: Any) -> Any:
        if arg.__class__.__name__ == "ndarray":  # Numpy Array
            assert not self.is_packed
            return (arg.ctypes.data_as(ctypes.c_voidp), ArgTypeCode.HANDLE)
        elif arg.__class__.__name__ == "NDArray":  # TVM NDArray or our NDArray
            if self.is_packed:
                return (
                    CArgValue(v_handle=ctypes.cast(arg.handle, ctypes.c_void_p)),
                    ArgTypeCode.NDARRAY_HANDLE,
                )
            else:
                return (
                    ctypes.cast(arg.handle.contents.dl_tensor.data, ctypes.c_void_p),
                    ArgTypeCode.HANDLE,
                )
        else:
            assert 0, f"Unsupported argument class: {arg.__class__.__name__}"

    def args_tuples(self, args: Any) -> list[Any]:
        return [self.arg_tuple(arg) for arg in args]

    def __call__(self, *args: Any):
        args_tuples = self.args_tuples(args)
        if self.is_packed:
            args_array = (CArgValue * len(args_tuples))(
                *[arg[0] for arg in args_tuples]
            )
            args_codes = (CArgCode * len(args_tuples))(*[arg[1] for arg in args_tuples])
            result_val = CRetValue(0)
            result_code = CArgCode(ArgTypeCode.INT)
            res = CPackedFunc(self.handle)(
                args_array,
                args_codes,
                len(args_tuples),
                ctypes.byref(result_val),
                ctypes.byref(result_code),
                ctypes.c_int(len(args_tuples)),
            )
            assert res == 0, f"error calling packed function"
        else:
            data_args = [arg[0] for arg in args_tuples]
            self.handle(*data_args)


class Evaluator:
    def __init__(
        self,
        f: Any,
        runtime: ModuleType,
        repeat: int = 1,
        number: int = 1,
        min_repeat_ms: int = 0,
        pmu_counters: list[str] = [],
    ) -> None:
        assert repeat > 0
        assert number > 0
        assert min_repeat_ms >= 0
        self.repeat = repeat
        self.number = number
        self.min_repeat_ms = min_repeat_ms
        self.pmu_counters = pmu_counters
        self.runtime = runtime
        self.cfunc = CFunc(f)

    def _str_list_to_c(self, str_list: list[str]) -> Any:
        return (ctypes.c_char_p * len(str_list))(
            *[str.encode("utf-8") for str in str_list]
        )

    def __call__(self, *args: Any) -> list[float]:
        args_tuples = self.cfunc.args_tuples(args)
        values_num = 1
        if len(self.pmu_counters) > 0:
            values_num = len(self.pmu_counters)
            if (
                any(counter.startswith("gpu.") for counter in self.pmu_counters)
                and self.runtime.type() != RuntimeType.GPU
            ):
                raise ValueError(
                    "GPU PMU counters are not requested but target is not a GPU."
                )
        results_array = (ctypes.c_double * (self.repeat * values_num))()
        if self.cfunc.is_packed:
            args_array_packed = (CArgValue * len(args_tuples))(
                *[arg[0] for arg in args_tuples]
            )
            args_codes_packed = (CArgCode * len(args_tuples))(
                *[arg[1] for arg in args_tuples]
            )
            self.runtime.evaluate_packed_perf(
                ctypes.cast(results_array, ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(len(self.pmu_counters)),
                self._str_list_to_c(self.pmu_counters),
                ctypes.c_int(self.repeat),
                ctypes.c_int(self.number),
                ctypes.c_int(self.min_repeat_ms),
                ctypes.cast(self.cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
                ctypes.cast(args_array_packed, ctypes.POINTER(ctypes.c_voidp)),
                ctypes.cast(args_codes_packed, ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(len(args_tuples)),
            )
        else:
            args_array = (ctypes.c_voidp * len(args_tuples))(
                *[arg[0] for arg in args_tuples]
            )
            self.runtime.evaluate_perf(
                ctypes.cast(results_array, ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(len(self.pmu_counters)),
                self._str_list_to_c(self.pmu_counters),
                ctypes.c_int(self.repeat),
                ctypes.c_int(self.number),
                ctypes.c_int(self.min_repeat_ms),
                ctypes.cast(self.cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
                ctypes.cast(args_array, ctypes.POINTER(ctypes.c_voidp)),
                ctypes.c_int(len(args_tuples)),
            )
        return [float(x) for x in results_array]


class Executor:
    def __init__(self, f: Any) -> None:
        self.func = CFunc(f)

    def __call__(self, *args: Any) -> None:
        self.func(*args)
