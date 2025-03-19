#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import numpy as np
import ctypes

from .runtime import (
    evaluate,
    evaluate_packed,
)

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
    def __init__(self, f, packed=False):
        self.handle = f
        self.is_packed = packed or (
            hasattr(self.handle, "packed") and self.handle.packed
        )

    def arg_tuple(self, arg):
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

    def args_tuples(self, args):
        return [self.arg_tuple(arg) for arg in args]

    def __call__(self, *args):
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
    def __init__(self, f, repeat=1, number=1, min_repeat_ms=0):
        assert repeat > 0
        assert number > 0
        assert min_repeat_ms >= 0
        self.repeat = repeat
        self.number = number
        self.min_repeat_ms = min_repeat_ms
        self.cfunc = CFunc(f)

    def __call__(self, *args):
        args_tuples = self.cfunc.args_tuples(args)
        results_array = (ctypes.c_double * self.repeat)()
        if self.cfunc.is_packed:
            args_array = (CArgValue * len(args_tuples))(
                *[arg[0] for arg in args_tuples]
            )
            args_codes = (CArgCode * len(args_tuples))(*[arg[1] for arg in args_tuples])
            evaluate_packed(
                ctypes.cast(results_array, ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(self.repeat),
                ctypes.c_int(self.number),
                ctypes.c_int(self.min_repeat_ms),
                ctypes.cast(self.cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
                ctypes.cast(args_array, ctypes.POINTER(ctypes.c_voidp)),
                ctypes.cast(args_codes, ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(len(args_tuples)),
            )
        else:
            args_array = (ctypes.c_voidp * len(args_tuples))(
                *[arg[0] for arg in args_tuples]
            )
            evaluate(
                ctypes.cast(results_array, ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(self.repeat),
                ctypes.c_int(self.number),
                ctypes.c_int(self.min_repeat_ms),
                ctypes.cast(self.cfunc.handle, ctypes.CFUNCTYPE(ctypes.c_voidp)),
                ctypes.cast(args_array, ctypes.POINTER(ctypes.c_voidp)),
                ctypes.c_int(len(args_tuples)),
            )
        return np.array(results_array, dtype="float64")


class Executor:
    def __init__(self, f):
        self.func = CFunc(f)

    def __call__(self, *args):
        self.func(*args)
