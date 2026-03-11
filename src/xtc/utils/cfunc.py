#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import ctypes

__all__ = [
    "CFunc",
    "CArgValue",
    "CArgCode",
    "CRetValue",
    "CPackedFunc",
    "_c_ascii_str",
    "_str_list_to_c",
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
            if (
                hasattr(arg, "is_on_device") and arg.is_on_device()
            ):  # Device living NDArray
                if self.is_packed:
                    raise RuntimeError("TODO: device NDArray not supported yet")
                else:
                    return (
                        ctypes.cast(arg.data, ctypes.c_void_p),
                        ArgTypeCode.HANDLE,
                    )
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


class _c_ascii_str:
    @staticmethod
    def from_param(obj: str | bytes):
        if isinstance(obj, str):
            obj = obj.encode("ascii")
        return ctypes.c_char_p.from_param(obj)


def _str_list_to_c(str_list: list[str]) -> Any:
    return (ctypes.c_char_p * len(str_list))(*[str.encode("utf-8") for str in str_list])
