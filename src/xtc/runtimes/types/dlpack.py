#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes

__all__ = [
    "DLDeviceTypeCode",
    "DLDeviceType",
    "DLDeviceId",
    "DLDevice",
    "DLDataTypeCode",
    "DLDataType",
    "DLTensor",
    "CNDArray",
]


class DLDeviceTypeCode:
    kDLCPU = 1


DLDeviceType = ctypes.c_int32
DLDeviceId = ctypes.c_int32


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", DLDeviceType),
        ("device_id", DLDeviceId),
    ]


class DLDataTypeCode:
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_voidp),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.POINTER(ctypes.c_uint64)),
    ]


class CNDArray(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("reference_count", ctypes.c_int32),
    ]
