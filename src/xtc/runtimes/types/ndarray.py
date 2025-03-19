#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import numpy as np

__all__ = [
    "NDArray",
]

from .dlpack import (
    DLDevice,
    DLDeviceTypeCode,
    DLDataType,
    DLDataTypeCode,
    DLTensor,
    CNDArray,
)

import xtc.runtimes.host.runtime as runtime


class NDArray:
    np_dtype_map = {
        "int8": (DLDataTypeCode.INT, 8),
        "int16": (DLDataTypeCode.INT, 16),
        "int32": (DLDataTypeCode.INT, 32),
        "int64": (DLDataTypeCode.INT, 64),
        "uint8": (DLDataTypeCode.UINT, 8),
        "uint16": (DLDataTypeCode.UINT, 16),
        "uint32": (DLDataTypeCode.UINT, 32),
        "uint64": (DLDataTypeCode.UINT, 64),
        "float32": (DLDataTypeCode.FLOAT, 32),
        "float64": (DLDataTypeCode.FLOAT, 64),
    }
    rev_np_dtype_map = {}

    def __init__(self, array):
        if not self.rev_np_dtype_map:
            self.rev_np_dtype_map.update(
                {v: k for k, v in NDArray.np_dtype_map.items()}
            )

        self.handle = None
        if isinstance(array, NDArray):
            raise RuntimeError("TODO: copy from CNDArray not supported yet")
        elif isinstance(array, np.ndarray):
            self._from_numpy(array)
        else:
            assert 0

    def _from_numpy(self, nparray):
        assert nparray.flags["C_CONTIGUOUS"]
        self.handle = self._new(nparray.shape, str(nparray.dtype))
        self._copy_from(self.handle, nparray.ctypes.data_as(ctypes.c_voidp))

    def _to_numpy(self):
        shape = self.shape
        np_dtype = self.dtype_str
        nparray = np.empty(shape=shape, dtype=np_dtype)
        self._copy_to(self.handle, nparray.ctypes.data_as(ctypes.c_voidp))
        return nparray

    def numpy(self):
        return self._to_numpy()

    @property
    def dtype_str(self):
        dtype = self.handle.contents.dl_tensor.dtype
        assert dtype.lanes == 1
        dtype_tuple = (dtype.code, dtype.bits)
        assert dtype_tuple in self.rev_np_dtype_map
        return self.rev_np_dtype_map[dtype_tuple]

    @property
    def dtype(self):
        return np.dtype(self.dtype_str)

    @property
    def dims(self):
        return self.handle.contents.dl_tensor.ndim

    @property
    def shape(self):
        shape = [self.handle.contents.dl_tensor.shape[d] for d in range(self.dims)]
        return tuple(shape)

    @property
    def size(self):
        size = 1
        for d in self.shape:
            size = size * d
        return size

    @property
    def data(self):
        return self.handle.contents.dl_tensor.data

    @classmethod
    def _copy_from(cls, handle, data_handle):
        runtime.cndarray_copy_from_data(
            handle,
            data_handle,
        )

    @classmethod
    def _copy_to(cls, handle, data_handle):
        runtime.cndarray_copy_to_data(
            handle,
            data_handle,
        )

    @classmethod
    def _dldatatype(cls, np_dtype):
        assert np_dtype in cls.np_dtype_map
        return DLDataType(*cls.np_dtype_map[np_dtype], 1)

    @classmethod
    def _new(cls, shape, np_dtype, device=None):
        if device is None:
            device = DLDevice(DLDeviceTypeCode.kDLCPU, 0)
        shape_array = (ctypes.c_int64 * len(shape))(*shape)
        dldtype = cls._dldatatype(np_dtype)
        handle = runtime.cndarray_new(
            len(shape),
            ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
            dldtype,
            device,
        )
        if handle is None:
            raise RuntimeError(f"C Runtime: unable to allocate CNDArray")
        array_handle = ctypes.cast(handle, ctypes.POINTER(CNDArray))
        return array_handle

    def __del__(self):
        if self.handle is not None:
            runtime.cndarray_del(self.handle)
            self.handle = None

    @classmethod
    def set_alloc_alignment(cls, alignment):
        runtime.cndarray_set_alloc_alignment(alignment)
