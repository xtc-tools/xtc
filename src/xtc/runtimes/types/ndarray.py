#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import ctypes
import numpy as np
from enum import Enum

__all__ = [
    "NDArray",
]

from .dlpack import DLDevice, DLDeviceTypeCode, DLDataType, DLDataTypeCode, CNDArray

from xtc.runtimes.host.HostRuntime import HostRuntime

from xtc.itf.runtime.common import CommonRuntimeInterface
from xtc.itf.runtime.accelerator import AcceleratorDevice


class NDArrayLocation(Enum):
    HOST = 0
    DEVICE = 1


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
    rev_np_dtype_map: dict[tuple[int, int], str] = {}

    def __init__(
        self, array: Any, runtime: CommonRuntimeInterface | None = None
    ) -> None:
        if not self.rev_np_dtype_map:
            self.rev_np_dtype_map.update(
                {v: k for k, v in NDArray.np_dtype_map.items()}
            )

        self.handle = None
        self.device_handle = None
        self.runtime = runtime
        if self.runtime is None:
            self.runtime = HostRuntime()
        self.location = NDArrayLocation.HOST
        if isinstance(array, NDArray):
            raise RuntimeError("TODO: copy from CNDArray not supported yet")
        elif isinstance(array, np.ndarray):
            self._from_numpy(array)
        else:
            assert 0
        if isinstance(self.runtime, AcceleratorDevice):
            self._to_device()

    def _from_numpy(self, nparray: np.ndarray) -> None:
        assert nparray.flags["C_CONTIGUOUS"]
        self.handle = self._new(nparray.shape, str(nparray.dtype))
        self._copy_from(self.handle, nparray.ctypes.data_as(ctypes.c_voidp))

    def _to_numpy(self) -> np.ndarray:
        shape = self.shape
        np_dtype = self.dtype_str
        nparray = np.empty(shape=shape, dtype=np_dtype)
        self._copy_to(self.handle, nparray.ctypes.data_as(ctypes.c_voidp))
        return nparray

    def _copy_to_numpy(self, out: np.ndarray) -> np.ndarray:
        assert out.flags["C_CONTIGUOUS"]
        assert out.dtype == self.dtype
        assert out.size == self.size
        self._copy_to(self.handle, out.ctypes.data_as(ctypes.c_voidp))
        return out

    def numpy(self, out: np.ndarray | None = None) -> np.ndarray:
        if self.is_on_device():
            assert isinstance(self.runtime, AcceleratorDevice)
            assert self.handle is not None
            bytes_size = self.size * self.dtype.itemsize
            self.runtime.memory_copy_from(
                self.device_handle, self.handle.contents.dl_tensor.data, bytes_size
            )
        if out is None:
            return self._to_numpy()
        else:
            return self._copy_to_numpy(out)

    def _to_device(self) -> None:
        assert (
            isinstance(self.runtime, AcceleratorDevice)
            and self.location == NDArrayLocation.HOST
        )
        assert self.handle is not None
        bytes_size = self.size * self.dtype.itemsize
        self.device_handle = self.runtime.memory_allocate(bytes_size)
        self.runtime.memory_copy_to(
            self.device_handle, self.handle.contents.dl_tensor.data, bytes_size
        )
        self.location = NDArrayLocation.DEVICE

    def _from_device(self) -> None:
        assert (
            isinstance(self.runtime, AcceleratorDevice)
            and self.location == NDArrayLocation.DEVICE
        )
        assert self.handle is not None
        bytes_size = self.size * self.dtype.itemsize
        self.runtime.memory_copy_from(
            self.device_handle, self.handle.contents.dl_tensor.data, bytes_size
        )
        self.runtime.memory_free(self.device_handle)
        self.device_handle = None
        self.location = NDArrayLocation.HOST

    def is_on_device(self) -> bool:
        return self.location == NDArrayLocation.DEVICE

    @property
    def dtype_str(self) -> str:
        assert self.handle is not None
        dtype = self.handle.contents.dl_tensor.dtype
        assert dtype.lanes == 1
        dtype_tuple = (dtype.code, dtype.bits)
        assert dtype_tuple in self.rev_np_dtype_map
        return self.rev_np_dtype_map[dtype_tuple]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)

    @property
    def dims(self) -> int:
        assert self.handle is not None
        return self.handle.contents.dl_tensor.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.handle is not None
        shape = [self.handle.contents.dl_tensor.shape[d] for d in range(self.dims)]
        return tuple(shape)

    @property
    def size(self) -> int:
        size = 1
        for d in self.shape:
            size = size * d
        return size

    @property
    def data(self) -> Any:
        assert self.handle is not None
        if self.is_on_device():
            assert isinstance(self.runtime, AcceleratorDevice)
            return self.runtime.memory_data_pointer(self.device_handle)
        return self.handle.contents.dl_tensor.data

    @classmethod
    def _copy_from(cls, handle: Any, data_handle: Any) -> None:
        HostRuntime.get().cndarray_copy_from_data(
            handle,
            data_handle,
        )

    @classmethod
    def _copy_to(cls, handle: Any, data_handle: Any) -> None:
        HostRuntime.get().cndarray_copy_to_data(
            handle,
            data_handle,
        )

    @classmethod
    def _dldatatype(cls, np_dtype: str) -> DLDataType:
        assert np_dtype in cls.np_dtype_map
        return DLDataType(*cls.np_dtype_map[np_dtype], 1)

    @classmethod
    def _new(
        cls, shape: tuple[int, ...], np_dtype: str, device: DLDevice | None = None
    ) -> Any:
        if device is None:
            device = DLDevice(DLDeviceTypeCode.kDLCPU, 0)
        shape_array = (ctypes.c_int64 * len(shape))(*shape)
        dldtype = cls._dldatatype(np_dtype)
        handle = HostRuntime.get().cndarray_new(
            len(shape),
            ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
            dldtype,
            device,
        )
        if handle is None:
            raise RuntimeError(f"C Runtime: unable to allocate CNDArray")
        array_handle = ctypes.cast(handle, ctypes.POINTER(CNDArray))
        return array_handle

    def __del__(self) -> None:
        if self.handle is not None:
            assert self.runtime is not None
            self.runtime.cndarray_del(self.handle)
            self.handle = None
        if self.device_handle is not None:
            assert isinstance(self.runtime, AcceleratorDevice)
            self.runtime.memory_free(self.device_handle)
            self.device_handle = None

    @classmethod
    def set_alloc_alignment(cls, alignment: int) -> None:
        HostRuntime.get().cndarray_set_alloc_alignment(alignment)
