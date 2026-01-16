#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import numpy as np
import ctypes

from xtc.runtimes.types.ndarray import NDArray
import xtc.runtimes.gpu.runtime as gpu_runtime
from xtc.utils.numpy import (
    np_init,
)
from xtc.utils.tools import get_mlir_prefix
from xtc.utils.loader import LibLoader
from xtc.utils.ext_tools import cuda_runtime_lib

import xtc.itf as itf
import xtc.targets.gpu as gpu
from xtc.targets.host import HostEvaluator, HostExecutor


__all__ = [
    "GPUEvaluator",
    "GPUExecutor",
]


class GPUEvaluator(HostEvaluator):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._runtime_lib = LibLoader(f"{get_mlir_prefix()}/lib/{cuda_runtime_lib}")
        kwargs["register_buffer_fn"] = self._register_buffer
        kwargs["unregister_buffer_fn"] = self._unregister_buffer
        kwargs["runtime"] = gpu_runtime
        super().__init__(module, **kwargs)

    def __exit(self, exc_type, exc_value, traceback) -> None:
        runtime_lib.close()

    def _register_buffer(self, buffer: NDArray) -> None:
        nb_bytes = buffer.size * buffer.dtype.itemsize
        nb_bytes_c = ctypes.c_int64(nb_bytes)
        buffer_ptr = ctypes.cast(buffer.data, ctypes.c_void_p)
        func_name = "mgpuMemHostRegister"
        func = getattr(self._runtime_lib.lib, func_name)
        assert func is not None, (
            f"Cannot find symbol {func_name} in lib {self._runtime_lib.lib}"
        )
        func(buffer_ptr, nb_bytes_c)

    def _unregister_buffer(self, buffer: NDArray) -> None:
        buffer_ptr = ctypes.cast(buffer.data, ctypes.c_void_p)
        func_name = "mgpuMemHostUnregister"
        func = getattr(self._runtime_lib.lib, func_name)
        assert func is not None, (
            f"Cannot find symbol {func_name} in lib {self._runtime_lib.lib}"
        )
        func(buffer_ptr)


class GPUExecutor(HostExecutor):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._evaluator = GPUEvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=1,
            **kwargs,
        )
