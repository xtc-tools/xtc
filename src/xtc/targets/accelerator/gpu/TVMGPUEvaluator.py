#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import time
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

import tvm
from tvm.runtime import _tensor

import xtc.itf as itf
from xtc.runtimes.types.ndarray import NDArray
import xtc.targets.accelerator.gpu as gpu
from xtc.utils.numpy import np_init, np_allclose

__all__ = [
    "TVMGPUEvaluator",
    "TVMGPUExecutor",
]


class TVMGPUEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._module = module
        self._repeat = kwargs.get("repeat", 1)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 100)
        self._number = kwargs.get("number", 1)
        self._validate = kwargs.get("validate", False)
        self._parameters = kwargs.get("parameters")
        self._init_zero = kwargs.get("init_zero", False)
        self._np_inputs_spec = kwargs.get(
            "np_inputs_spec", self._module._np_inputs_spec
        )
        self._np_outputs_spec = kwargs.get(
            "np_outputs_spec", self._module._np_outputs_spec
        )
        self._reference_impl = kwargs.get(
            "reference_impl", self._module._reference_impl
        )
        assert self._module.file_type == "shlib", "only support shlib for evaluation"

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        if isinstance(arr, NDArray):
            return arr.numpy()
        return np.asarray(arr)

    def _host_buffers(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[Any] | None]:
        assert self._np_inputs_spec is not None
        assert self._np_outputs_spec is not None
        inputs_spec = self._np_inputs_spec()
        outputs_spec = self._np_outputs_spec()
        out_init = np.zeros if self._init_zero else np.empty
        if self._parameters is None:
            inputs_np = [
                np_init(shape=tuple(spec["shape"]), dtype=spec["dtype"])
                for spec in inputs_spec
            ]
            outputs_np = [
                out_init(shape=tuple(spec["shape"]), dtype=spec["dtype"])
                for spec in outputs_spec
            ]
            return inputs_np, outputs_np, None
        raw_inputs, raw_outputs = self._parameters
        inputs_np = [self._to_numpy(arr) for arr in raw_inputs]
        outputs_np = [self._to_numpy(arr) for arr in raw_outputs]
        return inputs_np, outputs_np, list(raw_outputs)

    def _measure(self, func: Any, args: list[Any], dev: Any) -> list[float]:
        number = max(self._number, 1)
        times: list[float] = []
        for _ in range(self._repeat):
            func(*args)  # warmup
            dev.sync()
            start = time.perf_counter()
            for _ in range(number):
                func(*args)
            dev.sync()
            times.append(time.perf_counter() - start)
        return times

    def _write_back(
        self,
        outputs_dev: list[Any],
        outputs_np: list[np.ndarray],
        raw_outputs: list[Any] | None,
    ) -> None:
        for out_dev, np_out in zip(outputs_dev, outputs_np):
            np_out[...] = out_dev.numpy()
        if raw_outputs is None:
            return
        for dst, np_out in zip(raw_outputs, outputs_np):
            if isinstance(dst, NDArray):
                dst._copy_from(dst.handle, np_out.ctypes.data_as(ctypes.c_void_p))
            elif dst is not np_out:
                dst[...] = np_out

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        assert self._np_inputs_spec is not None
        assert self._np_outputs_spec is not None
        inputs_spec = self._np_inputs_spec()
        outputs_spec = self._np_outputs_spec()

        inputs_np, outputs_np, raw_outputs = self._host_buffers()

        # Any: the tvm Device stubs do not declare sync(), it exists at runtime
        dev: Any = tvm.device("cuda", 0)
        # Load through the tvm runtime: a tvm cuda module carries its device
        # code as a runtime imports blob that a plain dlopen would not
        # register.
        lib = tvm.runtime.load_module(str(Path(self._module.file_name).absolute()))
        func = lib.get_function(self._module.payload_name)
        assert func is not None, (
            f"payload {self._module.payload_name} not found in {self._module.file_name}"
        )

        # The kernel reads and writes every buffer on the device, so stage
        # all inputs and allocate all outputs as cuda arrays regardless of
        # the host/device annotation of the graph.
        inputs_dev = [
            _tensor.empty(tuple(spec["shape"]), spec["dtype"], dev)
            for spec in inputs_spec
        ]
        outputs_dev = [
            _tensor.empty(tuple(spec["shape"]), spec["dtype"], dev)
            for spec in outputs_spec
        ]
        for dev_arr, host_arr in zip(inputs_dev, inputs_np):
            dev_arr.copyfrom(host_arr)
        args = [*inputs_dev, *outputs_dev]

        # Check the correctness of the outputs
        if self._validate:
            assert self._reference_impl is not None
            ref_outputs = [np.empty_like(out) for out in outputs_np]
            self._reference_impl(*inputs_np, *ref_outputs)
            func(*args)
            dev.sync()
            for out_ref, out_dev in zip(ref_outputs, outputs_dev):
                if not np_allclose(out_ref, out_dev.numpy()):
                    self._write_back(outputs_dev, outputs_np, raw_outputs)
                    return ([], 1, "Error in validation: outputs differ")

        # Measure the performance
        results = (self._measure(func, args, dev), 0, "")

        # Copy out outputs
        self._write_back(outputs_dev, outputs_np, raw_outputs)

        return results

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class TVMGPUExecutor(itf.exec.Executor):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._evaluator = TVMGPUEvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=1,
            **kwargs,
        )

    @override
    def execute(self) -> int:
        results, code, err_msg = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
