#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

from xtc.runtimes.accelerator.gpu.GPUDevice import GPUDevice
import xtc.targets.accelerator.gpu as gpu
import xtc.itf as itf
from xtc.utils.evaluation import (
    ensure_ndarray_parameters,
    validate_outputs,
    evaluate_performance,
    copy_outputs,
)

__all__ = [
    "GPUEvaluator",
    "GPUExecutor",
]


class GPUEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._device = GPUDevice()
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
        self._pmu_counters = kwargs.get("pmu_counters", [])

        assert self._module.file_type == "shlib", "only support shlib for evaluation"

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        assert self._module._bare_ptr, "bare_ptr is not supported for evaluation"

        # Initialize the device and load the module
        self._device.init_device()
        self._device.load_module(self._module)
        sym = self._module.payload_name
        func = self._device.get_module_function(self._module, sym)
        results: tuple[list[float], int, str] = ([], 0, "")
        validation_failed = False

        # Prepare the parameters
        parameters = ensure_ndarray_parameters(
            self._parameters,
            self._np_inputs_spec,
            self._np_outputs_spec,
            self._init_zero,
        )

        # Map the buffers
        # TODO Replace memory mapping of buffers by explicit transfers
        for i, buffer in enumerate(parameters[0]):
            if self._np_inputs_spec()[i]["device"] is None:
                self._device._register_buffer(
                    buffer.data, buffer.size * buffer.dtype.itemsize
                )
        for i, buffer in enumerate(parameters[1]):
            if self._np_outputs_spec()[i]["device"] is None:
                self._device._register_buffer(
                    buffer.data, buffer.size * buffer.dtype.itemsize
                )

        # Check the correctness of the outputs
        if self._validate:
            results = validate_outputs(func, parameters, self._reference_impl)
            validation_failed = results[1] != 0

        # Measure the performance
        if not validation_failed:
            results = evaluate_performance(
                func,
                parameters,
                self._pmu_counters,
                self._repeat,
                self._number,
                self._min_repeat_ms,
                self._device,
            )

        # Unmap the buffers
        for i, buffer in enumerate(parameters[0]):
            if self._np_inputs_spec()[i]["device"] is None:
                self._device._unregister_buffer(buffer.data)
        for i, buffer in enumerate(parameters[1]):
            if self._np_outputs_spec()[i]["device"] is None:
                self._device._unregister_buffer(buffer.data)

        # Unload the module
        self._device.unload_module(self._module)

        # Copy out outputs
        if self._parameters is not None:
            copy_outputs(parameters, self._parameters)

        return results

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class GPUExecutor(itf.exec.Executor):
    def __init__(self, module: "gpu.GPUModule", **kwargs: Any) -> None:
        self._evaluator = GPUEvaluator(
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
