#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

import xtc.targets.accelerator.mppa as mppa
import xtc.itf as itf
from xtc.runtimes.accelerator.mppa import MppaDevice
from xtc.utils.evaluation import (
    ensure_ndarray_parameters,
    validate_outputs,
    evaluate_performance,
    copy_outputs,
)

__all__ = [
    "MppaEvaluator",
    "MppaExecutor",
]


class MppaEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "mppa.MppaModule", **kwargs: Any) -> None:
        self._device = MppaDevice(module._mppa_config)
        self._module = module
        self._repeat = kwargs.get("repeat", 2)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 0)
        assert self._min_repeat_ms == 0, "min_repeat_ms > 0 is not supported yet"
        self._number = kwargs.get("number", 1)
        assert self._number == 1, "number > 1 is not supported yet"  # TODO
        # TODO support min_repeat_ms and number
        # But execution on MPPA has almost no noise except DDR refresh
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


class MppaExecutor(itf.exec.Executor):
    def __init__(self, module: "mppa.MppaModule", **kwargs: Any) -> None:
        self._evaluator = MppaEvaluator(
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
