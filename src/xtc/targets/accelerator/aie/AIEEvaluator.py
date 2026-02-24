#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import numpy as np

import xtc.targets.accelerator.aie as aie
import xtc.itf as itf
from xtc.utils.evaluation import (
    ensure_ndarray_parameters,
    validate_outputs,
    evaluate_performance,
    copy_outputs,
)
from xtc.utils.numpy import np_init

from xtc.runtimes.accelerator.aie import AIEDevice

__all__ = [
    "AIEEvaluator",
    "AIEExecutor",
]


class AIEEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "aie.AIEModule", **kwargs: Any) -> None:
        self._device = AIEDevice()
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
        assert self._parameters is None
        inputs = [np_init(**spec) for spec in self._np_inputs_spec()]
        out_init = np.zeros if self._init_zero else np.empty
        outputs = [
            out_init(**{k: v for k, v in spec.items() if k != "device"})
            for spec in self._np_outputs_spec()
        ]
        parameters = (inputs, outputs)

        # Check the correctness of the outputs
        if self._validate:
            raise NotImplementedError("evaluation is not implemented yet")

        # Measure the performance
        if not validation_failed:
            print("(XTC: Proper runtime evaluation harness is not supported yet)")
            res = (func(*parameters[0], *parameters[1]),)
            assert len(self._pmu_counters) == 0, "PMU counters are not supported yet"

        # Copy out outputs
        if self._parameters is not None:
            copy_outputs(parameters, self._parameters)

        return res, 0, ""

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class AIEExecutor(itf.exec.Executor):
    def __init__(self, module: "aie.AIEModule", **kwargs: Any) -> None:
        self._evaluator = AIEEvaluator(
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
