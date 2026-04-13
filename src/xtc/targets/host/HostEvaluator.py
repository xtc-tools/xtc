#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import numpy as np
from pathlib import Path

from xtc.runtimes.types.ndarray import NDArray
from xtc.utils.numpy import (
    np_init,
)

from xtc.runtimes.host.HostRuntime import HostRuntime

import xtc.itf as itf
import xtc.targets.host as host

from xtc.utils.loader import LibLoader
from xtc.utils.evaluation import (
    ensure_ndarray_parameters,
    validate_outputs,
    evaluate_performance,
    copy_outputs,
)

__all__ = [
    "HostEvaluator",
    "HostExecutor",
]


class HostEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
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
        self._runtime = kwargs.get("runtime", HostRuntime())
        assert self._module.file_type == "shlib", "only support shlib for evaluation"

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        # Load the module
        dll = str(Path(self._module.file_name).absolute())
        lib = LibLoader(dll)
        sym = self._module.payload_name
        func = getattr(lib.lib, sym)
        func.packed = not self._module._bare_ptr
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
            assert self._runtime is not None
            results = evaluate_performance(
                func,
                parameters,
                self._pmu_counters,
                self._repeat,
                self._number,
                self._min_repeat_ms,
                self._runtime,
            )

        # Unload the module
        lib.close()

        # Copy out outputs
        if self._parameters is not None:
            copy_outputs(parameters, self._parameters)

        return results

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class HostExecutor(itf.exec.Executor):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        self._evaluator = HostEvaluator(
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
