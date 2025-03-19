#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import numpy as np

from xtc.runtimes.types.ndarray import NDArray
from xtc.utils.numpy import (
    np_init,
)
from xtc.runtimes.host.evaluate import load_and_evaluate, load_and_execute

import xtc.itf as itf
import xtc.targets.host as host


__all__ = [
    "HostEvaluator",
    "HostExecutor",
]


class HostEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        self._module = module
        self._repeat = kwargs.get("repeat", 1)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 0)
        self._number = kwargs.get("number", 1)
        self._validate = kwargs.get("validate", False)
        self._parameters = kwargs.get("parameters")
        self._reference_impl = kwargs.get("reference_impl")
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

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        parameters = self._parameters
        if parameters is None:
            assert self._np_inputs_spec is not None
            assert self._np_outputs_spec is not None
            inputs_spec = self._np_inputs_spec()
            outputs_spec = self._np_outputs_spec()
            out_init = np.zeros if self._init_zero else np.empty
            inputs = [np_init(**spec) for spec in inputs_spec]
            outputs = [out_init(**spec) for spec in outputs_spec]
            parameters = (
                [NDArray(inp) for inp in inputs],
                [NDArray(out) for out in outputs],
            )
        ref_outputs = []
        if self._validate:
            assert self._reference_impl is not None
            ref_inputs = [inp.numpy() for inp in parameters[0]]
            ref_outputs = [
                np.empty(shape=out.shape, dtype=out.dtype) for out in parameters[1]
            ]
            self._reference_impl(*ref_inputs, *ref_outputs)

        return load_and_evaluate(
            module_file=self._module.file_name,
            module_name=self._module.name,
            payload_name=self._module.payload_name,
            bare_ptr=self._module._bare_ptr,
            parameters=parameters,
            validate=self._validate,
            ref_outputs=ref_outputs,
            repeat=self._repeat,
            min_repeat_ms=self._min_repeat_ms,
            number=self._number,
        )

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class HostExecutor(itf.exec.Executor):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        init_zero = kwargs.get("init_zero", False)
        np_inputs_spec = kwargs.get("np_inputs_spec")
        np_outputs_spec = kwargs.get("np_outputs_spec")
        self._evaluator = HostEvaluator(
            module=module,
            init_zero=init_zero,
            np_inputs_spec=np_inputs_spec,
            np_outputs_spec=np_outputs_spec,
            repeat=1,
            min_repeat_ms=0,
            number=1,
        )

    @override
    def execute(self) -> int:
        results, code, err_msg = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
