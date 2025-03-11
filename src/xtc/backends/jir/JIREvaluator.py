#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import numpy as np

from xtc.ndarray import NDArray
import xtc.utils as utils
from xtc.backends.utils.exec import load_and_evaluate, load_and_execute

import xtc.backends.jir as backend
import xtc.itf as itf


class JIREvaluator(itf.exec.Evaluator):
    def __init__(self, module: "backend.JIRModule", **kwargs: Any) -> None:
        self._module = module
        self._repeat = kwargs.get("repeat", 1)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 0)
        self._number = kwargs.get("number", 1)
        self._validate = kwargs.get("validate", False)
        self._parameters = kwargs.get("parameters", None)
        self._reference = kwargs.get("reference", None)
        self._init_zero = kwargs.get("init_zero", False)
        self._operation = self._module._operation

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        parameters = self._parameters
        validate = self._validate
        reference = self._reference
        if parameters is None:
            inputs_spec = self._operation.np_inputs_spec()
            outputs_spec = self._operation.np_outputs_spec()
            out_init = np.zeros if self._init_zero else np.empty
            inputs = [utils.np_init(**spec) for spec in inputs_spec]
            outputs = [out_init(**spec) for spec in outputs_spec]
            parameters = (
                [NDArray(inp) for inp in inputs],
                [NDArray(out) for out in outputs],
            )
        ref_outputs = []
        if validate:
            ref_inputs = [inp.numpy() for inp in parameters[0]]
            ref_outputs = [
                np.empty(shape=out.shape, dtype=out.dtype) for out in parameters[1]
            ]
            if reference is None:
                reference = self._operation.reference_impl
            reference(*ref_inputs, *ref_outputs)

        return load_and_evaluate(
            module_file=self._module.file_name,
            module_name=self._module.name,
            payload_name=self._module.payload_name,
            bare_ptr=self._module._bare_ptr,
            parameters=parameters,
            validate=validate,
            ref_outputs=ref_outputs,
            repeat=self._repeat,
            min_repeat_ms=self._min_repeat_ms,
            number=self._number,
        )

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class JIRExecutor(itf.exec.Executor):
    def __init__(self, module: "backend.JIRModule", **kwargs: Any) -> None:
        init_zero = kwargs.get("init_zero", False)
        self._evaluator = JIREvaluator(
            module=module,
            init_zero=init_zero,
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
