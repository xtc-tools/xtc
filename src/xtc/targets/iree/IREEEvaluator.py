#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import numpy as np

import xtc.itf as itf
from xtc.utils.numpy import np_init
from xtc.utils.evaluation import compare_to_reference
from xtc.runtimes.iree.IREERuntime import IREERuntime

if TYPE_CHECKING:
    from .IREEModule import IREEModule

__all__ = [
    "IREEEvaluator",
    "IREEExecutor",
]


class IREEEvaluator(itf.exec.Evaluator):
    """Evaluate an `IREEModule` through the shared C measurement loop.

    The compiled ``.vmfb`` is driven by the ``xtc_iree_shim`` native library and
    timed by the same ``evaluate_perf`` used for the host backend, so IREE
    timings are directly comparable to the other backends (identical warmup,
    ``min_repeat_ms`` auto-scaling and clock).
    """

    def __init__(self, module: "IREEModule", **kwargs: Any) -> None:
        self._module = module
        self._repeat = kwargs.get("repeat", 1)
        self._number = kwargs.get("number", 1)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 100)
        self._validate = kwargs.get("validate", False)
        self._init_zero = kwargs.get("init_zero", False)
        # num_threads sizes the IREE worker pool: <=1 -> local-sync (inline, on
        # the measuring thread); >1 -> local-task with that many P-core workers.
        # Back-compat: a `single_thread` bool still selects 1 vs the default pool.
        if "num_threads" in kwargs:
            self._num_threads = max(1, int(kwargs["num_threads"]))
        elif kwargs.get("single_thread"):
            self._num_threads = 1
        else:
            self._num_threads = os.cpu_count() or 1
        self._single_thread = self._num_threads <= 1
        self._pmu_counters = kwargs.get("pmu_counters", [])
        # Counters are per-task (inherit=1), so they only capture workers forked
        # as descendants of the measuring thread inside the timed region (as the
        # host/MLIR/TVM OpenMP kernels do). IREE local-task dispatches onto a
        # persistent pool created at setup, outside that tree, so inherit misses
        # it; only local-sync (single_thread) runs on the measuring thread.
        if self._pmu_counters and not self._single_thread:
            raise NotImplementedError(
                "IREE PMU counters require single_thread (local-sync); "
                "local-task worker threads are not captured"
            )
        # Optional explicit (inputs, outputs) numpy arrays; outputs are written
        # back in place after execution, mirroring the host evaluator.
        self._parameters = kwargs.get("parameters")
        self._np_inputs_spec = kwargs.get("np_inputs_spec", module._np_inputs_spec)
        self._np_outputs_spec = kwargs.get("np_outputs_spec", module._np_outputs_spec)
        self._reference_impl = kwargs.get("reference_impl", module._reference_impl)

    def _make_inputs(self) -> list[np.ndarray]:
        assert self._np_inputs_spec is not None
        inputs: list[np.ndarray] = []
        for spec in self._np_inputs_spec():
            shape = spec["shape"]
            dtype = spec["dtype"]
            if self._init_zero:
                inputs.append(np.zeros(shape=shape, dtype=dtype))
            else:
                inputs.append(np_init(shape=shape, dtype=dtype))
        return inputs

    def _make_outputs(self) -> list[np.ndarray]:
        assert self._np_outputs_spec is not None
        return [
            np.empty(shape=spec["shape"], dtype=spec["dtype"])
            for spec in self._np_outputs_spec()
        ]

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        runtime = IREERuntime()

        if self._parameters is not None:
            inputs = [np.ascontiguousarray(x) for x in self._parameters[0]]
        else:
            inputs = self._make_inputs()
        # The shim writes results into these buffers in place after each invoke.
        outputs = self._make_outputs()

        ctx = runtime.setup(
            vmfb_path=self._module.file_name,
            entry_function=self._module.payload_name,
            num_threads=self._num_threads,
            inputs=inputs,
            outputs=outputs,
        )
        try:
            # A single invocation both validates correctness and, when the caller
            # provided output arrays, produces the values to write back.
            if self._validate or self._parameters is not None:
                runtime.invoke(ctx)
                if self._validate:
                    assert self._reference_impl is not None
                    code, msg = compare_to_reference(
                        outputs, inputs, self._reference_impl
                    )
                    if code != 0:
                        return ([], code, msg)

            results = runtime.evaluate_perf(
                ctx=ctx,
                pmu_events=self._pmu_counters,
                repeat=self._repeat,
                number=self._number,
                min_repeat_ms=self._min_repeat_ms,
            )
        finally:
            runtime.teardown(ctx)

        if self._parameters is not None:
            for dst, got in zip(self._parameters[1], outputs):
                dst[:] = got

        return (results, 0, "")

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class IREEExecutor(itf.exec.Executor):
    """Run an `IREEModule` once, returning its status code."""

    def __init__(self, module: "IREEModule", **kwargs: Any) -> None:
        self._evaluator = IREEEvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=1,
            **kwargs,
        )

    @override
    def execute(self) -> int:
        _, code, _ = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
