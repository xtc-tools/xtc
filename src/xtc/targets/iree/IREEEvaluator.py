#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import time
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import numpy as np

import xtc.itf as itf
from xtc.utils.numpy import np_init

if TYPE_CHECKING:
    from .IREEModule import IREEModule

__all__ = [
    "IREEEvaluator",
    "IREEExecutor",
]

# Auto-scaling factor for the timing window, matching the host runtime.
_NUMBER_FACTOR = 2


def _load_function(module: "IREEModule", driver: str) -> tuple[Any, Any]:
    """Load the vmfb and return ``(bound_function, device)``.

    ``driver`` selects the HAL device: ``local-task`` spreads the workgroup grid
    across worker threads, ``local-sync`` runs everything on the calling thread
    (single-threaded execution of the same module).
    """
    from iree import runtime as ireert

    with open(module.file_name, "rb") as f:
        vmfb = f.read()
    config = ireert.Config(driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    bound_module = getattr(ctx.modules, vm_module.name)
    return bound_module[module.payload_name], config.device


class IREEEvaluator(itf.exec.Evaluator):
    """Evaluate an :class:`IREEModule` through the IREE runtime."""

    def __init__(self, module: "IREEModule", **kwargs: Any) -> None:
        self._module = module
        self._repeat = kwargs.get("repeat", 1)
        self._number = kwargs.get("number", 1)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 100)
        self._validate = kwargs.get("validate", False)
        self._init_zero = kwargs.get("init_zero", False)
        # The IREE runtime exposes no per-dispatch hardware counters (unlike the
        # host runtime's libpfm path); refuse rather than silently drop them.
        if kwargs.get("pmu_counters"):
            raise NotImplementedError(
                "IREE backend does not support hardware PMU counters"
            )
        # single_thread=True runs on the local-sync HAL device, so the
        # distribution grid is executed sequentially (no thread parallelism).
        self._driver = "local-sync" if kwargs.get("single_thread") else "local-task"
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

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        from iree import runtime as ireert

        func, device = _load_function(self._module, self._driver)
        if self._parameters is not None:
            inputs = [np.asarray(x) for x in self._parameters[0]]
        else:
            inputs = self._make_inputs()
        # Pre-bind inputs to device buffers once, so per-call host<->device
        # marshalling is excluded from the timing.
        dev_inputs = [ireert.asdevicearray(device, x) for x in inputs]

        # Compute the result once when an output is needed (validation against
        # the numpy reference, or writing it back to caller-provided arrays).
        if self._validate or self._parameters is not None:
            result = func(*dev_inputs)
            results = result if isinstance(result, (list, tuple)) else [result]
            actual = [np.asarray(r.to_host()) for r in results]

            if self._validate:
                assert self._reference_impl is not None
                assert self._np_outputs_spec is not None
                ref_outputs = [
                    np.empty(shape=spec["shape"], dtype=spec["dtype"])
                    for spec in self._np_outputs_spec()
                ]
                self._reference_impl(*inputs, *ref_outputs)
                for ref, got in zip(ref_outputs, actual):
                    if not np.allclose(ref, got, rtol=1e-4, atol=1e-4):
                        return ([], 1, "Error in validation: outputs differ")

            if self._parameters is not None:
                for dst, got in zip(self._parameters[1], actual):
                    dst[:] = got

        # Measure the performance: one warmup call, then `repeat` measurements,
        # each averaging over a window auto-scaled to at least min_repeat_ms.
        func(*dev_inputs)
        timings: list[float] = []
        for _ in range(self._repeat):
            attempts = self._number
            while True:
                start = time.perf_counter()
                for _ in range(attempts):
                    func(*dev_inputs)
                elapsed = time.perf_counter() - start
                if self._min_repeat_ms <= 0 or elapsed * 1000 >= self._min_repeat_ms:
                    break
                attempts *= _NUMBER_FACTOR
            timings.append(elapsed / attempts)
        return (timings, 0, "")

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
