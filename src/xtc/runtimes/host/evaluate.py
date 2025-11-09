#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from types import ModuleType
import numpy as np
import numpy.typing
from pathlib import Path

from xtc.utils.loader import LibLoader
from xtc.runtimes.types.ndarray import NDArray

from .evaluator import Executor, Evaluator


def load_and_evaluate(
    runtime: ModuleType,
    module_file: str,
    module_name: str,
    payload_name: str,
    **kwargs: Any,
) -> tuple[list[float], int, str]:
    bare_ptr = kwargs.get("bare_ptr", True)
    dll = str(Path(module_file).absolute())
    sym = payload_name
    parameters: tuple[list[NDArray], list[NDArray]] = kwargs.get("parameters", [])
    ref_outputs: list[numpy.typing.NDArray] = kwargs.get("ref_outputs", [])
    validate = kwargs.get("validate", False)
    repeat = kwargs.get("repeat", 1)
    number = kwargs.get("number", 1)
    min_repeat_ms = kwargs.get("min_repeat_ms", 0)
    pmu_counters = kwargs.get("pmu_counters", [])
    with LibLoader(dll) as lib:
        func = getattr(lib, sym)
        assert func is not None, f"Cannot find symbol {sym} in lib {dll}"
        func.packed = not bare_ptr
        if validate:
            exec_func = Executor(func)
            exec_func(*parameters[0], *parameters[1])
            for out_ref, out in zip(
                ref_outputs, [out.numpy() for out in parameters[1]]
            ):
                if not np.allclose(out_ref, out):
                    return [], 1, "Error in validation: outputs differ"
        eval_func = Evaluator(
            func,
            runtime,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            pmu_counters=pmu_counters,
        )
        results = eval_func(*parameters[0], *parameters[1])
    return results, 0, ""


def load_and_execute(
    runtime: ModuleType,
    module_file: str,
    module_name: str,
    payload_name: str,
    **kwargs: Any,
) -> int:
    _, code, _ = load_and_evaluate(
        runtime,
        module_file,
        module_name,
        payload_name,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        **kwargs,
    )
    return code
