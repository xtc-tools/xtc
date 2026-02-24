#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Callable, cast
from xtc.itf.graph import Graph
import ctypes
import numpy as np
from xtc.utils.numpy import np_init
from xtc.runtimes.types.ndarray import NDArray
from xtc.graphs.xtc.graph import XTCGraph
from xtc.graphs.xtc.expr import XTCTensorExpr
from xtc.graphs.xtc.data import XTCTensor
from xtc.utils.cfunc import CFunc, CArgValue, CArgCode
from xtc.itf.runtime.common import CommonRuntimeInterface
from xtc.runtimes.host.HostRuntime import HostRuntime

__all__ = []


def graph_np_inputs_spec(graph: Graph) -> Callable[[], list[dict[str, Any]]]:
    assert isinstance(graph, XTCGraph)
    assert all(
        [
            isinstance(node._expr, XTCTensorExpr) and node._expr.type.is_constant()
            for node in graph.inputs_nodes
        ]
    ), f"graph inputs are not tensors"

    def _graph_np_inputs_spec() -> list[dict[str, Any]]:
        inputs_types = [
            cast(XTCTensorExpr, node._expr).type for node in graph.inputs_nodes
        ]
        return [
            {
                "shape": type.constant_shape,
                "dtype": type.constant_dtype,
                "device": type.device,
            }
            for type in inputs_types
        ]

    return _graph_np_inputs_spec


def graph_np_outputs_spec(graph: Graph) -> Callable[[], list[dict[str, Any]]]:
    assert isinstance(graph, XTCGraph)
    assert all([node._outputs_types is not None for node in graph.outputs_nodes]), (
        f"graph types were not forwarded"
    )

    def _graph_np_outputs_spec() -> list[dict[str, Any]]:
        return [
            {
                "shape": type.constant_shape,
                "dtype": type.constant_dtype,
                "device": type.device,
            }
            for type in [
                cast(list, node._outputs_types)[0] for node in graph.outputs_nodes
            ]
        ]

    return _graph_np_outputs_spec


def graph_reference_impl(graph: Graph) -> Callable[[], None]:
    def _graph_reference_impl(*args: Any) -> None:
        inputs = [XTCTensor(inp) for inp in args[: len(graph.inputs)]]
        outputs = graph.forward(inputs)
        for idx, out in enumerate(args[len(graph.inputs) :]):
            out[:] = outputs[idx].numpy()

    return _graph_reference_impl


def ensure_ndarray_parameters(
    parameters: tuple[Any, Any] | None,
    np_inputs_spec: Callable[[], list[dict[str, Any]]] | None,
    np_outputs_spec: Callable[[], list[dict[str, Any]]] | None,
    init_zero: bool = False,
) -> tuple[list[NDArray], list[NDArray]]:
    if parameters is None:
        assert np_inputs_spec is not None
        assert np_outputs_spec is not None
        inputs_spec = np_inputs_spec()
        outputs_spec = np_outputs_spec()
        out_init = np.zeros if init_zero else np.empty
        inputs = [
            (np_init(**spec), spec["device"] if "device" in spec else HostRuntime.get())
            for spec in inputs_spec
        ]
        outputs = [
            out_init(**{k: v for k, v in spec.items() if k != "device"})
            for spec in outputs_spec
        ]
        parameters = (
            [NDArray(*inp) for inp in inputs],
            [
                NDArray(
                    out,
                    runtime=spec["device"] if "device" in spec else HostRuntime.get(),
                )
                for out, spec in zip(outputs, outputs_spec)
            ],
        )
    else:
        inputs, outputs = parameters
        nd_inputs = [
            NDArray(inp) if isinstance(inp, np.ndarray) else inp for inp in inputs
        ]
        nd_outputs = [
            NDArray(out) if isinstance(out, np.ndarray) else out for out in outputs
        ]
        parameters = (nd_inputs, nd_outputs)
    return parameters


def validate_outputs(
    func: Callable[[Any], Any],
    parameters: tuple[list[NDArray], list[NDArray]],
    reference_impl: Callable[[], None],
) -> tuple[list[float], int, str]:
    # Get the reference outputs
    assert reference_impl is not None
    ref_inputs = [inp.numpy() for inp in parameters[0]]
    ref_outputs = [np.empty(shape=out.shape, dtype=out.dtype) for out in parameters[1]]
    reference_impl(*ref_inputs, *ref_outputs)
    # Get the function outputs
    CFunc(func)(*parameters[0], *parameters[1])
    # Compare
    for out_ref, out in zip(ref_outputs, [out.numpy() for out in parameters[1]]):
        if not np.allclose(out_ref, out):
            return ([], 1, "Error in validation: outputs differ")
    return ([], 0, "")


def evaluate_performance(
    func: Callable[[Any], Any],
    parameters: tuple[list[NDArray], list[NDArray]],
    pmu_counters: list[str],
    repeat: int,
    number: int,
    min_repeat_ms: int,
    runtime: CommonRuntimeInterface | Any,
) -> tuple[list[float], int, str]:
    # TODO migrate host runtime to CommonRuntimeInterface
    cfunc = CFunc(func)
    args_tuples = cfunc.args_tuples([*parameters[0], *parameters[1]])
    values_num = 1
    if len(pmu_counters) > 0:
        values_num = len(pmu_counters)
        # FIXME check if the PMU counters are supported by the target
    results_array = (ctypes.c_double * (repeat * values_num))()
    if cfunc.is_packed:
        args_array_packed = (CArgValue * len(args_tuples))(
            *[arg[0] for arg in args_tuples]
        )
        args_codes_packed = (CArgCode * len(args_tuples))(
            *[arg[1] for arg in args_tuples]
        )
        runtime.evaluate_packed_perf(
            results_array,
            pmu_counters,
            repeat,
            number,
            min_repeat_ms,
            cfunc,
            args_array_packed,
            args_codes_packed,
            len(args_tuples),
        )
        eval_results = [float(x) for x in results_array]
    else:
        eval_results = runtime.evaluate_perf(
            pmu_counters,
            repeat,
            number,
            min_repeat_ms,
            cfunc,
            args_tuples,
        )
    return (eval_results, 0, "")


def copy_outputs(
    parameters: tuple[list[NDArray], list[NDArray]],
    target_parameters: tuple[list[NDArray], list[NDArray]],
) -> None:
    _, outputs = target_parameters
    _, outputs_copy = parameters
    for out, out_copy in zip(outputs, outputs_copy):
        if isinstance(out, np.ndarray):
            out_copy.numpy(out=out)
