#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, cast

import numpy as np

from xtc.graphs.xtc.data import XTCTensor
from xtc.graphs.xtc.expr import XTCTensorExpr
from xtc.graphs.xtc.graph import XTCGraph
from xtc.itf.graph import Graph
from xtc.itf.runtime.common import CommonRuntimeInterface
from xtc.runtimes.host.HostRuntime import HostRuntime
from xtc.runtimes.types.ndarray import NDArray
from xtc.utils.cfunc import CArgCode, CArgValue, CFunc
from xtc.utils.numpy import np_init

__all__: list[str] = []


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
                "layout": type.layout,
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
            (
                np_init(
                    **{k: v for k, v in spec.items() if k != "device" and k != "layout"}
                ),
                spec["device"] if "device" in spec else HostRuntime.get(),
                spec["layout"] if "layout" in spec else None,
            )
            for spec in inputs_spec
        ]
        outputs = [
            out_init(
                **{k: v for k, v in spec.items() if k != "device" and k != "layout"}
            )
            for spec in outputs_spec
        ]
        return (
            [NDArray(*inp) for inp in inputs],
            [
                NDArray(
                    out,
                    runtime=spec["device"] if "device" in spec else HostRuntime.get(),
                )
                for out, spec in zip(outputs, outputs_spec)
            ],
        )

    inputs, outputs = parameters
    nd_inputs: list[NDArray] = [
        NDArray(inp) if isinstance(inp, np.ndarray) else cast(NDArray, inp)
        for inp in inputs
    ]
    nd_outputs: list[NDArray] = [
        NDArray(out) if isinstance(out, np.ndarray) else cast(NDArray, out)
        for out in outputs
    ]
    return (nd_inputs, nd_outputs)


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


# skylake
# perf list --no-desc | grep -i -E -A 59 tma_L[1-4]_group:
DERIVED_METRICS_SIZES = {
    "TopdownL1": 4,  # tma_backend_bound, tma_bad_speculation, tma_frontend_bound, tma_info_core_coreipc, tma_info_inst_mix_instructions, tma_info_thread_slots, tma_retiring
    "TopdownL2": 8,  # tma_branch_mispredicts, tma_core_bound, tma_fetch_bandwidth, tma_fetch_latency, tma_heavy_operations, tma_light_operations, tma_machine_clears, tma_memory_bound
    "TopdownL3": 26,  # tma_branch_resteers, tma_divider, tma_dram_bound, tma_dsb, tma_dsb_switches, tma_few_uops_instructions, tma_fp_arith, tma_fused_instructions, tma_icache_misses, tma_itlb_misses, tma_l1_bound, tma_l2_bound, tma_l3_bound, tma_lcp, tma_memory_operations, tma_microcode_sequencer, tma_mite, tma_ms_switches, tma_non_fused_branches, tma_other_light_ops, tma_other_mispredicts, tma_other_nukes, tma_pmm_bound, tma_ports_utilization, tma_serializing_operation, tma_store_bound
    "TopdownL3_Mem": 5,
    "TopdownL4": 32,  # tma_4k_aliasing, tma_assists, tma_cisc, tma_clears_resteers, tma_contested_accesses, tma_data_sharing, tma_decoder0_alone, tma_dtlb_load, tma_dtlb_store, tma_false_sharing, tma_fb_full, tma_fp_scalar, tma_fp_vector, tma_l1_hit_latency, tma_l3_hit_latency, tma_lock_latency, tma_mem_bandwidth, tma_mem_latency, tma_mispredicts_resteers, tma_nop_instructions, tma_ports_utilized_0, tma_ports_utilized_1, tma_ports_utilized_2, tma_ports_utilized_3m, tma_slow_pause, tma_split_loads, tma_split_stores, tma_sq_full, tma_store_fwd_blk, tma_store_latency, tma_unknown_branches, tma_x87_use
    "TopdownL5": 15,  # tma_alu_op_utilization, tma_fp_assists, tma_fp_vector_128b, tma_fp_vector_256b, tma_fp_vector_512b, tma_load_op_utilization, tma_load_stlb_hit, tma_load_stlb_miss, tma_local_mem, tma_mixing_vectors, tma_remote_cache, tma_remote_mem, tma_store_op_utilization, tma_store_stlb_hit, tma_store_stlb_miss
    "TopdownL6": 8,  # tma_port_0, tma_port_1, tma_port_2, tma_port_3, tma_port_4, tma_port_5, tma_port_6, tma_port_7
    # AMD specific
    "backend_bound_memory": 1,
    "backend_bound_cpu": 1,
    "frontend_bound_latency": 1,
    "frontend_bound_bandwidth": 1,
}


def _fallback_perf_stat(
    failed_counters: list[str], run_dummy_workload: Callable[[], None]
) -> str:
    perf_path = shutil.which("perf")
    if not perf_path:
        return "perf tool not found in PATH"

    my_pid = str(os.getpid())

    perf_metrics = [c for c in failed_counters if c in DERIVED_METRICS_SIZES]
    perf_events = [c for c in failed_counters if c not in DERIVED_METRICS_SIZES]

    #    is_amd = False
    #    if sys.platform == "linux":
    #        try:
    #            with open("/proc/cpuinfo", "r") as f:
    #                if "AuthenticAMD" in f.read():
    #                    is_amd = True
    #        except Exception:
    #            pass

    # if is_amd and "TopdownL2" in perf_metrics:
    #    perf_metrics.remove("TopdownL2")
    #    perf_metrics.extend([
    #        "backend_bound_memory",
    #        "backend_bound_cpu",
    #        "frontend_bound_latency",
    #        "frontend_bound_bandwidth",
    #    ])

    cmd = [perf_path, "stat", "-p", my_pid]

    if perf_events:
        cmd.extend(["-e", ",".join(perf_events)])
    if perf_metrics:
        cmd.extend(["-M", ",".join(perf_metrics)])

    try:
        # print("[DEBUG] Starting perf...")q
        perf_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Time to hook to the process
        time.sleep(0.75)

        run_dummy_workload()

        perf_proc.send_signal(signal.SIGINT)
        _, stderr_output = perf_proc.communicate(timeout=5.0)

        cmd_str = " ".join(cmd)
        formatted_fallback_output = f"$ {cmd_str}\n\n{stderr_output}"

        print(f"\n====== Fallback 'perf stat' Output for {failed_counters} ======\n")
        print(stderr_output)
        print("===================================\n")

        return formatted_fallback_output

    except Exception as e:
        # print(f"[DEBUG] Fallback perf stat failed : {e}")
        return f"Fallback perf stat failed: {e}"


def evaluate_performance(
    func: Callable[[Any], Any],
    parameters: tuple[list[NDArray], list[NDArray]],
    hw_counters: list[str],
    repeat: int,
    number: int,
    min_repeat_ms: int,
    runtime: CommonRuntimeInterface | Any,
) -> tuple[list[float], int, str]:
    # TODO migrate host runtime to CommonRuntimeInterface
    cfunc = CFunc(func)
    args_tuples = cfunc.args_tuples([*parameters[0], *parameters[1]])

    if len(hw_counters) > 0:
        values_num = 0
        for counter in hw_counters:
            values_num += DERIVED_METRICS_SIZES.get(counter, 1)
            # FIXME check if the HW counters are supported by the target
    else:
        values_num = 1
    # print(f"[DEBUG] values_num : {values_num}")
    results_array = (ctypes.c_double * (repeat * values_num))()

    args_array_packed = None
    args_codes_packed = None
    args_array = None

    if cfunc.is_packed:
        args_array_packed = (CArgValue * len(args_tuples))(
            *[arg[0] for arg in args_tuples]
        )
        args_codes_packed = (CArgCode * len(args_tuples))(
            *[arg[1] for arg in args_tuples]
        )
        runtime.evaluate_packed_perf(
            results_array,
            hw_counters,
            repeat,
            number,
            min_repeat_ms,
            cfunc,
            args_array_packed,
            args_codes_packed,
            len(args_tuples),
        )
    else:
        args_array = (ctypes.c_voidp * len(args_tuples))(
            *[arg[0] for arg in args_tuples]
        )
        runtime.evaluate_perf(
            results_array,
            hw_counters,
            repeat,
            number,
            min_repeat_ms,
            cfunc,
            args_array,
            len(args_array),
        )
    # print(f"[DEBUG] results: {[round(x, 2) for x in results_array]}")
    eval_results = [float(x) for x in results_array]

    failed_counters = []
    current_idx = 0

    for counter in hw_counters:
        size = DERIVED_METRICS_SIZES.get(counter, 1)
        chunk = eval_results[current_idx : current_idx + size]

        if any(x == -1.0 for x in chunk) or all(x == 0.0 for x in chunk):
            failed_counters.append(counter)

        current_idx += size

    # Fallback on linux perf tool
    fallback_output = ""
    if failed_counters:
        print(
            f"[WARNING] Some hardware counters failed: {failed_counters}. Fallback to 'perf stat'..."
        )

        def dummy_workload():
            dummy_results = (ctypes.c_double * repeat)()
            if cfunc.is_packed:
                runtime.evaluate_packed_perf(
                    dummy_results,
                    [],
                    repeat,
                    number,
                    min_repeat_ms,
                    cfunc,
                    args_array_packed,
                    args_codes_packed,
                    len(args_tuples),
                )
            else:
                _args_array = (ctypes.c_voidp * len(args_tuples))(
                    *[arg[0] for arg in args_tuples]
                )
                runtime.evaluate_perf(
                    dummy_results,
                    [],
                    repeat,
                    number,
                    min_repeat_ms,
                    cfunc,
                    args_array,
                    len(args_tuples),
                )

        fallback_output = _fallback_perf_stat(failed_counters, dummy_workload)

    return (eval_results, 0, fallback_output)


def copy_outputs(
    parameters: tuple[list[NDArray], list[NDArray]],
    target_parameters: tuple[list[NDArray], list[NDArray]],
) -> None:
    _, outputs = target_parameters
    _, outputs_copy = parameters
    for out, out_copy in zip(outputs, outputs_copy):
        if isinstance(out, np.ndarray):
            out_copy.numpy(out=out)
