#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from cpuinfo import get_cpu_info
from typing import Dict, Any


def cpu_info() -> Dict[str, Any]:
    """
    Returm cpu info dict with fields:
    - ipc: max number of vec ops per cycle
    - vsize: dict of max elements per type ("float32", "float64", ...)
    - freq: frequency as cycles/sec
    - arch: arch ("x86_64", "aarch64", ...)
    """
    vec_info_map = {
        "avx512": {
            "vsize": {
                "float32": 16,
                "float64": 8,
            }
        },
        "avx2": {
            "vsize": {
                "float32": 8,
                "float64": 4,
            }
        },
        "neon": {
            "vsize": {
                "float32": 4,
                "float64": 2,
            }
        },
        "scalar": {
            "vsize": {
                "float32": 1,
                "float64": 1,
            }
        },
    }
    info = get_cpu_info()
    arch = info["arch_string_raw"]
    flags = info.get("flags", [])
    freqs = info.get("hz_advertised", [4e9])
    if arch == "x86_64":
        if "avx512f" in flags:
            cpu_info = {**vec_info_map["avx512"]}
        elif "avx2" in flags:
            cpu_info = {**vec_info_map["avx2"]}
        elif "sse" in flags:
            cpu_info = {**vec_info_map["sse"]}
        else:
            cpu_info = {**vec_info_map["scalar"]}
        cpu_info["ipc"] = 2
    elif arch == "aarch64":
        if "asimd" in flags:
            cpu_info = {**vec_info_map["neon"]}
        else:
            cpu_info = {**vec_info_map["scalar"]}
        cpu_info["ipc"] = 1
    elif arch == "arm64":
        cpu_info = {**vec_info_map["neon"]}
        cpu_info["ipc"] = 1
    else:
        cpu_info = {**vec_info_map["scalar"]}
        cpu_info["ipc"] = 1
    cpu_info["freq"] = freqs[0]
    cpu_info["arch"] = arch
    return cpu_info


def cpu_peak_cycle(ops: int, dtype: str, threads: int = 1) -> float:
    """
    Return the peak number of cycles (minimal number of cycles) for the
    number of operations given the dtype ("float32", "float64", ...) and
    the number of threads.
    From this value, for instance, given an effective execution time
    for 256x256 ops on float32 with 4 threads,
    one can compute peak performance ratio as:

        peak_perf = num_cycle /cpu_peak_cycle(256*256, "float32", threads=4)

    """
    info = cpu_info()
    cycles = ops / info["ipc"] / info["vsize"][dtype] / threads
    return cycles


def cpu_peak_time(ops: int, dtype: str, threads: int = 1) -> float:
    """
    Return the peak time (minimal time in seconds) for the number of
    operations given the dtype ("float32", "float64", ...) and the
    number of threads.
    From this value, for instance, given an effective execution time
    for 256x256 ops on float32 with 4 threads,
    one can compute peak performance ratio as:

        peak_perf = time/cpu_peak_time(256*256, "float32", threads=4)

    """
    info = cpu_info()
    cycles = ops / info["ipc"] / info["vsize"][dtype] / threads
    time = cycles / info["freq"]
    return time
