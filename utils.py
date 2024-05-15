#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
Some utilitary and math functions.
"""

import numpy as np
import operator
from functools import reduce
from typing import Dict, List, Union, Tuple, Any


_divisors_list_memo: Dict[int, List[int]] = {}


def divisors_list(n: int) -> List[int]:
    """
    Returns the ordered list of divisors, including 1,
    for instance:
    divisors_list(1) = [1]
    divisors_list(6) = [1, 2, 3, 6]
    divisors_list(97) = [1, 97]
    divisors_list(112) = [1, 2, 4, 7, 8, 14, 16, 28, 56, 112]
    """
    if n in _divisors_list_memo:
        return _divisors_list_memo[n]
    factors = []
    step = 1 if n % 2 == 0 else 2
    for i in range(1, int(np.sqrt(n)) + 1, step):
        if n % i == 0:
            factors.append(i)
            if n // i != i:
                factors.append(n // i)
    factors.sort()
    _divisors_list_memo[n] = factors
    return factors


_factors_enumeration_memo: Dict[Tuple[int, int], List[List[int]]] = {}


def factors_enumeration(n: int, n_factors: int) -> List[List[int]]:
    """
    Returns all valid combinations of n_factors for a number
    such that the multiplied factors divides the number.
    factorization into n_factors.
    Note that the list is ordered with smaller factors firsts.
    For instance:
    factors_enumeration(1, 2): [[1,1]]
    factors_enumeration(97, 2): [[1,1], [1,97], [97,1]]
    factors_enumeration(28, 2): [[1, 1],[1, 2],[1, 4],[1, 7],[1, 14],[1, 28],
                                 [2, 1],[2, 2],[2, 7],[2, 14],
                                 [4, 1],[4, 7],[7, 1],[7, 2],[7, 4],
                                 [14, 1],[14, 2],[28, 1]]
    """
    if (n, n_factors) in _factors_enumeration_memo:
        return _factors_enumeration_memo[(n, n_factors)]
    enumeration = []
    current = [0] * n_factors

    def enumerate(level: int, remain: int) -> None:
        if level >= n_factors:
            enumeration.append(current.copy())
        else:
            for factor in divisors_list(remain):
                current[level] = factor
                enumerate(level + 1, remain // factor)

    enumerate(0, n)
    _factors_enumeration_memo[(n, n_factors)] = enumeration
    return enumeration


def sizes_to_factors(splits: List[int]) -> List[int]:
    """Convert an array of outer-inner sizes to outer-inner factors
    For instance:
    sizes_to_factors([]) -> []
    sizes_to_factors([8]) -> [8]
    sizes_to_factors([16, 4, 4, 2]) -> [4, 1, 2, 2]
    """
    red = lambda seq, x: seq + [x // reduce(operator.mul, seq, 1)]
    return list(reversed(reduce(red, reversed(splits), [])))


def factors_to_sizes(splits: List[int]) -> List[int]:
    """Convert an array of outer-inner factors to outer-inner sizes
    For instance:
    factors_to_sizes([]) -> []
    factors_to_sizes([8]) -> [8]
    factors_to_sizes([4, 1, 2, 2]) -> [16, 4, 4, 2]
    """
    red = lambda seq, x: seq + [x * seq[-1] if len(seq) > 0 else x]
    return list(reversed(reduce(red, reversed(splits), [])))


def mulall(args: List[int]) -> int:
    """Multiply all args in list"""
    return reduce(operator.mul, args, 1)


class LazyImport:
    """
    Lazy load module:

        math = LazyLoader("math")
        math.cell(1.7)

    Ref to: https://stackoverflow.com/questions/4177735/best-practice-for-lazy-loading-python-modules
    """

    def __init__(self, modname):
        self._modname = modname
        self._mod = None

    def __getattr__(self, attr):
        """Import module on first attribute access"""
        if self._mod is None:
            import importlib

            self._mod = importlib.import_module(self._modname)
        return getattr(self._mod, attr)


def cpu_info() -> Dict[str, Any]:
    """
    Returm cpu info dict with fields:
    - ipc: max number of vec ops per cycle
    - vsize: dict of max elements per type ("float32", "float64, ...)
    - freq: frequency as cycles/sec
    - arch: arch ("x86_64", "aarch64", ...)
    """
    from cpuinfo import get_cpu_info

    vec_info_map = {
        "avx512": {
            "vsize": {
                "float32": 16,
                "float64": 8,
            }
        },
        "avx2": {
            "vsize": {
                "float32": 6,
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
    flags = info["flags"]
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
    else:
        cpu_info = {**vec_info_map["scalar"]}
        cpu_info["ipc"] = 1
    cpu_info["freq"] = info["hz_advertised"][0]
    cpu_info["arch"] = arch
    return cpu_info


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
