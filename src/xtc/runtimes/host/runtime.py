#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import ctypes.util
import tempfile
import subprocess
import threading
import shlex
import logging
from sys import platform
from pathlib import Path
from typing import Any
from enum import Enum

from xtc.utils.tools import get_mlir_prefix, get_cuda_prefix

__all__ = ["runtime_funcs", "resolve_runtime", "RuntimeType"]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

from xtc.runtimes.types.dlpack import DLDevice, DLDataType


class RuntimeType(Enum):
    HOST = 0
    GPU = 1


class _c_ascii_str:
    @staticmethod
    def from_param(obj: str | bytes):
        if isinstance(obj, str):
            obj = obj.encode("ascii")
        return ctypes.c_char_p.from_param(obj)


runtime_funcs: dict[str, dict[str, Any]] = {
    "evaluate": {
        "sym": "evaluate",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
        ],
        "restype": None,
    },
    "evaluate_perf": {
        "sym": "evaluate_perf",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
        ],
        "restype": None,
    },
    "evaluate_packed": {
        "sym": "evaluate_packed",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ],
        "restype": None,
    },
    "evaluate_packed_perf": {
        "sym": "evaluate_packed_perf",
        "argtypes": [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_voidp),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ],
        "restype": None,
    },
    "cndarray_new": {
        "sym": "CNDArray_new",
        "argtypes": [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            DLDataType,
            DLDevice,
        ],
        "restype": ctypes.c_voidp,
    },
    "cndarray_del": {
        "sym": "CNDArray_del",
        "argtypes": [
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_copy_from_data": {
        "sym": "CNDArray_copy_from_data",
        "argtypes": [
            ctypes.c_voidp,
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_copy_to_data": {
        "sym": "CNDArray_copy_to_data",
        "argtypes": [
            ctypes.c_voidp,
            ctypes.c_voidp,
        ],
        "restype": None,
    },
    "cndarray_set_alloc_alignment": {
        "sym": "CNDArray_set_alloc_alignment",
        "argtypes": [
            ctypes.c_int64,
        ],
        "restype": None,
    },
    "evaluate_flops": {
        "sym": "evaluate_flops",
        "argtypes": [
            _c_ascii_str,
        ],
        "restype": ctypes.c_double,
    },
}


def _compile_runtime(out_dll: str, tdir: str, runtime_type: RuntimeType):
    has_pfm = ctypes.util.find_library("pfm") is not None
    pfm_opts = "-DHAS_PFM=1" if has_pfm else ""
    pfm_libs = "-lpfm" if has_pfm else ""
    has_gpu = runtime_type == RuntimeType.GPU
    gpu_opts = "-DHAS_GPU=1" if has_gpu else ""
    debug_opts = "-DRUNTIME_DEBUG=1" if RUNTIME_DEBUG else ""
    files = [
        "evaluate_perf.c",
        "cndarray.c",
        "alloc.c",
        "fclock.c",
        "evaluate_flops.c",
        "perf_event_darwin.c" if platform == "darwin" else "perf_event_linux.c",
    ]
    top_dir = Path(__file__).parents[2]
    src_dir = top_dir / "csrcs" / "runtimes" / "host"
    src_files = [f"{src_dir}/{file}" for file in files]

    if has_gpu:
        # Compile runtime GPU extension
        gpu_lib_file = f"{tdir}/libgpu_runtime_extension.a"
        _compile_runtime_gpu_extension(gpu_lib_file, tdir)

    # Compile runtime
    obj_files = [f"{tdir}/{Path(file).stem}.o" for file in src_files]
    for i, file in enumerate(src_files):
        cmd = (
            "cc -c -O2 -march=native -fPIC "
            f"-I{src_dir} {debug_opts} {pfm_opts} {gpu_opts} -I{src_dir}/../accelerator/gpu "
            f"-o {obj_files[i]} {file}"
        )
        logger.debug("Compiling runtime: %s", cmd)
        p = subprocess.run(shlex.split(cmd), text=True)
        assert p.returncode == 0, f"unable to compile runtime: {cmd}"

    # Link runtime
    if has_gpu:
        mlir_lib_dir = get_mlir_prefix() / "lib"
        cuda_install_dir = get_cuda_prefix() / "lib64"
        cuda_libs = f"-L{mlir_lib_dir} -L{cuda_install_dir} -lgpu_runtime_extension -lmlir_cuda_runtime -lcupti -lcuda -lcudart"
    else:
        cuda_libs = ""
    cmd = (
        "c++ --shared -O2 -march=native -fPIC "
        f"-o {out_dll} {' '.join(obj_files)} "
        f"-L{tdir} "
        f"{pfm_libs} "
        f"{cuda_libs} "
    )
    logger.debug("Compiling runtime: %s", cmd)
    p = subprocess.run(shlex.split(cmd), text=True)
    assert p.returncode == 0, f"unable to compile runtime: {cmd}"


def _compile_runtime_gpu_extension(out_lib: str, tdir: str):
    debug_opts = "-DRUNTIME_DEBUG=1" if RUNTIME_DEBUG else ""
    files = [
        "perf_event_gpu.cpp",
    ]
    top_dir = Path(__file__).parents[2]
    src_dir = top_dir / "csrcs" / "runtimes" / "accelerator" / "gpu"
    src_files = [f"{src_dir}/{file}" for file in files]

    # Compile
    cuda_install_dir = get_cuda_prefix()
    obj_file = f"{tdir}/perf_event_gpu.o"
    cmd = (
        "c++ -c -O2 -march=native -fPIC "
        f"-I{src_dir} {debug_opts} -I{src_dir}/../../host "
        f"-I{cuda_install_dir}/include "
        f"-o {obj_file} {' '.join(src_files)}"
    )
    logger.debug("Compiling runtime GPU extension: %s", cmd)
    p = subprocess.run(shlex.split(cmd), text=True)
    assert p.returncode == 0, f"unable to compile runtime GPU extension: {cmd}"

    # Create static library
    cmd = f"ar rcs {out_lib} {obj_file}"
    logger.debug("Creating static library: %s", cmd)
    p = subprocess.run(shlex.split(cmd), text=True)
    assert p.returncode == 0, f"unable to create static library: {cmd}"


_runtime_libs_locks = [threading.Lock(), threading.Lock()]
_runtime_libs: list[ctypes.CDLL | None] = [None, None]


def _compile(runtime_type: RuntimeType):
    global _runtime_libs
    global _runtime_libs_locks
    if _runtime_libs[runtime_type.value] is not None:
        return
    with _runtime_libs_locks[runtime_type.value]:
        if _runtime_libs[runtime_type.value] is not None:
            return
        with tempfile.TemporaryDirectory() as tdir:
            lib_path = f"{tdir}/runtime.so"
            _compile_runtime(lib_path, tdir, runtime_type)
            _runtime_libs[runtime_type.value] = ctypes.CDLL(lib_path)


_runtime_entries_locks = [threading.Lock(), threading.Lock()]
_runtime_entries: list[dict | None] = [None, None]


def resolve_runtime(runtime_type: RuntimeType):
    global _runtime_entries
    global _runtime_entries_locks
    if _runtime_entries[runtime_type.value] is not None:
        return _runtime_entries[runtime_type.value]
    with _runtime_entries_locks[runtime_type.value]:
        if _runtime_entries[runtime_type.value] is not None:
            return _runtime_entries[runtime_type.value]
        _compile(runtime_type)
        entries = {}
        for name, func_info in runtime_funcs.items():
            entries[name] = getattr(_runtime_libs[runtime_type.value], func_info["sym"])
            entries[name].argtypes = func_info["argtypes"]
            entries[name].restype = func_info["restype"]
            logger.debug(
                "Registring runtime function: %s: %s -> %s",
                name,
                entries[name].argtypes,
                entries[name].restype,
            )
        _runtime_entries[runtime_type.value] = entries
        return _runtime_entries[runtime_type.value]
