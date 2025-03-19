#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
import tempfile
import subprocess
import threading
import shlex
import logging
from pathlib import Path

__all__ = [
    "compile_runtime",
]

logger = logging.getLogger(__name__)

# Can be set to True for RUNTIME_DEBUG
RUNTIME_DEBUG = False

from xtc.runtimes.types.dlpack import DLDevice, DLDataType


class _c_ascii_str:
    @staticmethod
    def from_param(obj):
        if isinstance(obj, str):
            obj = obj.encode("ascii")
        return ctypes.c_char_p.from_param(obj)


_runtime_funcs = {
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


def compile_runtime(out_dll):
    debug_opts = "-DRUNTIME_DEBUG=1" if RUNTIME_DEBUG else ""
    files = ["evaluate.c", "cndarray.c", "alloc.c", "fclock.c", "evaluate_flops.c"]
    top_dir = Path(__file__).parents[2]
    src_dir = top_dir / "csrcs" / "runtimes" / "host"
    src_files = [f"{src_dir}/{file}" for file in files]
    cmd = (
        "cc --shared -O2 -march=native -fPIC "
        f"-I{src_dir} {debug_opts} "
        f"-o {out_dll} {' '.join(src_files)}"
    )
    logger.debug("Compiling runtime: %s", cmd)
    p = subprocess.run(shlex.split(cmd), text=True)
    assert p.returncode == 0, f"unable to compile runtime: {cmd}"


_runtime_lib_lock = threading.Lock()
_runtime_lib = None


def _compile():
    global _runtime_lib
    global _runtime_lib_lock
    if _runtime_lib is not None:
        return
    with _runtime_lib_lock:
        if _runtime_lib is not None:
            return
        with tempfile.TemporaryDirectory() as tdir:
            lib_path = f"{tdir}/runtime.so"
            compile_runtime(lib_path)
            _runtime_lib = ctypes.CDLL(lib_path)


_runtime_entries_lock = threading.Lock()
_runtime_entries = None


def _resolve_runtime():
    global _runtime_entries
    global _runtime_entries_lock
    if _runtime_entries is not None:
        return
    with _runtime_entries_lock:
        if _runtime_entries is not None:
            return
        _compile()
        _runtime_entries = {}
        for name, func_info in _runtime_funcs.items():
            _runtime_entries[name] = getattr(_runtime_lib, func_info["sym"])
            _runtime_entries[name].argtypes = func_info["argtypes"]
            _runtime_entries[name].restype = func_info["restype"]
            logger.debug(
                "Registring runtime function: %s: %s -> %s",
                name,
                _runtime_entries[name].argtypes,
                _runtime_entries[name].restype,
            )


def __getattr__(x):
    if x in _runtime_funcs:
        _resolve_runtime()
        return _runtime_entries[x]
    raise AttributeError(f"undefined runtime function: {x}")
