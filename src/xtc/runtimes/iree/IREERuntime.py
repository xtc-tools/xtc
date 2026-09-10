#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import _ctypes
import ctypes
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from xtc.utils.cfunc import CFunc
from xtc.utils.tools import get_iree_sdk
from xtc.runtimes.host.HostRuntime import HostRuntime

__all__ = ["IREERuntime"]

# Directory holding the shim sources shipped with the package.
_SHIM_SRC_DIR = Path(__file__).parents[2] / "csrcs" / "runtimes" / "iree"

# Compile defines the shim needs on top of the installed IREE C SDK. These
# mirror scripts/iree/build_runtime.sh: IREE_ALLOCATOR_SYSTEM_CTL selects the
# libc system allocator (its default on Linux/macOS) so iree_allocator_system()
# is declared.
_SHIM_DEFINES = [
    "-DNDEBUG",
    "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl",
]

# Serializes the lazy shim build within a process.
_shim_build_lock = threading.Lock()

# Per-process build directory for the shim.
_shim_tmp_dir = tempfile.TemporaryDirectory()


def _shim_library_path() -> Path:
    """Path of the ``xtc_iree_shim`` shared library, building it if absent.

    The shim is compiled lazily from the IREE C SDK (headers + archives installed
    by ``scripts/iree/build_runtime.sh``) with a plain ``cc``/``c++`` invocation
    into a per-process temporary directory, so it is rebuilt once per process and
    always tracks the current shim sources.
    """
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = Path(_shim_tmp_dir.name) / "lib" / f"libxtc_iree_shim{ext}"
    if lib_path.exists():
        return lib_path
    with _shim_build_lock:
        if not lib_path.exists():
            _compile_shim(lib_path)
    return lib_path


def _compile_shim(lib_path: Path) -> None:
    """Compile ``iree_shim.c`` against the IREE C SDK into ``lib_path``."""
    include_dir, archives = get_iree_sdk()
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tdir:
        obj = f"{tdir}/iree_shim.o"
        # Link into the destination directory so the publish rename stays on the
        # same filesystem (and hence atomic).
        fd, out = tempfile.mkstemp(dir=lib_path.parent, suffix=lib_path.suffix)
        os.close(fd)
        compile_cmd = (
            f"cc -c -O2 -fPIC -std=gnu11 {' '.join(_SHIM_DEFINES)} "
            f"-I{_SHIM_SRC_DIR} -I{include_dir} "
            f"-o {obj} {_SHIM_SRC_DIR / 'iree_shim.c'}"
        )
        # Link with c++: some IREE archives carry C++ objects, so the C++ runtime
        # must be pulled in (as the host runtime does for its own shared library).
        link_cmd = (
            f"c++ -shared -fPIC -o {out} {obj} {' '.join(str(a) for a in archives)} -lm"
        )
        try:
            for cmd in (compile_cmd, link_cmd):
                proc = subprocess.run(shlex.split(cmd), text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"failed to build xtc_iree_shim: {cmd}")
            # Atomic publish so concurrent builders never expose a partial library.
            os.replace(out, lib_path)
        finally:
            if os.path.exists(out):
                os.unlink(out)


class _NDArrayDesc(ctypes.Structure):
    """Mirror of ``xtc_ndarray_desc_t`` in csrcs/runtimes/iree/iree_shim.h."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("rank", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("dtype", ctypes.c_char_p),
    ]


# numpy dtype name -> IREE element type spelling parsed by
# iree_hal_parse_element_type in the shim.
_NUMPY_TO_IREE_ELEMENT_TYPE: dict[str, bytes] = {
    "float64": b"f64",
    "float32": b"f32",
    "float16": b"f16",
    "bfloat16": b"bf16",
    "int64": b"i64",
    "int32": b"i32",
    "int16": b"i16",
    "int8": b"i8",
}


def _iree_dtype(dtype: np.dtype[Any]) -> bytes:
    name = str(dtype)
    if name not in _NUMPY_TO_IREE_ELEMENT_TYPE:
        raise NotImplementedError(f"IREE runtime: unsupported element type {name!r}")
    return _NUMPY_TO_IREE_ELEMENT_TYPE[name]


class IREERuntime:
    """Drives the ``xtc_iree_shim`` native library.

    Prepares an IREE invocation for a compiled ``.vmfb`` and times it through
    the shared C measurement loop.
    """

    _instance: "IREERuntime | None" = None
    _lib: ctypes.CDLL | None = None

    def __new__(cls) -> "IREERuntime":
        # Singleton: reuse the one instance so the native library is loaded and
        # its ctypes signatures wired up only once per process (see _library).
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _library(self) -> ctypes.CDLL:
        # Lazy-load: open the native library and wire up its ctypes signatures
        # on first use, then cache it in self._lib. Callers that never touch the
        # IREE runtime pay nothing, and later calls reuse the cached handle.
        if self._lib is None:
            lib = ctypes.CDLL(str(_shim_library_path()))
            lib.xtc_iree_setup.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(_NDArrayDesc),
                ctypes.c_int,
                ctypes.POINTER(_NDArrayDesc),
                ctypes.c_int,
            ]
            lib.xtc_iree_setup.restype = ctypes.c_void_p
            lib.xtc_iree_invoke.argtypes = [ctypes.c_void_p]
            lib.xtc_iree_invoke.restype = None
            lib.xtc_iree_invoke_readback.argtypes = [ctypes.c_void_p]
            lib.xtc_iree_invoke_readback.restype = None
            lib.xtc_iree_teardown.argtypes = [ctypes.c_void_p]
            lib.xtc_iree_teardown.restype = None
            lib.xtc_iree_last_error.argtypes = []
            lib.xtc_iree_last_error.restype = ctypes.c_char_p
            self._lib = lib
        return self._lib

    def close(self) -> None:
        """Unload the native shim library and drop the cached singleton.

        Invocation contexts are already torn down per-evaluation (see
        IREEEvaluator), and the shim keeps no global IREE state, so the only
        process-level resource left is the dlopen handle: release it here. A
        subsequent IREERuntime() reloads the library on demand.
        """
        lib = self._lib
        if lib is not None:
            self._lib = None
            IREERuntime._instance = None
            # _ctypes may already be torn down when called from __del__ at
            # interpreter shutdown; guard so finalization never raises.
            if _ctypes is not None:
                try:
                    _ctypes.dlclose(lib._handle)
                except Exception:
                    pass

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _make_descs(arrays: list[np.ndarray]) -> Any:
        # Assigning the shape arrays and dtype bytes to the descriptors'
        # pointer fields makes ctypes keep them alive as long as the returned
        # array lives, so the local ``shape`` outliving this loop is fine.
        descs = (_NDArrayDesc * len(arrays))()
        for i, arr in enumerate(arrays):
            shape = (ctypes.c_int64 * arr.ndim)(*arr.shape)
            descs[i].data = arr.ctypes.data
            descs[i].rank = arr.ndim
            descs[i].shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
            descs[i].dtype = _iree_dtype(arr.dtype)
        return descs

    def setup(
        self,
        vmfb_path: str,
        entry_function: str,
        num_threads: int,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
    ) -> ctypes.c_void_p:
        """Prepare an invocation context; results land in ``outputs`` in place.

        ``num_threads`` sizes the IREE worker pool: <= 1 runs on local-sync
        (inline), > 1 uses local-task with that many P-core workers.
        """
        lib = self._library()
        in_descs = self._make_descs(inputs)
        out_descs = self._make_descs(outputs)
        # IREE expects a fully-qualified import name (module.func); XTC compiles
        # to an unnamed MLIR module, which IREE names "module".
        if "." not in entry_function:
            entry_function = f"module.{entry_function}"
        ctx = lib.xtc_iree_setup(
            vmfb_path.encode("ascii"),
            entry_function.encode("ascii"),
            int(num_threads),
            in_descs,
            len(inputs),
            out_descs,
            len(outputs),
        )
        if not ctx:
            err = lib.xtc_iree_last_error()
            msg = err.decode("utf-8") if err else "unknown error"
            raise RuntimeError(f"IREE runtime setup failed: {msg}")
        return ctypes.c_void_p(ctx)

    def invoke(self, ctx: ctypes.c_void_p) -> None:
        """Run one invocation with host read-back (validation / write-back)."""
        self._library().xtc_iree_invoke_readback(ctx)

    def teardown(self, ctx: ctypes.c_void_p) -> None:
        self._library().xtc_iree_teardown(ctx)

    def evaluate_perf(
        self,
        ctx: ctypes.c_void_p,
        pmu_events: list[str],
        repeat: int,
        number: int,
        min_repeat_ms: int,
    ) -> list[float]:
        """Time ``xtc_iree_invoke(ctx)`` through the shared C measurement loop."""
        invoke = self._library().xtc_iree_invoke
        args = (ctypes.c_void_p * 1)(ctx)
        return HostRuntime().evaluate_perf(
            pmu_events,
            repeat,
            number,
            min_repeat_ms,
            CFunc(invoke),
            args,
            1,
        )
