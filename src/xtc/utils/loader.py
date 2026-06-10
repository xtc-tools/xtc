#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
from typing import Any, cast
from pathlib import Path


class LibLoader:
    """
    Managed shared library loading.
    This is a simple wrapper arround ctypes.LoadLibary which provides
    context manager and a close() method to unload the loaded libary.
    Note that unless when exiting the context manager or
    with explicit call to close() method, the library is not unloaded.
    """

    def __init__(self, libpath: str | Path) -> None:
        self._lib: ctypes.CDLL | None = ctypes.cdll.LoadLibrary(str(libpath))
        assert self._lib is not None, f"unable to load libary: {libpath}"
        self._dlclose = cast(Any, ctypes.CDLL(None)).dlclose
        self._dlclose.argtypes = [ctypes.c_void_p]
        self._dlclose.restype = ctypes.c_int

    @property
    def lib(self) -> ctypes.CDLL:
        """The ctypes lib handle"""
        assert self._lib is not None, f"libary was closed"
        return self._lib

    def close(self) -> None:
        """Unload the loaded library, must be called only once"""
        assert self._lib is not None, f"libary was closed"
        self._dlclose(self._lib._handle)
        self._lib = None

    def __enter__(self) -> ctypes.CDLL:
        """Context manager returns the ctypes lib handle"""
        assert self._lib is not None, f"libary was closed"
        return self._lib

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Context manager unloads the library"""
        self.close()
