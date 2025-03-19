#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes


class LibLoader:
    """
    Managed shared library loading.
    This is a simple wrapper arround ctypes.LoadLibary which provides
    context manager and a close() method to unload the loaded libary.
    Note that unless when exiting the context manager or
    with explicit call to close() method, the library is not unloaded.
    """

    def __init__(self, libpath: str) -> None:
        self._lib = ctypes.cdll.LoadLibrary(libpath)
        self._dlclose = ctypes.CDLL(None).dlclose
        self._dlclose.argtypes = [ctypes.c_void_p]
        self._dlclose.restype = ctypes.c_int

    @property
    def lib(self) -> ctypes.CDLL:
        """The ctypes lib handle"""
        return self._lib

    def close(self) -> None:
        """Unload the loaded library, must be called only once"""
        self._dlclose(self._lib._handle)
        self._lib = None

    def __enter__(self) -> ctypes.CDLL:
        """Context manager returns the ctypes lib handle"""
        return self._lib

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager unloads the library"""
        self.close()
