#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import importlib


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
            self._mod = importlib.import_module(self._modname)
        return getattr(self._mod, attr)
