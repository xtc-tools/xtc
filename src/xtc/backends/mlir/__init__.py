#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

# Install safe implementation of ImplicitBuilder
from xtc.utils.xdsl_implicit_builder import install_locked_implicit_builder_as_default

install_locked_implicit_builder_as_default()

from .MlirBackend import (
    MlirBackend,
)

from .MlirCompiler import (
    MlirCompiler,
)

from .MlirScheduler import (
    MlirScheduler,
    MlirSchedule,
)

from .MlirGraphBackend import MlirGraphBackend as Backend

__all__ = [
    "MlirBackend",
    "MlirCompiler",
    "MlirScheduler",
    "MlirSchedule",
    "Backend",
]
