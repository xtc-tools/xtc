#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# type: ignore
from .operations import (
    register_operation,
    get_operation,
    list_operations,
    has_operation,
)

from .register_ttile_operations import _register_operations
from .register_subgraph_operations import (
    _register_operations as _register_subgraph_operations,
)

_register_operations()
_register_subgraph_operations()
