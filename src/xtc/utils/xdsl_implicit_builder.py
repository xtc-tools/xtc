#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
safe_implicit_builder.py: re-implementation of ImplicitBuilder

xdsl ImplicitBuilder as of version 0.57 is not thread safe.

Install for instance early in your application, before import of ImplicitBuilder:

    install_locked_implicit_builder_as_default()
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Iterator
from typing import Any, NoReturn

import xdsl.builder as _xdsl_builder
from xdsl.builder import Builder
from xdsl.ir import Block, Region
from xdsl.ir.core import BlockArgument


# Capture the original before poisoning.
_ORIGINAL_IMPLICIT_BUILDER = _xdsl_builder.ImplicitBuilder

# Process-wide lock protecting the global Operation.__post_init__ monkey-patch
# performed internally by xdsl.builder.ImplicitBuilder.
_IMPLICIT_BUILDER_LOCK = threading.RLock()

# Optional guard flag to make diagnostics nicer.
_tls = threading.local()


@contextlib.contextmanager
def LockedImplicitBuilder(
    arg: Builder | Block | Region,
) -> Iterator[tuple[BlockArgument, ...]]:
    """
    Thread-safe wrapper around xdsl.builder.ImplicitBuilder.

    This serializes entry into xdsl's original ImplicitBuilder, which is needed
    because xdsl 0.57.x temporarily patches Operation.__post_init__ globally.

    Use exactly like xdsl.builder.ImplicitBuilder:

        block = Block()
        with LockedImplicitBuilder(block):
            arith.ConstantOp(...)
            func.ReturnOp()
    """
    depth = getattr(_tls, "locked_implicit_builder_depth", 0)

    with _IMPLICIT_BUILDER_LOCK:
        _tls.locked_implicit_builder_depth = depth + 1
        try:
            with _ORIGINAL_IMPLICIT_BUILDER(arg) as block_args:
                yield block_args
        finally:
            if depth == 0:
                try:
                    delattr(_tls, "locked_implicit_builder_depth")
                except AttributeError:
                    pass
            else:
                _tls.locked_implicit_builder_depth = depth


def _poisoned_implicit_builder(*args: Any, **kwargs: Any) -> NoReturn:
    raise RuntimeError(
        "Use of xdsl.builder.ImplicitBuilder is disabled in this process. "
        "Use safe_implicit_builder.LockedImplicitBuilder instead. "
        "xdsl 0.57.x ImplicitBuilder is not thread-safe because it globally "
        "patches Operation.__post_init__."
    )


def poison_xdsl_implicit_builder() -> None:
    """
    Replace xdsl.builder.ImplicitBuilder with a poisoned version.

    This catches code that does:

        import xdsl.builder
        with xdsl.builder.ImplicitBuilder(...):
            ...

    or imports ImplicitBuilder *after* this function has run.

    It cannot catch modules that already did:

        from xdsl.builder import ImplicitBuilder

    before poisoning, because those modules hold their own reference.
    """
    _xdsl_builder.ImplicitBuilder = _poisoned_implicit_builder  # type: ignore[assignment]


def install_locked_implicit_builder_as_default() -> None:
    """
    Alternative to poisoning: replace xdsl.builder.ImplicitBuilder with the
    locked implementation.

    This is less strict than poison_xdsl_implicit_builder(), but it preserves
    xdsl helpers that internally refer to xdsl.builder.ImplicitBuilder.
    """
    _xdsl_builder.ImplicitBuilder = LockedImplicitBuilder  # type: ignore[assignment]


def is_inside_locked_implicit_builder() -> bool:
    return getattr(_tls, "locked_implicit_builder_depth", 0) > 0
