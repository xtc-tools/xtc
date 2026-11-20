#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Backend-independent preparation of vectorization and unroll annotations."""

from __future__ import annotations

from copy import deepcopy
from typing import TypeVar

from xtc.utils.math import pow2divisor

from .exceptions import ScheduleValidationError
from .loop_nest import LoopNest, LoopNestNode
from .loop_nest_analysis import (
    LoopNestAnalysis,
    analyze_loop_nest,
    iter_loop_nodes,
)


def _checked_copy(nest: LoopNest) -> tuple[LoopNest, LoopNestAnalysis]:
    analysis = analyze_loop_nest(nest)
    # Scalar snapshots are keyed by paths, which copying preserves. No original
    # node identities or mutable traversal scopes escape into the rewrite.
    return deepcopy(nest), analysis


def _reserved_names(nest: LoopNest) -> set[str]:
    names = set(nest.abstract_dims)
    for node in nest.nodes:
        names.update(node.interchange)
        names.update(name for tiles in node.tiles.values() for name in tiles)
        names.update(name for branches in node.splits.values() for name in branches)
    return names


def _fresh_name(node: LoopNestNode, stem: str, reserved: set[str]) -> str:
    used = reserved | set(node.interchange)
    name = stem
    suffix = 0
    while name in used:
        suffix += 1
        name = f"{stem}_{suffix}"
    return name


def _insert_tile(
    node: LoopNestNode, axis: str, target: str, name: str, span: int
) -> None:
    node.interchange.insert(node.interchange.index(target) + 1, name)
    tiles = node.tiles.setdefault(axis, {})
    tiles[name] = span
    # Both the nest and its consumers see enclosing-to-inner span order.
    ordered = [loop for loop in node.interchange if loop in tiles]
    ordered.extend(loop for loop in tiles if loop not in ordered)
    node.tiles[axis] = {loop: tiles[loop] for loop in ordered}


_Placement = TypeVar("_Placement")


def _retarget_placements(node: LoopNestNode, outer: str, inner: str) -> None:
    """Keep placements below the loop representing the original iterations."""

    def retarget(placements: dict[str, _Placement]) -> None:
        if outer in placements:
            renamed = {
                inner if name == outer else name: value
                for name, value in placements.items()
            }
            placements.clear()
            placements.update(renamed)

    retarget(node.buffer_at)
    retarget(node.pack_at)
    retarget(node.fuse_producer_at)
    node.fuse_consumer_at[:] = [
        inner if name == outer else name for name in node.fuse_consumer_at
    ]
    retarget(node.external_at)


def _check_result(nest: LoopNest, transform: str) -> LoopNest:
    try:
        nest.check()
    except ScheduleValidationError as exc:
        raise ScheduleValidationError(
            f"{transform} normalization failed: {exc}"
        ) from exc
    return nest


def loop_nest_vectorize_as_pow2(nest: LoopNest) -> LoopNest:
    """Factor a single vector loop per node with a known trip count.

    A known width of 48 becomes three fully unrolled groups of 16 vector lanes,
    using the largest power-of-two divisor. The original loop becomes the group
    loop without changing its span. Multi-loop vector groups, unknown widths,
    and widths already a power of two are left unchanged.

    Return an independently owned, validated nest without modifying the input.
    The transform is structurally idempotent.
    """
    result, analysis = _checked_copy(nest)
    reserved = _reserved_names(result)
    for path, node in iter_loop_nodes(result):
        if len(node.vectorize) != 1:
            continue
        (target,) = node.vectorize
        context = analysis.loop_contexts[path][target]
        trips = context.trip_count
        if trips is None:
            continue
        lanes = pow2divisor(trips)
        if lanes == trips:
            continue
        lane = _fresh_name(node, f"__v_{target}", reserved)
        lane_span = lanes * context.next_inner_stride
        _insert_tile(node, context.axis, target, lane, lane_span)
        node.vectorize[0] = lane
        node.unroll[target] = trips // lanes
    return _check_result(result, "Vector")


def loop_nest_unroll_as_full(nest: LoopNest) -> LoopNest:
    """Tile factor-unroll requests for backends supporting only full unrolling.

    For example, unrolling an untiled loop by four introduces a four-trip inner
    tile, then fully unrolls it. A factor covering every known trip is kept
    unchanged on the original loop, with no additional tile. Factor-one
    requests are no-ops. Tile spans differ from trip counts: unrolling the
    64-span loop above an 8-span tile by four introduces span 32, not span 4.
    Placement directives on a split loop move to the new inner loop.

    Return an independently owned, validated nest without modifying the input,
    including on failure. The transform is structurally idempotent. Unsupported
    rewrites, such as changing a partition target incompatibly, raise
    ScheduleValidationError rather than returning an invalid nest.
    """
    result, analysis = _checked_copy(nest)
    contexts = analysis.loop_contexts
    reserved = _reserved_names(result)
    for path, node in iter_loop_nodes(result):
        unroll: dict[str, int] = {}
        for target, factor in node.unroll.items():
            if factor == 1:
                # Do not turn a no-op into an enclosing tile that could take
                # a shared refinement out of scope in the branches.
                continue
            context = contexts[path][target]
            trips = context.trip_count
            if trips is not None and factor >= trips:
                unroll[target] = factor
                continue
            span = factor * context.next_inner_stride
            inner = _fresh_name(node, f"__u_{target}", reserved)
            _insert_tile(node, context.axis, target, inner, span)
            _retarget_placements(node, target, inner)
            unroll[inner] = factor
        node.unroll.clear()
        node.unroll.update(unroll)
    return _check_result(result, "Full-unroll")
