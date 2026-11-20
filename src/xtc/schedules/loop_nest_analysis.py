#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Backend-independent declaration, extent, and logical loop analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TypeAlias

from typing_extensions import override

from .exceptions import ScheduleValidationError
from .loop_nest import LoopNest, LoopNestNode


class LoopNestValidationError(ScheduleValidationError):
    """A logical validation error with optional schedule context."""

    def __init__(
        self,
        message: str,
        *,
        nest: LoopNest | None = None,
        node: LoopNestNode | None = None,
    ) -> None:
        super().__init__(message)
        self.nest = nest
        self.node = node

    @override
    def __str__(self) -> str:
        message = super().__str__()
        context = self.nest if self.nest is not None else self.node
        if context is None:
            return message
        heading = "Loop nest" if self.nest is not None else "Loop node"
        try:
            dump = context.pretty_print(detailed=True) or "<empty>"
        except Exception as exc:  # noqa: BLE001 - Never mask the validation error.
            dump = f"<pretty-print failed: {type(exc).__name__}>"
        return f"{heading}:\n{dump}\n\n{message}"


NodePath: TypeAlias = str
# Each requirement describes one node; whole-nest results map paths to these.
AxisExtentRequirement: TypeAlias = dict[str, int | None]
TileEntryRequirement: TypeAlias = dict[str, list[tuple[int, ...]]]


@dataclass(frozen=True)
class LoopNestAnalysis:
    """Checked snapshot of derived loop metadata.

    The result does not alias mutable analysis state, but its nested mappings
    are ordinary dictionaries and should be treated as read-only by callers.
    """

    node_info: dict[NodePath, LoopInfo]
    extents: dict[NodePath, AxisExtentRequirement]
    tile_entries: dict[NodePath, TileEntryRequirement]
    loop_contexts: dict[NodePath, dict[str, LogicalLoopContext]]
    shared_prefixes: dict[NodePath, dict[str, tuple[int, ...]]]


@dataclass(frozen=True)
class LoopInfo:
    """Snapshot of declarations on one node, not a subtree symbol table.

    Maps local loop identifiers to their original dimension axes, nominal tile
    spans and declared partition intervals. Whole-nest analysis indexes these
    records by node path so repeated branch-local names remain distinct.

    Attributes:
        dims: List of original dimension names.
        tiles_info: Maps tile loop names to (axis, size) tuples.
        splits_info: Maps split loop names to (axis, start, end) tuples.
    """

    dims: list[str]
    tiles_info: dict[str, tuple[str, int]] = field(default_factory=dict)
    splits_info: dict[str, tuple[str, int, int | None]] = field(default_factory=dict)

    @property
    def tiles_to_axis(self) -> dict[str, str]:
        return {name: axis for name, (axis, _) in self.tiles_info.items()}

    @property
    def splits_to_axis(self) -> dict[str, str]:
        return {name: axis for name, (axis, _, _) in self.splits_info.items()}

    @property
    def loops_to_axis(self) -> dict[str, str]:
        return (
            self.tiles_to_axis | self.splits_to_axis | dict(zip(self.dims, self.dims))
        )

    @property
    def splits_to_sizes(self) -> dict[str, int]:
        return {
            name: end - start
            for name, (_, start, end) in self.splits_info.items()
            if end is not None
        }


def collect_loop_node_info(node: LoopNestNode) -> LoopInfo:
    """Collect declarations on one node only; safe during builder construction."""
    tiles = {
        name: (axis, size)
        for axis, values in node.tiles.items()
        for name, size in values.items()
    }
    splits: dict[str, tuple[str, int, int | None]] = {}
    dims: dict[str, None] = {}
    for axis, values in node.splits.items():
        ordered = sorted(values.items(), key=lambda item: item[1])
        for i, (name, start) in enumerate(ordered):
            splits[name] = (
                axis,
                start,
                ordered[i + 1][1] if i + 1 < len(ordered) else None,
            )
    refined = set(tiles) | set(splits)
    for name in node.interchange:
        if name not in refined:
            dims[name] = None
    for axis, _ in tiles.values():
        dims[axis] = None
    for axis, _, _ in splits.values():
        dims[axis] = None
    return LoopInfo(list(dims), tiles, splits)


def iter_loop_nodes(nest: LoopNest) -> Iterator[tuple[NodePath, LoopNestNode]]:
    """Yield paths and nodes in DFS order, assuming valid acyclic topology.

    Checked analyzers establish this precondition before using the iterator.
    Paths remain unchanged when a nest is deep-copied without renaming nodes.
    """
    if nest.root_node is None:
        return
    stack = [(nest.root_node, nest.root_node.root)]
    while stack:
        node, path = stack.pop()
        yield path, node
        stack.extend(
            (child, f"{path}/{child.root}") for child in reversed(node.children)
        )


def _guard_topology(nest: LoopNest) -> None:
    """Reject unsafe cycles, reused nodes, parent links, and duplicate paths."""
    if nest.root_node is None:
        return
    seen: set[int] = set()
    paths: set[str] = set()
    stack: list[tuple[LoopNestNode, LoopNestNode | None, str]] = [
        (nest.root_node, None, nest.root_node.root)
    ]
    while stack:
        node, parent, path = stack.pop()
        if id(node) in seen:
            raise LoopNestValidationError(
                f"Loop-nest node {node.root} is reused or participates in a cycle.",
                node=node,
            )
        seen.add(id(node))
        if path in paths:
            raise LoopNestValidationError(
                f"Duplicate loop-nest node path {path}.", node=node
            )
        paths.add(path)
        if node.parent is not parent:
            raise LoopNestValidationError(
                f"Invalid parent link for loop-nest node {node.root}.", node=node
            )
        stack.extend(
            (child, node, f"{path}/{child.root}") for child in reversed(node.children)
        )


def _collect_axis_extents(
    nest: LoopNest,
    contexts: dict[NodePath, dict[str, LogicalLoopContext]],
) -> dict[NodePath, AxisExtentRequirement]:
    """Collect one boundary extent per axis and node, computed bottom-up.

    This summary does not represent every active tile level separately. A
    finite value is a known split/tile-bounded extent, ``None`` is an
    unknown full-axis extent, and an absent axis has no local requirement.
    Ordinary unrelated axes are omitted. Child requirements are hoisted
    only when the axis is materialized here and is not partitioned here.
    Requirements are never inherited by children. Lexical validation and
    code generation must retain their local loop/tile identities rather
    than treating this summary as a complete tile plan.
    """
    if nest.root_node is None:
        return {}

    requirements: dict[NodePath, AxisExtentRequirement] = {}

    def visit(
        node: LoopNestNode,
        path: str,
        incoming_axis: str | None = None,
    ) -> AxisExtentRequirement:
        local: AxisExtentRequirement = {}
        if incoming_axis is not None:
            # Logical traversal already resolved this branch's incoming range.
            # The base-axis alias retains that range even after local tiling.
            local[incoming_axis] = contexts[path][incoming_axis].nominal_extent

        for axis, tile_loops in node.tiles.items():
            if tile_loops:
                outermost = next(
                    name for name in node.interchange if name in tile_loops
                )
                if axis not in local:
                    # A base loop before its first tile exposes the whole
                    # dimension at this node's boundary.
                    if axis in node.interchange and node.interchange.index(
                        axis
                    ) < node.interchange.index(outermost):
                        local[axis] = None
                    else:
                        local[axis] = tile_loops[outermost]

        tile_axes = {
            axis
            for axis, tile_loops in node.tiles.items()
            if any(loop_name in tile_loops for loop_name in node.interchange)
        }
        split_axis = next(iter(node.splits), None)
        child_requirements = [
            visit(child, f"{path}/{child.root}", split_axis) for child in node.children
        ]
        candidate_axes = set(local).union(
            axis for child in child_requirements for axis in child
        )
        # A plain axis loop is relevant only when its subtree requires that
        # axis. Local tile/branch bounds take precedence; a full-axis loop
        # stays unknown rather than inheriting a finite descendant tile span.
        materialized_axes = tile_axes | (set(node.interchange) & candidate_axes)
        for axis in materialized_axes:
            local.setdefault(axis, None)

        # Propagate requirements for unrelated axes through real partitions
        # that do not materialize those axes. Split extents remain node-local:
        # alternatives from different branches must not be mistaken for a
        # shared extent.
        for axis in candidate_axes:
            if axis in local or axis in node.splits:
                continue
            child_values = [
                child[axis] for child in child_requirements if axis in child
            ]
            if child_values:
                known = {value for value in child_values if value is not None}
                local[axis] = (
                    next(iter(known))
                    if len(known) == 1 and None not in child_values
                    else None
                )

        requirements[path] = local
        return local

    visit(nest.root_node, nest.root_node.root)
    return requirements


@dataclass(frozen=True, eq=False)
class _Level:
    """A concrete logical loop identity within one branch scope.

    Equal axis/extent values in different branches still denote distinct
    loops. Extent is the nominal axis/tile range, not a quotient trip count.
    """

    axis: str
    extent: int | None


@dataclass(frozen=True)
class LogicalLoopContext:
    """Value snapshot of a named level after its scope's tile refinements.

    Unlike a private logical identity, its inner stride depends on the scope.
    Nominal span 64 above stride 8 represents eight trips, not 64 trips.
    """

    axis: str
    nominal_extent: int | None
    # A concrete inner tile span, or 1 for an innermost loop.
    next_inner_stride: int

    @property
    def trip_count(self) -> int | None:
        if self.nominal_extent is None:
            return None
        return (
            self.nominal_extent + self.next_inner_stride - 1
        ) // self.next_inner_stride


@dataclass
class _Scope:
    """Mutable lexical state for one logical walk branch.

    Copies retain loop identities while allowing child traversal to refine its
    own active levels and tile prefixes.
    """

    levels: list[_Level]
    # Bare dimension names identify original or rebased loops; tiling refines
    # active levels without changing what the bare names denote.
    base: dict[str, _Level]
    active: dict[str, _Level]
    prefixes: dict[str, list[_Level]] = field(default_factory=dict)

    def loop_context(self, level: _Level) -> LogicalLoopContext:
        # Trip counts follow refinement, not physical interchange order.
        prefix = self.prefixes.get(level.axis, [])
        index = prefix.index(level) + 1 if level in prefix else 0
        stride = prefix[index].extent if index < len(prefix) else 1
        # Only validated concrete tile levels enter the prefix.
        assert stride is not None
        return LogicalLoopContext(level.axis, level.extent, stride)

    def branch(self, inner: list[_Level], incoming: _Level) -> _Scope:
        axis = incoming.axis
        base = {name: level for name, level in self.base.items() if level in inner}
        base[axis] = incoming
        return _Scope(
            [incoming, *inner],
            base,
            {**self.active, axis: incoming},
            {
                name: list(chain)
                for name, chain in self.prefixes.items()
                if name != axis
            },
        )


def _ordered_tiles(node: LoopNestNode) -> dict[str, list[tuple[str, int]]]:
    result: dict[str, list[tuple[str, int]]] = {}
    for axis, tiles in node.tiles.items():
        result[axis] = [
            (name, tiles[name]) for name in node.interchange if name in tiles
        ]
    return result


def _shared_requirements(
    node: LoopNestNode,
    path: str,
    tiles: dict[str, list[tuple[str, int]]],
    entries: dict[NodePath, TileEntryRequirement],
    resolved_prefixes: dict[str, tuple[int, ...]],
) -> None:
    """Resolve the compatible entry of each level explicitly shared here.

    Requirements have already been collected bottom-up. This step checks their
    logical compatibility, not the ability of a backend to realize them using
    a particular sequence of primitives. Branch-local residual levels need not
    agree. Entries keep a full-axis request distinct from a finite tile request.
    """
    for axis, resolved in resolved_prefixes.items():
        prefix = list(resolved)
        if not prefix:
            alternatives = [
                chain
                for child in node.children
                for chain in entries[f"{path}/{child.root}"].get(axis, [()])
            ]
            raise LoopNestValidationError(
                f"Conflicting tile requirements for shared axis {axis} below {path}: "
                f"{alternatives}; every branch must have compatible shared tiles.",
                node=node,
            )
        existing = tiles.setdefault(axis, [])
        sizes = [size for _, size in existing]
        offset = sizes.index(prefix[0]) if prefix[0] in sizes else len(sizes)
        matched = 0
        for required, size in zip(prefix, sizes[offset:]):
            if required != size:
                break
            matched += 1
        remaining = prefix[matched:]
        if remaining and sizes and remaining[0] > sizes[-1]:
            raise LoopNestValidationError(
                f"Tile requirement {remaining[0]} for shared axis {axis} exceeds "
                f"its enclosing extent {sizes[-1]} at {path}.",
                node=node,
            )
        # These anonymous levels describe required shared refinements. Actual
        # tile names remain local to the nodes that declare them.
        existing.extend(
            (f"@{axis}:{index}", size) for index, size in enumerate(remaining)
        )


def _validate_logical_scopes(
    nest: LoopNest,
    entries: dict[NodePath, TileEntryRequirement],
    shared_prefixes: dict[NodePath, dict[str, tuple[int, ...]]],
) -> dict[NodePath, dict[str, LogicalLoopContext]]:
    """Validate logical scopes and return the read-only-by-convention snapshot."""
    root = nest.root_node
    assert root is not None  # Structural validation runs before this traversal.
    levels = [_Level(axis, None) for axis in nest.abstract_dims]
    dims = set(nest.abstract_dims)
    contexts: dict[NodePath, dict[str, LogicalLoopContext]] = {}

    def visit(node: LoopNestNode, path: str, scope: _Scope) -> None:
        aliases = dict(scope.base)
        for name in node.interchange:
            if name in dims and name not in aliases:
                raise LoopNestValidationError(
                    f"Loop {name} is outside its logical scope at {path}.", node=node
                )
        tiles = _ordered_tiles(node)
        # Diagnose incompatible prefixes here, in logical-walk order.
        _shared_requirements(node, path, tiles, entries, shared_prefixes[path])
        for axis, chain in tiles.items():
            if not chain:
                continue
            if axis not in scope.active:
                raise LoopNestValidationError(
                    f"Undefined tile axis {axis} at {path}.", node=node
                )
            previous = scope.prefixes.get(axis, [])
            start = next(
                (
                    index
                    for index, level in enumerate(previous)
                    if level.extent == chain[0][1]
                ),
                len(previous),
            )
            matched = 0
            for (name, size), level in zip(chain, previous[start:]):
                if size != level.extent:
                    break
                if name in node.tiles.get(axis, {}) and level not in scope.levels:
                    raise LoopNestValidationError(
                        f"Tile {name} on axis {axis} is outside its logical scope at {path}.",
                        node=node,
                    )
                aliases[name] = level
                matched += 1
            residual = chain[matched:]
            for name, size in residual:
                target = scope.active[axis]
                if target not in scope.levels:
                    if axis not in nest.abstract_dims:
                        raise LoopNestValidationError(
                            f"Cannot tile axis {axis} at {path}: loop is outside that scope.",
                            node=node,
                        )
                    scope.levels.append(target)
                if size <= 0:
                    raise LoopNestValidationError(
                        f"Tile {name} on axis {axis} must have positive size at {path}.",
                        node=node,
                    )
                if target.extent is not None and size > target.extent:
                    raise LoopNestValidationError(
                        f"Inner loop {name} on axis {axis} must be smaller than outer loop. "
                        f"Size {size} exceeds active extent {target.extent} at {path}.",
                        node=node,
                    )
                level = _Level(axis, size)
                scope.levels.insert(scope.levels.index(target) + 1, level)
                scope.active[axis] = level
                scope.prefixes.setdefault(axis, []).append(level)
                aliases[name] = level

        split_axis = next(iter(node.splits), None)
        split_target = scope.active.get(split_axis) if split_axis is not None else None
        if split_axis is not None and split_target not in scope.levels:
            raise LoopNestValidationError(
                f"Cannot split axis {split_axis} at {path}: loop is outside that scope.",
                node=node,
            )
        ordered = list(
            dict.fromkeys(
                aliases[name]
                for name in node.interchange
                if name in aliases and aliases[name] in scope.levels
            )
        )
        if split_target is not None:
            ordered = [level for level in ordered if level is not split_target] + [
                split_target
            ]
        scope.levels = list(dict.fromkeys(ordered + scope.levels))

        node_contexts: dict[str, LogicalLoopContext] = {}
        for name, level in aliases.items():
            if level not in scope.levels:
                continue
            node_contexts[name] = scope.loop_context(level)
        contexts[path] = node_contexts

        annotations = (
            set(node.vectorize)
            | set(node.parallelize)
            | set(node.unroll)
            | set(node.buffer_at)
            | set(node.pack_at)
            | set(node.fuse_producer_at)
            | set(node.fuse_consumer_at)
            | set(node.external_at)
        )
        for name in annotations:
            if name not in aliases or name not in node.interchange:
                raise LoopNestValidationError(
                    f"Undefined annotation loop {name} at {path}.", node=node
                )
            if aliases[name] not in scope.levels:
                raise LoopNestValidationError(
                    f"Annotation loop {name} is outside its logical scope at {path}.",
                    node=node,
                )

        execution_annotations: tuple[tuple[str, Iterable[str]], ...] = (
            ("unroll", node.unroll),
            ("vectorization", node.vectorize),
            ("parallelization", node.parallelize),
        )
        kind_by_level: dict[_Level, str] = {}
        for kind, names in execution_annotations:
            for name in names:
                previous_kind = kind_by_level.setdefault(aliases[name], kind)
                if previous_kind != kind:
                    raise LoopNestValidationError(
                        f"Cannot combine {previous_kind} and {kind} on loop {name} at {path}.",
                        node=node,
                    )

        for name, factor in node.unroll.items():
            if aliases[name] is split_target:
                raise LoopNestValidationError(
                    f"Cannot unroll partition target {name} at {path}.", node=node
                )
            if factor <= 0:
                raise LoopNestValidationError(
                    f"Unroll factor for {name} must be positive at {path}.", node=node
                )

        if node.vectorize:
            if node.children:
                raise LoopNestValidationError(
                    f"Vector annotations are only allowed in leaf nodes; "
                    f"{path} declares a partition.",
                    node=node,
                )
            vectors = {aliases[name] for name in node.vectorize}
            # The resolved order also includes loops omitted from interchange.
            first = next(i for i, level in enumerate(scope.levels) if level in vectors)
            for level in scope.levels[first + 1 :]:
                if level not in vectors:
                    name = next(
                        (n for n in node.interchange if aliases.get(n) is level),
                        level.axis,
                    )
                    raise LoopNestValidationError(
                        f"Inner loop {name} isn't vectorized but an outer one is. "
                        f"At {path}.",
                        node=node,
                    )

        if split_axis is None:
            # Structural validation guarantees that only leaves have no split.
            return

        assert split_target is not None
        branches = list(node.splits[split_axis].items())
        extent = split_target.extent
        if extent is not None and branches[-1][1] >= extent:
            raise LoopNestValidationError(
                f"Partition cut {branches[-1][1]} for axis {split_axis} is outside "
                f"its active extent {extent} at {path}.",
                node=node,
            )
        inner = scope.levels[scope.levels.index(split_target) + 1 :]
        for index, child in enumerate(node.children):
            start = branches[index][1]
            end = branches[index + 1][1] if index + 1 < len(branches) else extent
            incoming = _Level(split_axis, end - start if end is not None else None)
            visit(child, f"{path}/{child.root}", scope.branch(inner, incoming))

    base = dict(zip(nest.abstract_dims, levels))
    visit(root, root.root, _Scope(levels, base, dict(base)))
    return contexts


def _check_structure(nest: LoopNest) -> None:
    """Validate topology, declaration order and partition/child correspondence."""
    assert nest.root_node is not None
    _guard_topology(nest)
    for path, node in iter_loop_nodes(nest):
        interchange = set(node.interchange)
        for axis, tiles in node.tiles.items():
            for name in tiles:
                if name not in interchange:
                    raise LoopNestValidationError(
                        f"Tile {name} on axis {axis} must appear in the interchange at {path}.",
                        node=node,
                    )
            if any(inner > outer for outer, inner in pairwise(tiles.values())):
                raise LoopNestValidationError(
                    f"Tiles on axis {axis} must have non-increasing sizes "
                    f"in declaration order at {path}.",
                    node=node,
                )
        if len(node.splits) > 1:
            raise LoopNestValidationError(
                f"A loop-nest node may partition at most one axis; found "
                f"{', '.join(node.splits)}.",
                node=node,
            )
        if not node.splits:
            if node.children:
                raise LoopNestValidationError(
                    f"Loop-nest node {node.root} has children but declares no partition.",
                    node=node,
                )
            continue
        axis, branches = next(iter(node.splits.items()))
        if len(branches) < 2:
            raise LoopNestValidationError(
                f"Partition on axis {axis} must declare at least two branches.",
                node=node,
            )
        ordered = list(branches.items())
        starts = [start for _, start in ordered]
        if starts[0] != 0 or any(
            current <= previous for previous, current in pairwise(starts)
        ):
            raise LoopNestValidationError(
                f"Partition on axis {axis} must start at zero and have "
                "strictly increasing cuts in declaration order.",
                node=node,
            )
        if len(node.children) != len(ordered):
            raise LoopNestValidationError(
                f"Partition on axis {axis} declares {len(ordered)} branches "
                f"but has {len(node.children)} children.",
                node=node,
            )
        for (branch, _), child in zip(ordered, node.children):
            if child.root not in (branch, f"{node.root}/{branch}"):
                raise LoopNestValidationError(
                    f"Partition branch {branch} must have a matching child "
                    f"in declaration order; found {child.root}.",
                    node=child,
                )
            if axis not in child.interchange:
                raise LoopNestValidationError(
                    f"Split child {path}/{child.root} must include incoming axis "
                    f"{axis} in its interchange.",
                    node=child,
                )


def _check_names(nest: LoopNest, nodes: dict[NodePath, LoopNestNode]) -> None:
    """Check local names before declaration collection can overwrite collisions."""
    for path, node in nodes.items():
        names = set(nest.abstract_dims)
        for declarations in [*node.tiles.values(), *node.splits.values()]:
            for name in declarations:
                if name in names:
                    raise LoopNestValidationError(
                        f"Duplicate loop name {name} at {path}.", node=node
                    )
                names.add(name)
        seen: set[str] = set()
        for name in node.interchange:
            if name not in names:
                raise LoopNestValidationError(
                    f"Undefined interchange loop {name} at {path}.", node=node
                )
            if name in seen:
                raise LoopNestValidationError(
                    f"Duplicate interchange loop {name} at {path}.", node=node
                )
            seen.add(name)


def _check_use_defined_dims(
    nest: LoopNest, node_info: dict[NodePath, LoopInfo]
) -> None:
    used_dims = {dim for info in node_info.values() for dim in info.dims}
    for dim in nest.abstract_dims:
        if dim not in used_dims:
            raise LoopNestValidationError(f"{dim} defined but never used")


def _check_external_consistency(nest: LoopNest) -> None:
    for _, sched in iter_loop_nodes(nest):
        for axis in sched.external_at:
            if axis not in sched.interchange:
                raise LoopNestValidationError(
                    f"External axis {axis} is not part of the loop nest.", node=sched
                )
            axis_idx = sched.interchange.index(axis)
            descendants = sched.interchange[axis_idx + 1 :]
            if not descendants:
                raise LoopNestValidationError(
                    f"External axis {axis} has no nested tile to replace.", node=sched
                )
            conflicts = (
                set(descendants) & set(sched.vectorize)
                | set(descendants) & set(sched.unroll)
                | set(descendants) & set(sched.parallelize)
            )
            if conflicts:
                conflict_names = ", ".join(sorted(conflicts))
                raise LoopNestValidationError(
                    f"Loops below external axis {axis} have conflicting "
                    f"transformations: {conflict_names}.",
                    node=sched,
                )
            if sched.splits:
                raise LoopNestValidationError(
                    "External calls and split partitions cannot currently be "
                    "combined in the same loop nest.",
                    node=sched,
                )


def _check_tiling_consistency(
    nest: LoopNest, node_info: dict[NodePath, LoopInfo]
) -> None:
    if nest.root_node is None:
        return

    def visit(sched: LoopNestNode, path: str, incoming: set[str]) -> None:
        seen_axes = set(incoming)
        info = node_info[path]
        tiles_by_name = info.tiles_to_axis
        splits_by_name = info.splits_to_axis
        for loop_name in sched.interchange:
            if loop_name in tiles_by_name:
                axis = tiles_by_name[loop_name]
                size = info.tiles_info[loop_name][1]
                if axis not in seen_axes:
                    raise LoopNestValidationError(
                        f"""
                        `{axis}#{size}`: {axis} has not been materialized yet.
                        """,
                        node=sched,
                    )
                seen_axes.add(axis)
            elif loop_name in splits_by_name:
                seen_axes.add(splits_by_name[loop_name])
            elif loop_name in nest.abstract_dims or loop_name in info.dims:
                # Ordinary local references identify their own axis; refinement
                # names were resolved through declaration metadata above.
                seen_axes.add(loop_name)

        child_incoming = seen_axes | set(sched.splits)
        for child in sched.children:
            visit(child, f"{path}/{child.root}", child_incoming)

    visit(nest.root_node, nest.root_node.root, set())


def _common_prefix(chains: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not chains:
        return ()
    prefix = list(chains[0])
    for chain in chains[1:]:
        count = 0
        for left, right in zip(prefix, chain):
            if left != right:
                break
            count += 1
        prefix = prefix[:count]
    return tuple(prefix)


def _resolve_shared_prefixes(
    nest: LoopNest,
    entries: dict[NodePath, TileEntryRequirement],
    node_info: dict[NodePath, LoopInfo],
) -> dict[NodePath, dict[str, tuple[int, ...]]]:
    result: dict[NodePath, dict[str, tuple[int, ...]]] = {}
    for path, node in iter_loop_nodes(nest):
        info = node_info[path]
        prefixes: dict[str, tuple[int, ...]] = {}
        shared_axes = dict.fromkeys(
            axis
            for name in node.interchange
            for axis in info.dims
            if name == axis or name in node.tiles.get(axis, {})
        )
        for axis in shared_axes:
            if axis in node.splits or not node.children:
                continue
            alternatives = [
                chain
                for child in node.children
                for chain in entries[f"{path}/{child.root}"].get(axis, [()])
            ]
            if any(alternatives):
                prefixes[axis] = _common_prefix(alternatives)
        result[path] = prefixes
    return result


def check_loop_nest(nest: LoopNest) -> None:
    """Validate an executable loop nest without mutating its nodes."""
    analyze_loop_nest(nest)


def analyze_loop_nest(nest: LoopNest) -> LoopNestAnalysis:
    """Validate a nest and return the scoped analysis established by checking."""
    try:
        return _analyze_loop_nest(nest)
    except LoopNestValidationError as exc:
        exc.nest = nest
        raise


def _analyze_loop_nest(nest: LoopNest) -> LoopNestAnalysis:
    if nest.root_node is None:
        return LoopNestAnalysis({}, {}, {}, {}, {})
    _check_structure(nest)
    nodes = dict(iter_loop_nodes(nest))
    _check_names(nest, nodes)
    node_info = {path: collect_loop_node_info(node) for path, node in nodes.items()}
    _check_use_defined_dims(nest, node_info)
    _check_external_consistency(nest)
    _check_tiling_consistency(nest, node_info)
    entries = _collect_tile_entries(nest)
    shared_prefixes = _resolve_shared_prefixes(nest, entries, node_info)
    contexts = _validate_logical_scopes(nest, entries, shared_prefixes)
    return LoopNestAnalysis(
        node_info=node_info,
        extents=_collect_axis_extents(nest, contexts),
        tile_entries=entries,
        loop_contexts=contexts,
        shared_prefixes=shared_prefixes,
    )


def _collect_tile_entries(nest: LoopNest) -> dict[NodePath, TileEntryRequirement]:
    tile_entries: dict[NodePath, TileEntryRequirement] = {}

    def visit(
        node: LoopNestNode, path: str, incoming_axis: str | None = None
    ) -> TileEntryRequirement:
        child_axis = next(iter(node.splits), None)
        child_entries = [
            visit(child, f"{path}/{child.root}", child_axis) for child in node.children
        ]
        ordered_tiles = _ordered_tiles(node)
        axes = set(ordered_tiles) | {axis for child in child_entries for axis in child}
        root_leaf = node is nest.root_node and not node.children
        if root_leaf:
            axes.update(nest.abstract_dims)
        entries: TileEntryRequirement = {}
        partitioned = set(node.splits)
        if incoming_axis is not None:
            partitioned.add(incoming_axis)
        for axis in axes:
            tiles = ordered_tiles.get(axis, [])
            if tiles:
                outermost = tiles[0][0]
                if axis in node.interchange and node.interchange.index(
                    axis
                ) < node.interchange.index(outermost):
                    entries[axis] = [()]
                else:
                    entries[axis] = [tuple(size for _, size in tiles)]
            elif axis not in partitioned:
                if axis in node.interchange or (
                    root_leaf and axis in nest.abstract_dims
                ):
                    entries[axis] = [()]
                else:
                    values = [
                        chain
                        for child in child_entries
                        for chain in child.get(axis, [()])
                    ]
                    if values:
                        entries[axis] = values
        tile_entries[path] = entries
        return entries

    if nest.root_node is not None:
        visit(nest.root_node, nest.root_node.root)
    return tile_entries
