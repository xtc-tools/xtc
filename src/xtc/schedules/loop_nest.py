#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Generic, TypeVar
from dataclasses import dataclass, field

from .exceptions import ScheduleValidationError


@dataclass
class SplitOrigin:
    """Describes how a node was created via a split from its parent.

    Attributes:
        axis: The axis that was split to create this node.
        start: The starting position of the split (inclusive), or None if unbounded.
        end: The ending position of the split (exclusive), or None if unbounded.
    """

    axis: str
    start: int | None
    end: int | None


NodeT = TypeVar("NodeT", bound="Node")


@dataclass(kw_only=True)
class Node(Generic[NodeT]):
    """Base class for tree nodes with parent/child relationships.

    Provides tree structure and traversal operations. Subclasses add
    domain-specific data.

    Attributes:
        parent: Reference to the parent node, or None for the root.
        split_origin: Metadata describing how this node was created from
            its parent via a split. None for the root node.
        children: List of child nodes.
    """

    parent: NodeT | None = None
    split_origin: SplitOrigin | None = None
    children: list[NodeT] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        """Returns True if this node is the root (has no parent)."""
        return self.parent is None

    def add_child(self, child: NodeT) -> None:
        """Add a child node and set its parent to this node."""
        child.parent = self  # type: ignore[assignment]
        self.children.append(child)

    def ancestors(self) -> list[NodeT]:
        """Return list of ancestors from parent to root."""
        result: list[NodeT] = []
        current = self.parent
        while current is not None:
            result.append(current)
            current = current.parent
        return result

    def descendants_dfs(self) -> list[NodeT]:
        """Return all descendants in depth-first order."""
        result: list[NodeT] = []
        for child in self.children:
            result.append(child)
            result.extend(child.descendants_dfs())
        return result


@dataclass
class LoopNestNode(Node["LoopNestNode"]):
    """Represents a node in the loop nest tree with its transformations.

    Describes the loops attached to a single root and
    contains all the scheduling transformations applied to these loops.

    Attributes:
        root: Identifier of the node (either the base operation or
            the content of a split).
        tiles: Tiling configuration per axis. Maps axis names to dicts of
            tile loop names and their sizes.
        splits: Split configuration per axis. Maps axis names to dicts of
            split loop names and their starting positions.
        interchange: Ordered list of loop names defining the loop order.
        vectorize: List of loops to vectorize.
        parallelize: List of loops to parallelize.
        unroll: Maps loop names to their unroll factors.
    """

    root: str
    tiles: dict[str, dict[str, int]]
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    interchange: list[str] = field(default_factory=list)
    vectorize: list[str] = field(default_factory=list)
    parallelize: list[str] = field(default_factory=list)
    unroll: dict[str, int] = field(default_factory=dict)

    def pretty_print(self, indent: int = 0) -> str:
        """Return a human-readable representation of the loop nest.

        The output format uses a compact notation:
            - `loop X` for a regular loop over dimension X
            - `tile(X, N)` for a tile of size N on dimension X
            - `split(X, start, end)` for a split segment on dimension X
            - `// annotation` for vectorized, parallelized, unroll(N)
            - `...` for the innermost body

        Example output:
            loop i  // parallelized
              loop k
                loop j
                  tile(j, 16)  // vectorized
                    ...

        Args:
            indent: The initial indentation level (number of spaces).

        Returns:
            A multi-line string representing the loop nest structure.
        """
        lines: list[str] = []

        mapper = LoopInfo.build_from_node(self)
        tiles_info = mapper.tiles_info
        splits_info = mapper.splits_info

        # Map split loop names to their child nodes
        split_to_child: dict[str, LoopNestNode] = {}
        for child in self.children:
            if child.split_origin is not None:
                axis = child.split_origin.axis
                if axis in self.splits:
                    for loop_name, start in self.splits[axis].items():
                        if start == child.split_origin.start:
                            split_to_child[loop_name] = child
                            break

        # Group splits by axis for same-level printing
        axis_to_splits: dict[str, list[str]] = {}
        for loop_name, (axis, _, _) in splits_info.items():
            if loop_name in self.interchange:
                if axis not in axis_to_splits:
                    axis_to_splits[axis] = []
                axis_to_splits[axis].append(loop_name)

        processed_splits: set[str] = set()
        current_indent = indent

        for loop_name in self.interchange:
            # Skip already processed splits
            if loop_name in processed_splits:
                continue

            # Check if this is a split
            if loop_name in splits_info:
                axis, _, _ = splits_info[loop_name]
                axis_split_names = axis_to_splits.get(axis, [loop_name])
                processed_splits.update(axis_split_names)

                # Print all splits of this axis at the same level
                for split_name in axis_split_names:
                    split_axis, start, end = splits_info[split_name]
                    end_str = str(end) if end is not None else "..."
                    line = f"split({split_axis}, {start}, {end_str})"
                    line = self._add_annotations(line, split_name)
                    lines.append(" " * current_indent + line)

                    # Use child's pretty_print if available
                    if split_name in split_to_child:
                        child_output = split_to_child[split_name].pretty_print(
                            current_indent + 2
                        )
                        lines.append(child_output)
                    else:
                        lines.append(" " * (current_indent + 2) + "...")
            else:
                # Regular loop (tile or base dimension)
                if loop_name in tiles_info:
                    axis, size = tiles_info[loop_name]
                    line = f"tile({axis}, {size})"
                else:
                    # Extract basename (last part after /)
                    basename = loop_name.split("/")[-1]
                    line = f"loop {basename}"

                line = self._add_annotations(line, loop_name)
                lines.append(" " * current_indent + line)
                current_indent += 2

        # Add body if no splits were encountered
        if not processed_splits:
            lines.append(" " * current_indent + "...")

        return "\n".join(lines)

    def _add_annotations(self, line: str, loop_name: str) -> str:
        """Add annotations (parallelized, vectorized, unroll) to a loop line."""
        annotations: list[str] = []
        if loop_name in self.parallelize:
            annotations.append("parallelized")
        if loop_name in self.vectorize:
            annotations.append("vectorized")
        if loop_name in self.unroll:
            annotations.append(f"unroll({self.unroll[loop_name]})")
        if annotations:
            line += "  // " + ", ".join(annotations)
        return line


@dataclass
class LoopInfo:
    """Maps loop names to their corresponding axis names and metadata.

    This class tracks the relationship between loop identifiers (from tiling
    and splitting transformations) and the original dimension axes they
    derive from, along with their sizes and positions.

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

    @staticmethod
    def build_from_node(node: LoopNestNode) -> LoopInfo:
        tiles_info: dict[str, tuple[str, int]] = {}
        splits_info: dict[str, tuple[str, int, int | None]] = {}
        dims: set[str] = set()

        def collect(n: LoopNestNode) -> None:
            nonlocal dims

            # Build tiles_info: tile_name -> (axis, size)
            for axis, tile_loops in n.tiles.items():
                for loop_name, size in tile_loops.items():
                    tiles_info[loop_name] = (axis, size)

            # Build splits_info: split_name -> (axis, start, end)
            for axis, axis_splits in n.splits.items():
                split_starts = list(axis_splits.values())
                for i, (loop_name, start) in enumerate(axis_splits.items()):
                    end = split_starts[i + 1] if i + 1 < len(split_starts) else None
                    splits_info[loop_name] = (axis, start, end)

            # Collect dims
            refined_loops = set(tiles_info) | set(splits_info)
            dims.update(loop for loop in n.interchange if loop not in refined_loops)
            dims.update(axis for axis, _ in tiles_info.values())
            dims.update(axis for axis, _, _ in splits_info.values())

            # Recurse on children
            for child in n.children:
                collect(child)

        collect(node)

        return LoopInfo(list(dims), tiles_info, splits_info)


@dataclass
class LoopNest:
    """Represents a complete loop nest structure for scheduling.

    A loop nest contains abstract dimensions and a tree of nodes representing
    the schedule. Splits create child nodes, forming an explicit tree structure.

    Attributes:
        abstract_dims: List of abstract dimension names for the loop nest.
        root_node: The root node of the loop nest tree, or None if empty.
    """

    abstract_dims: list[str]
    root_node: LoopNestNode | None = None

    @property
    def empty(self) -> bool:
        return self.root_node is None

    @property
    def nodes(self) -> list[LoopNestNode]:
        """Flatten the tree into a list of nodes.

        Returns nodes in depth-first order, with the root node first,
        followed by children in the order they were created.
        """
        if self.root_node is None:
            return []
        return [self.root_node] + self.root_node.descendants_dfs()

    def build_root_node(self, root: str) -> LoopNestNode:
        """Build and set the root node of the loop nest tree."""
        node = LoopNestNode(root=root, tiles={a: {} for a in self.abstract_dims})
        self.root_node = node
        return node

    def check(self):
        assert self.root_node is not None
        info = LoopInfo.build_from_node(self.root_node)
        self._check_use_defined_dims(info)
        self._check_vectorization_consistency()
        self._check_tiling_consistency(info)
        self._check_sizes(info)

    def _check_use_defined_dims(self, info: LoopInfo):
        for dim in self.abstract_dims:
            if dim not in info.dims:
                raise ScheduleValidationError(f"{dim} defined but never used")

    def _check_vectorization_consistency(self):
        for sched in self.nodes:
            vect_above = False
            for loop_name in sched.interchange:
                if loop_name in sched.vectorize:
                    vect_above = True
                elif vect_above:
                    raise ScheduleValidationError(
                        f"Inner loop {loop_name} isn't vectorized but an outer one is."
                    )

    def _check_tiling_consistency(self, info: LoopInfo) -> None:
        seen_axes: dict[str, int | None] = {}
        for sched in self.nodes:
            for loop_name in sched.interchange:
                if loop_name in info.dims:
                    seen_axes[loop_name] = None
                elif loop_name in info.splits_to_axis:
                    axis = info.splits_to_axis[loop_name]
                    seen_axes[axis] = sched.splits[axis][loop_name]
                elif loop_name in info.tiles_to_axis:
                    axis = info.tiles_to_axis[loop_name]
                    size = sched.tiles[axis][loop_name]
                    if axis not in seen_axes:
                        raise ScheduleValidationError(
                            f"""
                            `{axis}#{size}`: {axis} has not been materialized yet.
                            """
                        )
                    seen_axes[axis] = size

    def _check_sizes(self, info: LoopInfo):
        current_size_of_split: dict[str, int | None] = {}
        for sched in self.nodes:
            current_size_of_tile: dict[str, int] = {}

            for loop_name in sched.interchange:
                axis = info.loops_to_axis[loop_name]
                current_sizes = (
                    {d: None for d in info.dims}
                    | current_size_of_split
                    | current_size_of_tile
                )
                loop_size = None
                if loop_name in info.dims:
                    if loop_name not in current_size_of_split:
                        current_size_of_split[loop_name] = None
                elif loop_name in info.tiles_to_axis:
                    loop_size = sched.tiles[axis][loop_name]
                    LoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_tile[axis] = loop_size
                elif (
                    loop_name in info.splits_to_axis
                    and loop_name in info.splits_to_sizes
                ):
                    loop_size = info.splits_to_sizes[loop_name]
                    LoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_split[axis] = loop_size

                if loop_name in sched.unroll:
                    unroll_factor = sched.unroll[loop_name]
                    if loop_size and loop_size < unroll_factor:
                        raise ScheduleValidationError(
                            f'`{{"unroll" = {unroll_factor}}}`: unroll factor should be smaller than {loop_size}.'
                        )

    @staticmethod
    def _must_be_smaller_routine(
        new_size: int, current_sizes: dict[str, int | None], loop_name: str, axis: str
    ):
        old_size = current_sizes[axis]
        if old_size is not None and new_size > old_size:
            raise ScheduleValidationError(
                f"""
                Inner loop {loop_name} on axis {axis} must be smaller than outer loop.
                """
            )
