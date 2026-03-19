#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from xtc.schedules.loop_nest import LoopNest, LoopNestNode, NodeT, SplitOrigin

from .exceptions import ScheduleValidationError

literal = int | str


@dataclass
class ParameterSplitOrigin:
    """Describes how a parameterised node was created via a split from its parent.

    Attributes:
        axis: The axis that was split to create this node.
        start: The starting position of the split (inclusive), or None if unbounded.
        end: The ending position of the split (exclusive), or None if unbounded.
    """

    axis: str
    start: literal | None
    end: literal | None

    def apply_sample(self, sample: dict[str, int]) -> SplitOrigin:
        start = sample[self.start] if isinstance(self.start, str) else self.start
        end = sample[self.end] if isinstance(self.end, str) else self.end
        return SplitOrigin(self.axis, start, end)


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
    split_origin: ParameterSplitOrigin | None = None
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
class ParameterLoopNestNode(Node["ParameterLoopNestNode"]):
    """Represents a parameterised node in the loop nest tree with its transformations.

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
        buffer_at: Buffer configuration per axis. Maps axis names to optional
            memory types (mtype). None means default memory type.
        pack_at: Pack configuration per axis. Maps axis names to tuples of
            (input_idx, mtype, pad). input_idx is the input buffer index,
            mtype is the memory type (None for default), pad enables padding.
    """

    root: str
    tiles: dict[str, dict[str, literal]]
    splits: dict[str, dict[str, literal]] = field(default_factory=dict)
    interchange: list[str] = field(default_factory=list)
    vectorize: list[str] = field(default_factory=list)
    vectorize_parameters: dict[str, str] = field(default_factory=dict)
    parallelize: list[str] = field(default_factory=list)
    parallelize_parameters: dict[str, str] = field(default_factory=dict)
    unroll: dict[str, literal] = field(default_factory=dict)
    buffer_at: dict[str, str | None] = field(default_factory=dict)
    pack_at: dict[str, tuple[int, str | None, bool | str]] = field(default_factory=dict)
    constraints: set[str] = field(default_factory=set)

    def apply_sample(self, sample: dict[str, int]) -> LoopNestNode:
        root = self.root
        tiles = {
            a: {
                b: sample[v_b] if isinstance(v_b, str) else v_b
                for b, v_b in v_a.items()
            }
            for a, v_a in self.tiles.items()
        }
        splits = {
            a: {
                b: sample[v_b] if isinstance(v_b, str) else v_b
                for b, v_b in v_a.items()
            }
            for a, v_a in self.splits.items()
        }
        interchange = self.interchange
        vectorize = self.vectorize
        for a, v in self.vectorize_parameters.items():
            if sample.get(v, False):
                vectorize.append(a)
        parallelize = self.parallelize
        for a, v in self.parallelize_parameters.items():
            if sample.get(v, False):
                parallelize.append(a)
        unroll = {
            a: sample[v_a] if isinstance(v_a, str) else v_a
            for a, v_a in self.unroll.items()
        }
        buffer_at = self.buffer_at
        pack_at = {
            a: (a1, a2, bool(sample[a3]) if isinstance(a3, str) else a3)
            for a, (a1, a2, a3) in self.pack_at.items()
        }
        children = [child.apply_sample(sample) for child in self.children]
        split_origin = (
            self.split_origin.apply_sample(sample)
            if self.split_origin is not None
            else None
        )
        return LoopNestNode(
            root=root,
            tiles=tiles,
            splits=splits,
            interchange=interchange,
            vectorize=vectorize,
            parallelize=parallelize,
            unroll=unroll,
            buffer_at=buffer_at,
            pack_at=pack_at,
            children=children,
            split_origin=split_origin,
        )

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

        mapper = ParameterLoopInfo.build_from_node(self)
        tiles_info = mapper.tiles_info
        splits_info = mapper.splits_info

        # Map split loop names to their child nodes
        split_to_child: dict[str, ParameterLoopNestNode] = {}
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
        """Add annotations (parallelized, vectorized, unroll, buffer, pack) to a loop line."""
        annotations: list[str] = []
        if loop_name in self.parallelize:
            annotations.append("parallelized")
        if loop_name in self.vectorize:
            annotations.append("vectorized")
        if loop_name in self.unroll:
            annotations.append(f"unroll({self.unroll[loop_name]})")
        if loop_name in self.buffer_at:
            mtype = self.buffer_at[loop_name]
            if mtype is not None:
                annotations.append(f"buffer({mtype})")
            else:
                annotations.append("buffer")
        if loop_name in self.pack_at:
            input_idx, mtype, pad = self.pack_at[loop_name]
            parts = [str(input_idx)]
            if mtype is not None:
                parts.append(mtype)
            if pad:
                parts.append("pad")
            annotations.append(f"pack({', '.join(parts)})")
        if annotations:
            line += "  // " + ", ".join(annotations)
        return line


@dataclass
class ParameterLoopInfo:
    """Maps parameterised loop names to their corresponding axis names and metadata.

    This class tracks the relationship between loop identifiers (from tiling
    and splitting transformations) and the original dimension axes they
    derive from, along with their sizes and positions.

    Attributes:
        dims: List of original dimension names.
        tiles_info: Maps tile loop names to (axis, size) tuples.
        splits_info: Maps split loop names to (axis, start, end) tuples.
    """

    dims: list[str]
    tiles_info: dict[str, tuple[str, literal]] = field(default_factory=dict)
    splits_info: dict[str, tuple[str, literal, literal | None]] = field(
        default_factory=dict
    )

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
    def splits_to_sizes(self) -> dict[str, literal]:
        return {
            name: end - start
            if isinstance(end, int) and isinstance(start, int)
            else f"{end} - {start}"
            for name, (_, start, end) in self.splits_info.items()
            if end is not None
        }

    @staticmethod
    def build_from_node(node: ParameterLoopNestNode) -> ParameterLoopInfo:
        tiles_info: dict[str, tuple[str, literal]] = {}
        splits_info: dict[str, tuple[str, literal, literal | None]] = {}
        dims: dict[
            str, None
        ] = {}  # ordered set: insertion order preserved, no duplicates

        def collect(n: ParameterLoopNestNode) -> None:
            # Build tiles_info: tile_name -> (axis, size)
            for axis, tile_loops in n.tiles.items():
                for loop_name, size in tile_loops.items():
                    tiles_info[loop_name] = (axis, size)

            # Build splits_info: split_name -> (axis, start, end)
            for axis, axis_splits in n.splits.items():
                sorted_splits = sorted(axis_splits.items(), key=lambda kv: kv[1])
                for i, (loop_name, start) in enumerate(sorted_splits):
                    end = (
                        sorted_splits[i + 1][1] if i + 1 < len(sorted_splits) else None
                    )
                    splits_info[loop_name] = (axis, start, end)

            # Collect dims in stable order
            refined_loops = set(tiles_info) | set(splits_info)
            for loop in n.interchange:
                if loop not in refined_loops:
                    dims[loop] = None
            for axis, _ in tiles_info.values():
                dims[axis] = None
            for axis, _, _ in splits_info.values():
                dims[axis] = None

            # Recurse on children
            for child in n.children:
                collect(child)

        collect(node)

        return ParameterLoopInfo(list(dims), tiles_info, splits_info)


@dataclass
class ParameterLoopNest:
    """Represents a complete loop nest structure for scheduling.

    A loop nest contains abstract dimensions and a tree of nodes representing
    the schedule. Splits create child nodes, forming an explicit tree structure.

    Attributes:
        abstract_dims: List of abstract dimension names for the loop nest.
        root_node: The root node of the loop nest tree, or None if empty.
    """

    abstract_dims: list[str]
    root_node: ParameterLoopNestNode | None = None

    @property
    def nodes(self) -> list[ParameterLoopNestNode]:
        """Flatten the tree into a list of nodes.

        Returns nodes in depth-first order, with the root node first,
        followed by children in the order they were created.
        """
        if self.root_node is None:
            return []
        return [self.root_node] + self.root_node.descendants_dfs()

    def build_root_node(self, root: str) -> ParameterLoopNestNode:
        """Build and set the root node of the loop nest tree."""
        node = ParameterLoopNestNode(
            root=root, tiles={a: {} for a in self.abstract_dims}
        )
        self.root_node = node
        return node

    def check(self):
        assert self.root_node is not None
        info = ParameterLoopInfo.build_from_node(self.root_node)
        self._check_use_defined_dims(info)
        self._check_vectorization_consistency()
        self._check_tiling_consistency(info)
        self._check_sizes(info)

    def apply_sample(self, sample: dict[str, int]) -> LoopNest:
        root_node = (
            self.root_node.apply_sample(sample) if self.root_node is not None else None
        )
        return LoopNest(self.abstract_dims, root_node)

    def collect_constraints(self) -> set[str]:
        constraints = set()
        for node in self.nodes:
            constraints |= node.constraints
        return constraints

    def _check_use_defined_dims(self, info: ParameterLoopInfo):
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

    def _check_tiling_consistency(self, info: ParameterLoopInfo) -> None:
        seen_axes: dict[str, literal | None] = {}
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

    def _check_sizes(self, info: ParameterLoopInfo):
        current_size_of_split: dict[str, literal | None] = {}
        for sched in self.nodes:
            current_size_of_tile: dict[str, literal] = {}
            if sched.split_origin is not None:
                axis = sched.split_origin.axis
                start = sched.split_origin.start
                end = sched.split_origin.end
                if end is not None and start is not None:
                    current_size_of_split[axis] = (
                        end - start
                        if isinstance(end, int) and isinstance(start, int)
                        else f"{end} - {start}"
                    )
                else:
                    current_size_of_split[axis] = None

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
                    ParameterLoopNest._must_be_smaller_routine(
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
                    ParameterLoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_split[axis] = loop_size

                if loop_name in sched.unroll:
                    unroll_factor = sched.unroll[loop_name]
                    if (
                        loop_size
                        and isinstance(loop_size, int)
                        and isinstance(unroll_factor, int)
                        and loop_size < unroll_factor
                    ):
                        raise ScheduleValidationError(
                            f'`{{"unroll" = {unroll_factor}}}`: unroll factor should be smaller than {loop_size}.'
                        )

    @staticmethod
    def _must_be_smaller_routine(
        new_size: literal,
        current_sizes: dict[str, literal | None],
        loop_name: str,
        axis: str,
    ):
        old_size = current_sizes[axis]
        if (
            old_size is not None
            and isinstance(new_size, int)
            and isinstance(old_size, int)
            and new_size > old_size
        ):
            raise ScheduleValidationError(
                f"""
                Inner loop {loop_name} on axis {axis} must be smaller than outer loop.
                """
            )
