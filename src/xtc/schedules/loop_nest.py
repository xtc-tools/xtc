#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any, Generic, TypeVar
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

    @property
    def splits_to_sizes(self) -> dict[str, int]:
        splits_to_sizes: dict[str, int] = {}
        for axis in self.splits:
            last_start = None
            for loop_name, start in reversed(self.splits[axis].items()):
                if last_start is not None:
                    size_of_split = last_start - start
                    splits_to_sizes[loop_name] = size_of_split
                last_start = start
        return splits_to_sizes

    @property
    def tiles_to_sizes(self) -> dict[str, int]:
        tiles_to_sizes: dict[str, int] = {}
        for tiles in self.tiles.values():
            for loop, size in tiles.items():
                tiles_to_sizes[loop] = size
        return tiles_to_sizes


@dataclass
class LoopsDimsMapper:
    """Maps loop names to their corresponding axis names.

    This class tracks the relationship between loop identifiers (from tiling
    and splitting transformations) and the original dimension axes they
    derive from.

    Attributes:
        tiles_to_axis: Maps tile loop names to their parent axis.
        splits_to_axis: Maps split loop names to their parent axis.
        dims: List of original dimension names.
    """

    tiles_to_axis: dict[str, str]
    splits_to_axis: dict[str, str]
    dims: list[str]

    @property
    def loops_to_axis(self) -> dict[str, str]:
        loops_to_axis = (
            self.tiles_to_axis | self.splits_to_axis | dict(zip(self.dims, self.dims))
        )
        return loops_to_axis

    @staticmethod
    def build_from_nodes(nodes: list[LoopNestNode]) -> LoopsDimsMapper:
        tiles_to_axis = {}
        splits_to_axis = {}
        dims = set()
        for node in nodes:
            tiles_to_axis.update(LoopsDimsMapper._get_subloops_to_axis(node.tiles))
            splits_to_axis.update(LoopsDimsMapper._get_subloops_to_axis(node.splits))
        refined_loops = list(tiles_to_axis) + list(splits_to_axis)
        for node in nodes:
            dims.update(
                [loop for loop in node.interchange if loop not in refined_loops]
            )
            dims.update(tiles_to_axis.values())
            dims.update(splits_to_axis.values())
        return LoopsDimsMapper(tiles_to_axis, splits_to_axis, list(dims))

    @staticmethod
    def _get_subloops_to_axis(subloops: dict[str, dict[str, Any]]) -> dict[str, str]:
        loop_to_axis: dict[str, str] = {}
        for axis_name, subloops in subloops.items():
            for loop_name in subloops:
                loop_to_axis[loop_name] = axis_name
        return loop_to_axis


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
        self._check_use_defined_dims()
        self._check_vectorization_consistency()
        self._check_tiling_consistency()
        self._check_sizes()

    def _check_use_defined_dims(self):
        mapper = LoopsDimsMapper.build_from_nodes(self.nodes)
        for dim in self.abstract_dims:
            if dim not in mapper.dims:
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

    def _check_tiling_consistency(self) -> None:
        mapper = LoopsDimsMapper.build_from_nodes(self.nodes)
        seen_axes: dict[str, int | None] = {}
        for sched in self.nodes:
            for loop_name in sched.interchange:
                if loop_name in mapper.dims:
                    seen_axes[loop_name] = None
                elif loop_name in mapper.tiles_to_axis:
                    axis = mapper.tiles_to_axis[loop_name]
                    size = sched.tiles_to_sizes[loop_name]
                    if axis not in seen_axes:
                        raise ScheduleValidationError(
                            f"""
                            `{axis}#{size}`: {axis} has not been materialized yet.
                            """
                        )
                    seen_axes[axis] = sched.tiles[axis][loop_name]

    def _check_sizes(self):
        mapper = LoopsDimsMapper.build_from_nodes(self.nodes)
        current_size_of_split: dict[str, int | None] = {}
        for sched in self.nodes:
            current_size_of_tile: dict[str, int] = {}

            for loop_name in sched.interchange:
                axis = mapper.loops_to_axis[loop_name]
                current_sizes = (
                    {d: None for d in mapper.dims}
                    | current_size_of_split
                    | current_size_of_tile
                )
                loop_size = None
                if loop_name in mapper.dims:
                    if loop_name not in current_size_of_split:
                        current_size_of_split[loop_name] = None
                elif loop_name in mapper.tiles_to_axis:
                    loop_size = sched.tiles[axis][loop_name]
                    LoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_tile[axis] = loop_size
                elif (
                    loop_name in mapper.splits_to_axis
                    and loop_name in sched.splits_to_sizes
                ):
                    loop_size = sched.splits_to_sizes[loop_name]
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
