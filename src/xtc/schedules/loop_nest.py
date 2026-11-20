#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

__all__ = ["LoopNest", "LoopNestNode", "Node"]


NodeT = TypeVar("NodeT", bound="Node")


@dataclass(frozen=True, kw_only=True)
class Node(Generic[NodeT]):
    """Base class for tree nodes with parent/child relationships.

    Provides tree structure and traversal operations. Subclasses add
    domain-specific data.

    Attributes:
        parent: Reference to the parent node, or None for the root.
        children: List of child nodes, all initially parentless, attached in
            order during construction.
    """

    parent: NodeT | None = None
    children: list[NodeT] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check all child ownership before attaching any constructor child."""
        children = self.children
        if any(child.parent is not None for child in children):
            raise AssertionError("Constructor children must not already have a parent.")
        object.__setattr__(self, "children", [])
        for child in children:
            self.add_child(child)

    @property
    def is_root(self) -> bool:
        """Returns True if this node is the root (has no parent)."""
        return self.parent is None

    def add_child(self, child: NodeT) -> None:
        """Add a child node and set its parent to this node.

        If the parent was already set, assert it's same.
        """
        if child.parent is not None:
            assert child.parent is self
        object.__setattr__(child, "parent", self)
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


@dataclass(frozen=True)
class LoopNestNode(Node["LoopNestNode"]):
    """Represents a node in the loop nest tree with its transformations.

    Describes the loops attached to a single root and
    contains all the scheduling transformations applied to these loops.
    In a complete loop nest, every non-leaf node declares the partition
    that creates its immediate children. Each child lists its incoming split
    axis in its local interchange. Other bare axes cannot name loops above the
    split, but remain valid dimension keys for tile declarations. Every local
    tile is listed in its node's interchange and resolves to an available loop.
    Only leaves may vectorize, and their vector loops must form a contiguous
    innermost group in the resolved order. Unroll, vector, and parallel
    annotations are mutually exclusive on each resolved loop.

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
        fuse_producer_at: Producer fusion configuration per axis. Maps axis
            names to producer indices.
        fuse_consumer_at: List of axes where the output consumer is fused.
    """

    root: str
    tiles: dict[str, dict[str, int]] = field(default_factory=dict)
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    interchange: list[str] = field(default_factory=list)
    vectorize: list[str] = field(default_factory=list)
    parallelize: list[str] = field(default_factory=list)
    unroll: dict[str, int] = field(default_factory=dict)
    buffer_at: dict[str, str | None] = field(default_factory=dict)
    pack_at: dict[str, tuple[int, str | None, bool]] = field(default_factory=dict)
    fuse_producer_at: dict[str, int] = field(default_factory=dict)
    fuse_consumer_at: list[str] = field(default_factory=list)
    external_at: dict[str, str] = field(default_factory=dict)

    def pretty_print(self, indent: int = 0, *, detailed: bool = False) -> str:
        """Render this node and its children, optionally labeling node paths."""
        from .loop_nest_pretty import pretty_print_loop_node

        return pretty_print_loop_node(self, indent, detailed=detailed)


@dataclass(frozen=True)
class LoopNest:
    """Represents a complete loop nest structure for scheduling.

    A loop nest contains abstract dimensions and a tree of nodes representing
    the schedule. Each split creates its immediate child nodes; structural
    relay nodes without a declared partition are not valid loop nests.

    Attributes:
        abstract_dims: List of abstract dimension names for the loop nest.
        root_node: The root node of the loop nest tree, or None if empty.
    """

    abstract_dims: list[str]
    root_node: LoopNestNode | None = None

    @property
    def nodes(self) -> list[LoopNestNode]:
        """Flatten the tree into a list of nodes.

        Returns nodes in depth-first order, with the root node first,
        followed by children in the order they were created.
        """
        if self.root_node is None:
            return []
        return [self.root_node] + self.root_node.descendants_dfs()

    def pretty_print(self, indent: int = 0, *, detailed: bool = False) -> str:
        """Render the root tree, optionally labeling node paths."""
        if self.root_node is None:
            return ""
        return self.root_node.pretty_print(indent, detailed=detailed)

    def check(self) -> None:
        """Validate this logical schedule without mutating its node tree."""
        from .loop_nest_analysis import check_loop_nest

        check_loop_nest(self)
