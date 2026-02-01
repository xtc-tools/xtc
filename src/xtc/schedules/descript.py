#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any, Generic, TypeVar
from dataclasses import dataclass, field
import re
from typing_extensions import override
from xtc.itf.schd.scheduler import Scheduler


class ScheduleParseError(RuntimeError):
    """Raised when schedule parsing fails."""

    pass


class ScheduleInterpretError(RuntimeError):
    """Raised when schedule interpretation fails."""

    pass


class ScheduleValidationError(RuntimeError):
    """Raised when schedule validation fails."""

    pass


@dataclass(frozen=True)
class Annotations:
    """AST Type : annotations that can be applied to a loop.

    Attributes:
        unroll_factor: The unroll factor. None means "unroll fully" (use loop size).
            Only meaningful when unroll_specified is True.
        unroll_specified: True if unroll was explicitly requested.
        vectorize: True if vectorization was requested.
        parallelize: True if parallelization was requested.
    """

    unroll_factor: int | None = None
    unroll_specified: bool = False
    vectorize: bool = False
    parallelize: bool = False


@dataclass(frozen=True)
class SplitDecl:
    """AST Type: a split declaration like 'axis[start:end]'."""

    axis: str
    start: int | None
    end: int | None
    body: ScheduleSpec

    @override
    def __str__(self) -> str:
        start_str = "" if self.start is None else str(self.start)
        end_str = "" if self.end is None else str(self.end)
        decl = f"{self.axis}[{start_str}:{end_str}]"
        return decl


@dataclass(frozen=True)
class TileDecl:
    """AST Type: a tile declaration like 'axis#size'."""

    axis: str
    size: int
    annotations: Annotations

    @override
    def __str__(self) -> str:
        return f"{self.axis}#{self.size}"


@dataclass(frozen=True)
class AxisDecl:
    """AST Type: a direct axis reference."""

    axis: str
    annotations: Annotations


ScheduleItem = SplitDecl | TileDecl | AxisDecl


@dataclass(frozen=True)
class ScheduleSpec:
    """AST Type: the complete parsed schedule specification."""

    items: tuple[ScheduleItem, ...]


class ScheduleParser:
    """Parses a dict-based schedule specification into an AST."""

    _SPLIT_PATTERN = re.compile(r"^(.*)\[(-\d+|\d*)?:(-\d+|\d*)?\]$")

    def parse(self, spec: dict[str, Any]) -> ScheduleSpec:
        """Parse a schedule specification dict into an AST."""
        items: list[ScheduleItem] = []

        for declaration, value in spec.items():
            item = self._parse_declaration(declaration, value)
            items.append(item)

        return ScheduleSpec(items=tuple(items))

    def _parse_declaration(self, declaration: str, value: Any) -> ScheduleItem:
        """Parse a single declaration into a ScheduleItem."""
        assert isinstance(value, dict)
        # Try split declaration first (e.g., "axis[0:10]")
        if ":" in declaration:
            return self._parse_split(declaration, value)

        # Try tile declaration (e.g., "axis#32")
        if "#" in declaration:
            return self._parse_tile(declaration, value)

        # Must be a direct axis reference
        return self._parse_axis_ref(declaration, value)

    def _parse_split(self, declaration: str, value: dict) -> SplitDecl:
        """Parse a split declaration like 'axis[start:end]'."""
        axis_name, start, end = self._parse_split_syntax(declaration)

        body = self.parse(value)
        return SplitDecl(axis=axis_name, start=start, end=end, body=body)

    def _parse_tile(self, declaration: str, value: dict) -> TileDecl:
        """Parse a tile declaration like 'axis#size'."""
        parts = declaration.split("#")
        if len(parts) != 2:
            raise ScheduleParseError(
                f"`{declaration}`: invalid tile syntax, expected 'axis#size'"
            )

        axis_name, size_str = parts

        try:
            size = int(size_str)
        except ValueError:
            raise ScheduleParseError(f"`{declaration}`: {size_str} is not an integer.")

        annotations = self._parse_annotations(value, declaration)
        return TileDecl(axis=axis_name, size=size, annotations=annotations)

    def _parse_axis_ref(self, declaration: str, value: dict) -> AxisDecl:
        """Parse a direct axis reference."""

        annotations = self._parse_annotations(value, declaration)
        return AxisDecl(axis=declaration, annotations=annotations)

    def _parse_annotations(self, value: dict[str, Any], context: str) -> Annotations:
        """Parse annotation dict into Annotations object."""

        unroll_factor: int | None = None
        unroll_specified = False
        vectorize = False
        parallelize = False

        for key, param in value.items():
            if key == "unroll":
                if param is True:
                    unroll_factor = None
                    unroll_specified = True
                elif param is False:
                    pass
                elif isinstance(param, int):
                    unroll_factor = param
                    unroll_specified = True
                else:
                    raise ScheduleParseError(
                        f'`{{"unroll" = {param}}}`: unroll parameter should be True, False, or an integer.'
                    )
            elif key == "vectorize":
                if not isinstance(param, bool):
                    raise ScheduleParseError(
                        f'`{{"vectorize" = {param}}}`: parameterized vectorization not implemented.'
                    )
                vectorize = param
            elif key == "parallelize":
                if not isinstance(param, bool):
                    raise ScheduleParseError(
                        f'`{{"parallelize" = {param}}}`: parameterized parallelization not implemented.'
                    )
                parallelize = param
            else:
                raise ScheduleParseError(f"Unknown annotation on {context}: {key}")

        return Annotations(
            unroll_factor=unroll_factor,
            unroll_specified=unroll_specified,
            vectorize=vectorize,
            parallelize=parallelize,
        )

    def _parse_split_syntax(
        self, declaration: str
    ) -> tuple[str, int | None, int | None]:
        """Parse the syntax of a split declaration."""
        match = self._SPLIT_PATTERN.match(declaration)
        if not match:
            raise ScheduleParseError(f"Wrong format {declaration}")

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if x_str else None
        y = int(y_str) if y_str else None
        return prefix, x, y


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


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    spec: dict[str, dict[str, Any]],
) -> None:
    """Apply a schedule specification to a scheduler.

    This is the main entry point for using the descript scheduling DSL.

    Args:
        scheduler: The scheduler to apply the schedule to.
        node_name: The name of the root node to schedule.
        abstract_axis: The list of abstract axis names (e.g., ["m", "n", "k"]).
        spec: The schedule specification as a nested dict.
    """
    descript = Descript(scheduler=scheduler, abstract_axis=abstract_axis)
    descript.apply(node_name=node_name, spec=spec)


class ScheduleInterpreter:
    """Interprets a parsed ScheduleSpec AST into a LoopNest."""

    def __init__(self, abstract_axis: list[str]):
        self.abstract_axis = abstract_axis

    def interpret(self, spec: ScheduleSpec, root: str) -> LoopNest:
        """Interpret a schedule specification into a LoopNest."""
        loop_nest = LoopNest(abstract_dims=self.abstract_axis)
        root_node = loop_nest.build_root_node(root)
        self._interpret_spec_into_node(spec, root_node, root, head=[])
        return loop_nest

    def _interpret_spec_into_node(
        self,
        spec: ScheduleSpec,
        node: LoopNestNode,
        root: str,
        head: list[str],
    ) -> None:
        """Interpret a schedule spec into an existing node (mutates node)."""
        # Track state during interpretation
        sizes: dict[str, int] = {}
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = list(head)

        for item in spec.items:
            if isinstance(item, SplitDecl):
                self._interpret_split(item, node, root, interchange, previous_cut)
            elif isinstance(item, TileDecl):
                loop_name = self._interpret_tile(item, node, interchange, sizes)
                self._apply_annotations(item.annotations, loop_name, sizes, node)
            elif isinstance(item, AxisDecl):
                loop_name = self._interpret_axis(item, interchange)
                self._apply_annotations(item.annotations, loop_name, sizes, node)

        # Check that all splits are complete
        for axis, cut in previous_cut.items():
            if cut is not None and cut != 0:
                raise ScheduleInterpretError(
                    f"Splitting of {axis} unachieved (stops at {cut})."
                )

        node.interchange = interchange

    def _interpret_split(
        self,
        item: SplitDecl,
        node: LoopNestNode,
        root: str,
        interchange: list[str],
        previous_cut: dict[str, int | None],
    ) -> None:
        """Interpret a split declaration."""
        axis_name = item.axis
        self._check_axis_existence(axis_name)
        x = item.start
        y = item.end

        # The only declaration where y (the cut) is None is the
        # last one, so it cannot be the previous one.
        cut = previous_cut[axis_name]

        # When x (the starting point of the split) is not specified,
        # it is the previous cut
        if x is None:
            x = cut
        assert x is not None

        self._check_splitting_intervals(item, cut, x)

        # Update the previous cut
        previous_cut[axis_name] = y

        # Save the cutting points of the new dimensions
        if axis_name not in node.splits:
            node.splits[axis_name] = {}
        new_dim_index = len(node.splits[axis_name])
        new_dim_name = f"{axis_name}[{new_dim_index}]"
        new_root_name = f"{root}/{new_dim_name}"
        node.splits[axis_name][new_dim_name] = x
        interchange.append(new_dim_name)

        # Create a child node for the nested schedule
        child_node = LoopNestNode(
            root=new_root_name,
            tiles={a: {} for a in self.abstract_axis},
            split_origin=SplitOrigin(axis=axis_name, start=x, end=y),
        )
        node.add_child(child_node)

        # Recursively interpret the nested schedule into the child node
        self._interpret_spec_into_node(
            item.body, child_node, new_root_name, head=[axis_name]
        )

    def _interpret_tile(
        self,
        item: TileDecl,
        node: LoopNestNode,
        interchange: list[str],
        sizes: dict[str, int],
    ) -> str:
        """Interpret a tile declaration. Returns the loop name."""
        self._check_axis_existence(item.axis)
        tile_num = len(node.tiles[item.axis])
        loop_name = f"{item.axis}{tile_num}"
        if item.size <= 0:
            raise ScheduleInterpretError(
                f"`{item}`: tile sizes should be strictly positive."
            )
        node.tiles[item.axis][loop_name] = item.size
        sizes[loop_name] = item.size
        interchange.append(loop_name)

        return loop_name

    def _interpret_axis(
        self,
        item: AxisDecl,
        interchange: list[str],
    ) -> str:
        """Interpret a direct axis reference. Returns the loop name."""
        axis_name = item.axis
        self._check_axis_existence(axis_name)

        # Unreachable when built from a Python dict (because keys
        # can't be duplicated).
        if axis_name in interchange:
            raise ScheduleInterpretError(
                f"Axis {axis_name} is scheduled twice (or more)."
            )

        interchange.append(axis_name)
        return axis_name

    def _check_axis_existence(self, axis: str) -> None:
        """Check that an axis is defined."""
        if axis not in self.abstract_axis:
            raise ScheduleInterpretError(
                f"Axis {axis} is not a defined axis (defined axis: {self.abstract_axis})."
            )

    def _apply_annotations(
        self,
        annotations: Annotations,
        loop_name: str,
        sizes: dict[str, int],
        node: LoopNestNode,
    ) -> None:
        """Apply annotations to a loop in the node."""
        if annotations.unroll_specified:
            unroll_factor = annotations.unroll_factor
            if unroll_factor is None:
                # None means "unroll fully" - use the loop size
                if loop_name not in sizes:
                    raise ScheduleInterpretError(
                        f"{loop_name}'s size being unknown, an unroll factor is needed."
                    )
                unroll_factor = sizes[loop_name]
            elif unroll_factor <= 0:
                raise ScheduleInterpretError(
                    f'`{{"unroll" = {unroll_factor}}}`: unroll parameter should be strictly positive.'
                )
            node.unroll[loop_name] = unroll_factor

        if annotations.vectorize:
            node.vectorize.append(loop_name)

        if annotations.parallelize:
            node.parallelize.append(loop_name)

    def _check_splitting_intervals(
        self,
        item: SplitDecl,
        cut: int | None,
        x: int,
    ) -> None:
        """Check that split intervals are valid and contiguous."""

        if cut is None:
            raise ScheduleInterpretError(f"{item}: {item.axis} already covered.")

        if x > cut:
            raise ScheduleInterpretError(
                f"{item}: splitting doesn't fully cover {item.axis} (jumps from {cut} to {x})."
            )
        elif x < cut:
            raise ScheduleInterpretError(
                f"{item}: the segment begins at {x} but the previous one ends at {cut}."
            )

        if item.end is not None and x >= item.end:
            raise ScheduleInterpretError(
                f"{item}: the ending point should be greater than the starting point."
            )


@dataclass(frozen=True)
class Descript:
    """Applies a parsed and interpreted schedule to a Scheduler.

    This class coordinates the parsing, interpretation, and application
    of schedule specifications. The flow is:
    1. Parse: dict -> ScheduleSpec (AST)
    2. Interpret: ScheduleSpec -> LoopNest
    3. Validate: LoopNest.check()
    4. Apply: LoopNest -> Scheduler
    """

    scheduler: Scheduler
    abstract_axis: list[str]

    def apply(self, node_name: str, spec: dict[str, dict[str, Any]]) -> None:
        """Parse, interpret, validate, and apply a schedule specification.

        Args:
            node_name: The name of the root node to schedule.
            spec: The schedule specification as a nested dict.

        Raises:
            ScheduleParseError: If the spec cannot be parsed.
            ScheduleInterpretError: If the spec cannot be interpreted.
            ScheduleValidationError: If the resulting schedule is invalid.
        """
        # Parse the specification into an AST
        parser = ScheduleParser()
        ast = parser.parse(spec)

        # Interpret the AST into a LoopNest
        interpreter = ScheduleInterpreter(self.abstract_axis)
        loop_nest = interpreter.interpret(ast, root=node_name)

        # Validate the loop nest
        loop_nest.check()

        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest)

    def _apply_loop_nest(self, loop_nest: LoopNest) -> None:
        """Apply a LoopNest to the scheduler."""
        self.scheduler.set_dims(self.abstract_axis)

        if loop_nest.root_node is not None:
            self._apply_node(loop_nest.root_node)

    def _apply_node(self, node: LoopNestNode) -> None:
        """Recursively apply a LoopNestNode and its children to the scheduler."""
        root = node.root

        for d, s in node.splits.items():
            self.scheduler.split(d, s, root=root)

        for d, s in node.tiles.items():
            self.scheduler.tile(d, s, root=root)

        self.scheduler.interchange(node.interchange, root=root)
        self.scheduler.vectorize(node.vectorize, root=root)
        self.scheduler.parallelize(node.parallelize, root=root)
        self.scheduler.unroll(node.unroll, root=root)

        # Recursively apply children
        for child in node.children:
            self._apply_node(child)
