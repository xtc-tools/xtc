#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from dataclasses import dataclass

from xtc.itf.schd.scheduler import Scheduler
from .exceptions import ScheduleInterpretError
from .parsing import (
    ScheduleParser,
    ScheduleSpec,
    SplitDecl,
    TileDecl,
    AxisDecl,
    Annotations,
)
from .loop_nest import LoopNestNode, LoopNest, SplitOrigin


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_dims: list[str],
    spec: dict[str, dict[str, Any]],
) -> None:
    """Apply a schedule specification to a scheduler.

    This is the main entry point for using the descript scheduling DSL.

    Args:
        scheduler: The scheduler to apply the schedule to.
        node_name: The name of the root node to schedule.
        abstract_dims: The list of abstract axis names (e.g., ["m", "n", "k"]).
        spec: The schedule specification as a nested dict.
    """
    descript = Descript(scheduler=scheduler, abstract_dims=abstract_dims)
    descript.apply(node_name=node_name, spec=spec)


class ScheduleInterpreter:
    """Interprets a parsed ScheduleSpec AST into a LoopNest."""

    def __init__(self, abstract_dims: list[str]):
        self.abstract_dims = abstract_dims

    def interpret(self, spec: ScheduleSpec, root: str) -> LoopNest:
        """Interpret a schedule specification into a LoopNest."""
        loop_nest = LoopNest(abstract_dims=self.abstract_dims)
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
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_dims}
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
            tiles={a: {} for a in self.abstract_dims},
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
        if axis not in self.abstract_dims:
            raise ScheduleInterpretError(
                f"Axis {axis} is not a defined axis (defined axis: {self.abstract_dims})."
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

        if annotations.buffer_specified:
            node.buffer_at[loop_name] = annotations.buffer

        if annotations.pack_specified and annotations.pack is not None:
            node.pack_at[loop_name] = annotations.pack

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
    abstract_dims: list[str]

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
        interpreter = ScheduleInterpreter(self.abstract_dims)
        loop_nest = interpreter.interpret(ast, root=node_name)

        # Validate the loop nest
        loop_nest.check()

        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest)

    def _apply_loop_nest(self, loop_nest: LoopNest) -> None:
        """Apply a LoopNest to the scheduler."""
        self.scheduler.set_dims(self.abstract_dims)

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

        for axis, mtype in node.buffer_at.items():
            self.scheduler.buffer_at(axis, mtype=mtype, root=root)

        for axis, (input_idx, mtype, pad) in node.pack_at.items():
            self.scheduler.pack_at(axis, input_idx, mtype=mtype, pad=pad, root=root)

        # Recursively apply children
        for child in node.children:
            self._apply_node(child)
