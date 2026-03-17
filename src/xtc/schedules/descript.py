#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from dataclasses import dataclass, field
from copy import deepcopy

from xtc.itf.schd.scheduler import Scheduler, ROOT_SEP, SPLIT_LEFT_SEP, SPLIT_RIGHT_SEP
from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.parameter_loop_nest import (
    ParameterLoopNest,
    ParameterLoopNestNode,
    ParameterSplitOrigin,
)
from .exceptions import ScheduleInterpretError
from .parsing import (
    ScheduleParser,
    ScheduleSpec,
    SplitDecl,
    TileDecl,
    AxisDecl,
    Annotations,
    YAMLParser,
)

literal = int | str


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
    descript = Descript(abstract_dims=abstract_dims)
    descript.apply(node_name=node_name, spec=spec, scheduler=scheduler)


@dataclass
class ScheduleInterpreter:
    """Interprets a parsed ScheduleSpec AST into a LoopNest."""

    abstract_dims: list[str]
    abstract_dim_sizes: dict[str, int] = field(default_factory=dict)
    abstract_matrix: list[str] = field(default_factory=list)
    partial_tiles = False
    partial_unfolds = False

    def interpret(self, spec: ScheduleSpec, root: str) -> ParameterLoopNest:
        """Interpret a schedule specification into a LoopNest."""
        loop_nest = ParameterLoopNest(abstract_dims=self.abstract_dims)
        root_node = loop_nest.build_root_node(root)
        self._interpret_spec_into_node(spec, root_node, root, head=[])
        return loop_nest

    def _interpret_spec_into_node(
        self,
        spec: ScheduleSpec,
        node: ParameterLoopNestNode,
        root: str,
        head: list[str],
        sizes: dict[str, list[literal]] | None = None,
    ) -> None:
        """Interpret a schedule spec into an existing node (mutates node)."""
        # Track state during interpretation
        previous_cut: dict[str, literal | None] = {a: 0 for a in self.abstract_dims}
        interchange: list[str] = list(head)
        last_split: list[tuple[literal, literal | None]] = []
        if not sizes:
            if self.abstract_dim_sizes:
                sizes = {a: [v] for a, v in self.abstract_dim_sizes.items()}
            else:
                sizes = {a: [] for a in self.abstract_dims}

        for item in spec.items:
            if isinstance(item, SplitDecl):
                self._interpret_split(
                    item=item,
                    node=node,
                    root=root,
                    interchange=interchange,
                    previous_cut=previous_cut,
                    sizes=sizes,
                    last_split=last_split,
                )
            elif isinstance(item, TileDecl):
                loop_name = self._interpret_tile(
                    item=item, node=node, interchange=interchange, sizes=sizes
                )
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
        node: ParameterLoopNestNode,
        root: str,
        interchange: list[str],
        previous_cut: dict[str, literal | None],
        sizes: dict[str, list[literal]],
        last_split: list[tuple[literal, literal | None]],
    ) -> None:
        """Interpret a split declaration."""
        axis_name = item.axis
        self._check_axis_existence(axis_name)
        x = item.start
        y = item.end
        z = item.size

        # The only declaration where y (the cut) is None is the
        # last one, so it cannot be the previous one.
        cut = previous_cut[axis_name]

        current_size = sizes[axis_name][-1] if sizes[axis_name] else None

        # When x (the starting point of the split) is not specified,
        # it is the previous cut
        if x is None:
            x = cut
        assert x is not None

        self._check_splitting_intervals(item, cut, x)

        if axis_name not in node.splits:
            node.splits[axis_name] = {}
        new_dim_index = len(node.splits[axis_name])
        new_dim_name = f"{axis_name}{SPLIT_LEFT_SEP}{new_dim_index}{SPLIT_RIGHT_SEP}"
        new_root_name = f"{root}{ROOT_SEP}{new_dim_name}"

        if z is None:
            # Update the previous cut
            previous_cut[axis_name] = y

            # Save the cutting points of the new dimensions
            node.splits[axis_name][new_dim_name] = x
            interchange.append(new_dim_name)

            inner_size = None
            if y is None:
                y = current_size
            if isinstance(x, int):
                if x == 0:
                    inner_size = y
                elif isinstance(y, int):
                    inner_size = y - x
            if inner_size is None:
                inner_size = root[1:] + new_dim_name
                inner_size = (
                    inner_size.replace(SPLIT_LEFT_SEP, "_")
                    .replace(SPLIT_RIGHT_SEP, "_")
                    .replace(ROOT_SEP, "")
                )
                node.constraints.add(f"{inner_size} <= {y}")
                if isinstance(x, str):
                    node.constraints.add(f"{x} <= {y}")
                node.constraints.add(f"{inner_size} + {x} == {y}")
        else:
            inner_size = z
            x = cut
            y = current_size
            assert x is not None
            node.splits[axis_name][new_dim_name] = x
            if isinstance(z, int) and isinstance(x, int):
                previous_cut[axis_name] = x + z
                if not isinstance(y, int):
                    node.constraints.add(f"{z + x} <= {y}")
            elif isinstance(x, int) and x == 0:
                previous_cut[axis_name] = z
                if not isinstance(y, int):
                    node.constraints.add(f"{z} <= {y}")
            else:
                new_cut = root[1:] + new_dim_name
                new_cut = new_cut.replace("/", "").replace("[", "_").replace("]", "_")
                previous_cut[axis_name] = new_cut
                if len(last_split) > 0:
                    a, b = last_split[0]
                    node.constraints.add(f"{a} <= {b}")
                last_split.append((new_cut, y))
                node.constraints.add(f"{z} + {x} == {new_cut}")

        # Create a child node for the nested schedule
        child_node = ParameterLoopNestNode(
            root=new_root_name,
            tiles={a: {} for a in self.abstract_dims},
            split_origin=ParameterSplitOrigin(axis=axis_name, start=x, end=y),
        )
        node.add_child(child_node)

        # Recursively interpret the nested schedule into the child node
        self._interpret_spec_into_node(
            spec=item.body,
            node=child_node,
            root=new_root_name,
            head=[axis_name],
            sizes=deepcopy(sizes),
        )

    def _interpret_tile(
        self,
        item: TileDecl,
        node: ParameterLoopNestNode,
        interchange: list[str],
        sizes: dict[str, list[literal]],
    ) -> str:
        """Interpret a tile declaration. Returns the loop name."""
        self._check_axis_existence(item.axis)
        tile_num = len(node.tiles[item.axis])
        loop_name = f"{item.axis}{tile_num}"
        if isinstance(item.size, int) and item.size <= 0:
            raise ScheduleInterpretError(
                f"`{item}`: tile sizes should be strictly positive."
            )
        node.tiles[item.axis][loop_name] = item.size
        if loop_name in sizes:
            list_sizes = sizes[loop_name]
            list_sizes.append(item.size)
        else:
            list_sizes = [item.size]
            sizes[loop_name] = list_sizes
        interchange.append(loop_name)
        if isinstance(item.size, str):
            if not self.abstract_dim_sizes:
                raise ScheduleInterpretError(
                    f"`{item}`: {item.size} is not an integer."
                )
            old_size = list_sizes[-2]
            partial = item.annotations.partial
            full = item.annotations.full
            if partial or (not full and self.partial_tiles):
                node.constraints.add(f"{item.size} <= {old_size}")
            else:
                s = (
                    ", ".join(map(str, list_sizes))
                    if len(list_sizes) > 1
                    else str(list_sizes[0])
                )
                s = f"{item.size} || {{{s}}}"
                node.constraints.add(s)

        return loop_name

    def _interpret_axis(
        self,
        item: AxisDecl,
        interchange: list[str],
    ) -> str:
        """Interpret a direct axis reference. Returns the loop name."""
        axis_name = item.axis
        if axis_name in self.abstract_matrix:
            return axis_name
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
        sizes: dict[str, list[literal]],
        node: ParameterLoopNestNode,
    ) -> None:
        """Apply annotations to a loop in the node."""
        if annotations.unroll_specified:
            unroll_factor = annotations.unroll_factor
            if unroll_factor is None:
                # None means "unroll fully" - use the loop size
                if loop_name not in sizes or not sizes[loop_name]:
                    raise ScheduleInterpretError(
                        f"{loop_name}'s size being unknown, an unroll factor is needed."
                    )
                unroll_factor = sizes[loop_name][-1]
            elif isinstance(unroll_factor, int) and unroll_factor <= 0:
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
            input_matrix, mtype, pad = annotations.pack
            if isinstance(input_matrix, str):
                idx = self.abstract_matrix.index(input_matrix)
                if idx == len(self.abstract_matrix) - 1:
                    node.buffer_at[loop_name] = mtype
                else:
                    node.pack_at[loop_name] = (idx, mtype, pad)
            else:
                node.pack_at[loop_name] = (input_matrix, mtype, pad)

    def _check_splitting_intervals(
        self,
        item: SplitDecl,
        cut: literal | None,
        x: literal,
    ) -> literal | None:
        """Check that split intervals are valid and contiguous."""
        y = item.end

        if cut is None:
            raise ScheduleInterpretError(f"{item}: {item.axis} already covered.")

        if isinstance(x, int) and isinstance(cut, int):
            if x > cut:
                raise ScheduleInterpretError(
                    f"{item}: splitting doesn't fully cover {item.axis} (jumps from {cut} to {x})."
                )
            elif x < cut:
                raise ScheduleInterpretError(
                    f"{item}: the segment begins at {x} but the previous one ends at {cut}."
                )
        else:
            if x != cut:
                raise ScheduleInterpretError(
                    f"{item}: Splitting ends at {cut} and begins at {x}. These need to be the same."
                )
        if y is None:
            return None

        if isinstance(x, int):
            if isinstance(y, int):
                if x >= y:
                    raise ScheduleInterpretError(
                        f"{item}: the ending point should be greater than the starting point."
                    )
                else:
                    return y - x
            if x == 0:
                return y
        return None


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

    abstract_dims: list[str]
    abstract_dim_sizes: dict[str, int] = field(default_factory=dict)
    abstract_matrix: list[str] = field(default_factory=list)

    def apply(
        self,
        node_name: str,
        spec: dict[str, dict[str, Any]] | str,
        scheduler: Scheduler,
        sample: dict[str, int] = {},
    ) -> None:
        """Parse, interpret, validate, and apply a schedule specification.

        Args:
            node_name: The name of the root node to schedule.
            spec: The schedule specification as a nested dict.

        Raises:
            ScheduleParseError: If the spec cannot be parsed.
            ScheduleInterpretError: If the spec cannot be interpreted.
            ScheduleValidationError: If the resulting schedule is invalid.
        """
        # Parse a YAML specification into a dict
        if isinstance(spec, str):
            yaml_parser = YAMLParser()
            spec = yaml_parser.parse(spec)

        # Parse the specification into an AST
        parser = ScheduleParser()
        ast = parser.parse(spec)

        # Interpret the AST into a LoopNest
        interpreter = ScheduleInterpreter(
            abstract_dims=self.abstract_dims,
            abstract_dim_sizes=self.abstract_dim_sizes,
            abstract_matrix=self.abstract_matrix,
        )
        loop_nest = interpreter.interpret(ast, root=node_name)

        # Apply the sample
        loop_nest = loop_nest.apply_sample(sample)

        # Validate the loop nest
        loop_nest.check()
        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest, scheduler)

    def _apply_loop_nest(self, loop_nest: LoopNest, scheduler: Scheduler) -> None:
        """Apply a LoopNest to the scheduler."""
        scheduler.set_dims(self.abstract_dims)

        if loop_nest.root_node is not None:
            self._apply_node(loop_nest.root_node, scheduler)

    def _apply_node(self, node: LoopNestNode, scheduler: Scheduler) -> None:
        """Recursively apply a LoopNestNode and its children to the scheduler."""
        root = node.root

        for d, s in node.splits.items():
            scheduler.split(d, s, root=root)

        for d, s in node.tiles.items():
            scheduler.tile(d, s, root=root)

        scheduler.interchange(node.interchange, root=root)
        scheduler.vectorize(node.vectorize, root=root)
        scheduler.parallelize(node.parallelize, root=root)
        scheduler.unroll(node.unroll, root=root)

        for axis, mtype in node.buffer_at.items():
            scheduler.buffer_at(axis, mtype=mtype, root=root)

        for axis, (input_idx, mtype, pad) in node.pack_at.items():
            scheduler.pack_at(axis, input_idx, mtype=mtype, pad=pad, root=root)

        # Recursively apply children
        for child in node.children:
            self._apply_node(child, scheduler)
