#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from math import factorial
from typing import Any
from dataclasses import dataclass, field
from copy import deepcopy

from xtc.itf.schd.scheduler import Scheduler

from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.parameter_loop_nest import (
    ParameterLoopNest,
    ParameterLoopNestNode,
    ParameterSplitOrigin,
)

from .exceptions import ScheduleInterpretError, ScheduleParseError
from .parsing import (
    ScheduleParser,
    ScheduleSpec,
    SplitDecl,
    TileDecl,
    AxisDecl,
    PRTDecl,
    Annotations,
    YAMLParser,
    literal,
    ansor_tile,
    tup_list,
    pre_parse,
)


_NODE_SEP = "/"
_SPLIT_LEFT = "["
_SPLIT_RIGHT = "]"


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_dims: list[str],
    spec: dict[str, dict[str, Any]] | str,
    abstract_dim_sizes: dict[str, int] = {},
    abstract_matrix: list[str] = [],
    sample: dict[str, int] = {},
) -> None:
    """Apply a schedule specification to a scheduler.

    This is the main entry point for using the descript scheduling DSL.

    Args:
        scheduler: The scheduler to apply the schedule to.
        node_name: The name of the root node to schedule.
        abstract_dims: The list of abstract axis names (e.g., ["m", "n", "k"]).
        spec: The schedule specification as a nested dict.
    """
    descript = Descript(
        abstract_dims=abstract_dims,
        abstract_dim_sizes=abstract_dim_sizes,
        abstract_matrix=abstract_matrix,
    )
    descript.apply(node_name=node_name, spec=spec, scheduler=scheduler, sample=sample)


@dataclass
class ScheduleInterpreter:
    """Interprets a parsed ScheduleSpec AST into a LoopNest."""

    abstract_dims: list[str]
    abstract_dim_sizes: dict[str, int] = field(default_factory=dict)
    abstract_p_dims: list[str] = field(default_factory=list)
    abstract_r_dims: list[str] = field(default_factory=list)
    abstract_matrix: list[str] = field(default_factory=list)
    partial_tiles: bool = False
    partial_unrolls: bool = False
    _abstract_dim_fresh: dict[str, int] = field(default_factory=dict)
    _levels: dict[str, dict[str, literal | None]] = field(default_factory=dict)

    def interpret(self, spec: ScheduleSpec, root: str) -> ParameterLoopNest:
        """Interpret a schedule specification into a LoopNest."""
        loop_nest = ParameterLoopNest(abstract_dims=self.abstract_dims)
        root_node = loop_nest.build_root_node(root)
        self._interpret_spec_into_node(spec, root_node, root, head=[])
        levels = {
            k: {k_: v_ if v_ is not None else 1 for k_, v_ in level.items()}
            for k, level in self._levels.items()
        }
        root_node._levels = levels
        return loop_nest

    def _interpret_spec_into_node(
        self,
        spec: ScheduleSpec,
        node: ParameterLoopNestNode,
        root: str,
        head: list[str],
        axes: dict[str, list[literal]] | None = None,
    ) -> None:
        """Interpret a schedule spec into an existing node (mutates node)."""
        # Track previous split for each axis
        previous_cut: dict[str, literal | None] = {a: 0 for a in self.abstract_dims}
        # Track finak split and corresponding missing size
        last_split: list[tuple[literal, literal | None]] = []
        # Name of the last loop nest (declared here for typing/liveness)
        loop_name: str = ""

        node.interchange = list(head)

        if not axes:
            axes = dict()
        if self.abstract_dim_sizes:
            for a, v in self.abstract_dim_sizes.items():
                if a not in axes:
                    axes[a] = [v]
        if not axes:
            axes = {a: [] for a in self.abstract_dims}

        for item in spec.items:
            if isinstance(item, SplitDecl):
                self._interpret_split(
                    item=item,
                    node=node,
                    root=root,
                    previous_cut=previous_cut,
                    axes=axes,
                    last_split=last_split,
                )
            elif isinstance(item, TileDecl):
                loop_name = self._interpret_tile(
                    item=item,
                    node=node,
                    axes=axes,
                )
                self._apply_annotations(item, loop_name, axes, node)
            elif isinstance(item, AxisDecl):
                _loop_name = self._interpret_axis(item, node)
                if _loop_name:
                    loop_name = _loop_name
                self._apply_annotations(item, loop_name, axes, node)
            elif isinstance(item, PRTDecl):
                loop_name = self._interpret_prt(
                    item=item,
                    loop_name=loop_name,
                    node=node,
                    axes=axes,
                )

        # Check interchange groups only use each axis at most once
        # And create the sampling constraint
        for k, v_d in node.interchange_groups.items():
            read: set[str] = set()
            for k_, v_d_ in v_d.items():
                for a in self.abstract_dims:
                    if a in v_d_:
                        if a in read:
                            raise ScheduleInterpretError(
                                f"Axis {a} is used twice in interchange {k}."
                            )
                        read.add(a)
            node.constraints.append(f"1 <= {k} <= {factorial(len(v_d))}")

        # Replace the placeholder of the last split with its size
        if len(last_split) > 0:
            a0, b0 = last_split[0]
            if isinstance(a0, int) and not isinstance(b0, int):
                a, b = str(b0), str(a0)
            else:
                a, b = str(a0), str(b0)
            node.constraints = [c.replace(a, b) for c in node.constraints]

        # Check that all splits are complete
        for axis, cut in previous_cut.items():
            if cut is not None and isinstance(cut, int) and cut not in [0] + axes[axis]:
                raise ScheduleInterpretError(
                    f"Splitting of {axis} unachieved (stops at {cut})."
                )

    def _interpret_split(
        self,
        item: SplitDecl,
        node: ParameterLoopNestNode,
        root: str,
        previous_cut: dict[str, literal | None],
        axes: dict[str, list[literal]],
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

        current_size = axes[axis_name][-1] if axes[axis_name] else None

        # When x (the starting point of the split) is not specified,
        # it is the previous cut
        if x is None:
            x = cut
        assert x is not None

        self._check_splitting_intervals(item, cut, x)

        if axis_name not in node.splits:
            node.splits[axis_name] = {}
        new_dim_index = len(node.splits[axis_name])
        new_dim_name = f"{axis_name}{_SPLIT_LEFT}{new_dim_index}{_SPLIT_RIGHT}"
        new_root_name = f"{root}{_NODE_SEP}{new_dim_name}"

        # i[x:y]
        if z is None:
            # Split from x to y

            # Update the previous cut
            previous_cut[axis_name] = y

            # Save the cutting points of the new dimensions
            node.splits[axis_name][new_dim_name] = x
            node.interchange.append(new_dim_name)

            # i[x:] => i[x:end]
            if y is None:
                y = current_size

            # Add constraints if the size of the split, or the size of the tile being split, is unknown
            if isinstance(x, str) or (x != 0 and isinstance(y, str)):
                inner_size = root[1:] + new_dim_name
                inner_size = (
                    inner_size.replace(_SPLIT_LEFT, "_")
                    .replace(_SPLIT_RIGHT, "_")
                    .replace(_NODE_SEP, "")
                )
                node.constraints.append(f"{inner_size} <= {y}")
                if isinstance(x, str):
                    node.constraints.append(f"{x} <= {y}")
                node.constraints.append(f"{inner_size} + {x} == {y}")

        # i[:z:]
        else:
            # Split from x, of size z, that fits in y
            y = current_size

            # Save the cutting points of the new dimensions
            node.splits[axis_name][new_dim_name] = x
            node.interchange.append(new_dim_name)

            # Update the previous cut, and add constraints if needed
            if isinstance(z, int) and isinstance(x, int):
                previous_cut[axis_name] = x + z
                if not isinstance(y, int):
                    node.constraints.append(f"{z + x} <= {y}")
            elif isinstance(x, int) and x == 0:
                previous_cut[axis_name] = z
                if not isinstance(y, int):
                    node.constraints.append(f"{z} <= {y}")
            else:
                new_cut = root[1:] + new_dim_name
                new_cut = (
                    new_cut.replace(_SPLIT_LEFT, "_")
                    .replace(_SPLIT_RIGHT, "_")
                    .replace(_NODE_SEP, "")
                )
                previous_cut[axis_name] = new_cut
                if len(last_split) > 0:
                    a, b = last_split[0]
                    node.constraints.append(f"{a} <= {b}")
                last_split.append((new_cut, y))
                node.constraints.append(f"{z} + {x} == {new_cut}")

        # Create a child node for the nested schedule
        child_node = ParameterLoopNestNode(
            root=new_root_name,
            tiles={a: {} for a in self.abstract_dims},
            split_origin=ParameterSplitOrigin(axis=axis_name, start=x, end=y),
        )
        node.add_child(child_node)

        # TODO: is this the correct footprint?
        if current_size is not None:
            self._update_levels(item.axis, current_size)

        # Recursively interpret the nested schedule into the child node
        self._interpret_spec_into_node(
            spec=item.body,
            node=child_node,
            root=new_root_name,
            head=[axis_name],
            axes=deepcopy(axes),
        )

    def _interpret_tile(
        self,
        item: TileDecl,
        node: ParameterLoopNestNode,
        axes: dict[str, list[literal]],
    ) -> str:
        """Interpret a tile declaration. Returns the loop name."""
        self._check_axis_existence(item.axis)
        if isinstance(item.size, int) and item.size <= 0:
            raise ScheduleInterpretError(
                f"`{item}`: tile sizes should be strictly positive."
            )

        tile_num = len(node.tiles[item.axis])
        loop_name = f"{item.axis}{tile_num}"
        node.tiles[item.axis][loop_name] = item.size

        # If the tile is in an interchange group, update that group
        inter_group = item.annotations.interchange
        if inter_group:
            if inter_group in node.interchange_groups:
                node.interchange_groups[inter_group][len(node.interchange)] = loop_name
            else:
                node.interchange_groups[inter_group] = {
                    len(node.interchange): loop_name
                }

        node.interchange.append(loop_name)

        # Alias for axes[item.axis]
        list_axis: list[int | str]
        if item.axis in axes:
            list_axis = axes[item.axis]
        else:
            list_axis = []
            axes[item.axis] = list_axis

        # Add the contraints
        if isinstance(item.size, str):
            partial = item.annotations.partial
            full = item.annotations.full
            # Partial tile
            if partial or (not full and self.partial_tiles):
                if not list_axis:
                    raise ScheduleInterpretError(
                        f"`{item}`: {item.size} is a partial tile, but its range cannot be computed."
                    )
                old_size = list_axis[-1]
                node.constraints.append(f"{item.size} <= {old_size}")
            # Full tile
            else:
                if not self.abstract_dim_sizes:
                    raise ScheduleInterpretError(
                        f"`{item}` is a full tile, but the axis sizes are unknown."
                    )
                s = ", ".join(map(str, list_axis))
                s = f"{item.size} || {{{s}}}"
                node.constraints.append(s)

        # Update axes
        list_axis.append(item.size)

        self._update_levels(item.axis, axes[item.axis][-1])
        return loop_name

    def _interpret_axis(
        self,
        item: AxisDecl,
        node: ParameterLoopNestNode,
    ) -> str:
        """Interpret a direct axis reference. Returns the loop name."""
        axis_name = item.axis
        interchange = node.interchange

        # AxisDecl is actually a bufferization line
        if axis_name in self.abstract_matrix:
            return ""

        self._check_axis_existence(axis_name)

        # Unreachable when built from a Python dict (because keys
        # can't be duplicated).
        if axis_name in interchange:
            raise ScheduleInterpretError(
                f"Axis {axis_name} is scheduled twice (or more)."
            )

        # If the axis is in an interchange group, update that group
        inter_group = item.annotations.interchange
        if inter_group:
            if inter_group in node.interchange_groups:
                node.interchange_groups[inter_group][len(node.interchange)] = axis_name
            else:
                node.interchange_groups[inter_group] = {
                    len(node.interchange): axis_name
                }

        node.interchange.append(axis_name)

        if self.abstract_dim_sizes:
            self._update_levels(axis_name, self.abstract_dim_sizes[axis_name])
        return axis_name

    def _fresh(self, axis: str):
        if axis not in self._abstract_dim_fresh:
            self._abstract_dim_fresh[axis] = 0
        else:
            self._abstract_dim_fresh[axis] += 1
        return f"prt_{axis}_{self._abstract_dim_fresh[axis]}"

    def _interpret_prt(
        self,
        item: PRTDecl,
        loop_name: str,
        node: ParameterLoopNestNode,
        axes: dict[str, list[literal]],
    ) -> str:
        """Interpret a sequence of PRT tiles. Returns the last loop name.
        The comportement of annotations is derived from BaseStrategyPRTScheme.
        Parallelize is applied on the first tile only.
        Unroll is applied on the last tile only.
        Vectorize is applied on the last axis only.
        Pack/Buffer are applied after the last axis.
        """

        # Collect annotations
        unroll_factor = item.annotations.unroll_factor
        vectorize = item.annotations.vectorize
        parallelize = item.annotations.parallelize
        buffer = item.annotations.buffer
        buffer_specified = item.annotations.buffer_specified
        pack = item.annotations.pack
        pack_specified = item.annotations.pack_specified

        level = item.annotations.level
        interchange = item.annotations.interchange
        if interchange == "interchange":
            interchange = self._fresh("interchange")
        if not interchange and item.shape == "U":
            interchange = self._fresh("interchange_u")

        def _annotations(
            axis: str,
            first_p_axes: bool,
            last_axis: bool,
        ):
            unroll_factor_ = unroll_factor
            if isinstance(unroll_factor_, str):
                unroll_factor_ = self._fresh(unroll_factor_ + "_" + axis)

            return Annotations(
                unroll_factor=unroll_factor_,
                unroll_specified=item.annotations.unroll_specified,
                vectorize=vectorize if last_axis else False,
                parallelize=parallelize if first_p_axes else False,
                buffer=buffer,
                buffer_specified=buffer_specified if last_axis else False,
                pack=pack,
                pack_specified=pack_specified if last_axis else False,
                interchange=interchange,
                partial=item.annotations.partial,
                full=item.annotations.full,
                level=level + "_" + axis if level else "",
            )

        def _interpret(axis: str, annotations: Annotations):
            decl: AxisDecl | TileDecl
            if axis not in node.interchange:
                decl = AxisDecl(axis, annotations)
                loop_name = self._interpret_axis(decl, node)
            else:
                decl = TileDecl(axis, self._fresh(axis), annotations)
                loop_name = self._interpret_tile(decl, node, axes)
            self._apply_annotations(decl, loop_name, axes, node)
            return loop_name

        tile: list[tuple[str, Annotations]] = []
        match item.shape:
            case "P":
                # All P-axes in order
                if not self.abstract_p_dims:
                    raise ScheduleInterpretError(
                        "P scheme used, but P axes are not specified"
                    )
                for axis in self.abstract_p_dims[:-1]:
                    annotations = _annotations(axis, first_p_axes=True, last_axis=False)
                    tile.append((axis, annotations))
                axis = self.abstract_p_dims[-1]
                annotations = _annotations(axis, first_p_axes=True, last_axis=True)
                tile.append((axis, annotations))

            case "R":
                # All R-axes in order
                if not self.abstract_r_dims:
                    raise ScheduleInterpretError(
                        "R scheme used, but R axes are not specified"
                    )
                for axis in self.abstract_r_dims[:-1]:
                    annotations = _annotations(
                        axis, first_p_axes=False, last_axis=False
                    )
                    tile.append((axis, annotations))
                axis = self.abstract_r_dims[-1]
                annotations = _annotations(axis, first_p_axes=False, last_axis=True)
                tile.append((axis, annotations))

            case "T" | "U":
                # All axes in order/a random order
                # Warning, annotations with a randomized order can have unintended comportments for now
                # TODO: Fix those
                first_p_axes = True
                for axis in self.abstract_dims[:-1]:
                    if (
                        first_p_axes
                        and self.abstract_p_dims
                        and axis not in self.abstract_p_dims
                    ):
                        first_p_axes = False
                    annotations = _annotations(
                        axis, first_p_axes=first_p_axes, last_axis=False
                    )
                    tile.append((axis, annotations))
                axis = self.abstract_dims[-1]
                if (
                    first_p_axes
                    and self.abstract_p_dims
                    and axis not in self.abstract_p_dims
                ):
                    first_p_axes = False
                annotations = _annotations(
                    axis, first_p_axes=first_p_axes, last_axis=True
                )
                tile.append((axis, annotations))

            case "O":
                # first P-axis then R-axes, then remaining P-axis
                if not self.abstract_p_dims:
                    raise ScheduleInterpretError(
                        "U scheme used, but P axes are not specified"
                    )
                if not self.abstract_r_dims:
                    raise ScheduleInterpretError(
                        "U scheme used, but R axes are not specified"
                    )
                axis = self.abstract_p_dims[0]
                annotations = _annotations(
                    axis, first_p_axes=True, last_axis=len(self.abstract_dims) > 1
                )
                tile = [(axis, annotations)]
                axes_ = self.abstract_r_dims + self.abstract_p_dims[1:]
                for axis in axes_[:-1]:
                    annotations = _annotations(
                        axis, first_p_axes=False, last_axis=False
                    )
                    tile.append((axis, annotations))
                annotations = _annotations(axis, first_p_axes=False, last_axis=True)
                tile.append((axes_[-1], annotations))

            case "W":
                # Add an output write buffer at this level
                node.buffer_at[loop_name] = None
                return loop_name

            case "F":
                # Fuse producer at this level
                raise ScheduleInterpretError("TODO: Fusion unimplemented")

            case _:
                # Should be unreachable
                raise ScheduleInterpretError(
                    f"PRTDecl with an invalid name: {item.shape}"
                )

        for axis, annotations in tile:
            loop_name = _interpret(axis, annotations)

        return loop_name

    def _check_axis_existence(self, axis: str) -> None:
        """Check that an axis is defined."""
        if axis not in self.abstract_dims:
            raise ScheduleInterpretError(
                f"Axis {axis} is not a defined axis (defined axis: {self.abstract_dims})."
            )

    def _apply_annotations(
        self,
        item: TileDecl | AxisDecl,
        loop_name: str,
        axes: dict[str, list[literal]],
        node: ParameterLoopNestNode,
    ) -> None:
        """Apply annotations to a loop in the node."""
        annotations = item.annotations

        if annotations.unroll_specified:
            unroll_factor = annotations.unroll_factor
            if isinstance(unroll_factor, int) and unroll_factor <= 0:
                raise ScheduleInterpretError(
                    f'`{{"unroll" = {unroll_factor}}}`: unroll parameter should be strictly positive.'
                )
            if unroll_factor is None or isinstance(unroll_factor, str):
                if not axes[item.axis]:
                    raise ScheduleInterpretError(
                        f"{loop_name}'s size being unknown, an unroll factor is needed."
                    )
                axis_size = axes[item.axis][-1]
                if unroll_factor is None:
                    # None means "unroll fully" - use the loop size
                    unroll_factor = axis_size
                else:
                    if self.partial_unrolls:
                        node.constraints.append(f"{unroll_factor} <= {axis_size}")
                    else:
                        node.constraints.append(f"{unroll_factor} || {axis_size}")
            node.unroll[loop_name] = unroll_factor

        if annotations.vectorize:
            if isinstance(annotations.vectorize, str):
                node.vectorize_parameters[loop_name] = annotations.vectorize
                node.constraints.append(f"{annotations.vectorize} in {{0, 1}}")
            else:
                node.vectorize.append(loop_name)

        if annotations.parallelize:
            if isinstance(annotations.parallelize, str):
                node.parallelize_parameters[loop_name] = annotations.parallelize
                node.constraints.append(f"{annotations.parallelize} in {{0, 1}}")
            else:
                node.parallelize.append(loop_name)

        if annotations.level:
            level = annotations.level
            if level in self._levels:
                raise ScheduleInterpretError(f"Level {level} used multiple times.")
            self._levels[level] = {k: None for k in self.abstract_dims}

        if annotations.buffer_specified:
            node.buffer_at[loop_name] = annotations.buffer

        if annotations.pack_specified and annotations.pack is not None:
            input_matrix, mtype, pad = annotations.pack
            if isinstance(input_matrix, str):
                idx = self.abstract_matrix.index(input_matrix)
                # Write buffer
                if idx == len(self.abstract_matrix) - 1:
                    node.buffer_at[loop_name] = mtype
                    if isinstance(annotations.pack_specified, str):
                        node.buffer_parameters[loop_name] = annotations.pack_specified
                        node.constraints.append(
                            f"{annotations.buffer_specified} in {{0, 1}}"
                        )
                # Read buffer
                else:
                    node.pack_at[loop_name] = (idx, mtype, pad)
                    if isinstance(annotations.pack_specified, str):
                        node.pack_parameters[loop_name] = annotations.pack_specified
                        node.constraints.append(
                            f"{annotations.pack_specified} in {{0, 1}}"
                        )
                    if isinstance(pad, str):
                        node.constraints.append(f"{pad} in {{0,1}}")
            else:
                node.pack_at[loop_name] = (input_matrix, mtype, pad)
                if isinstance(annotations.pack_specified, str):
                    node.pack_parameters[loop_name] = annotations.pack_specified
                    node.constraints.append(f"{annotations.pack_specified} in {{0, 1}}")
                    if isinstance(pad, str):
                        node.constraints.append(f"{pad} in {{0,1}}")

    def _update_levels(self, axis: str, size: literal):
        for name, level in self._levels.items():
            if level[axis] is None:
                level[axis] = size

    def _check_splitting_intervals(
        self,
        item: SplitDecl,
        cut: literal | None,
        x: literal,
    ) -> None:
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

        if isinstance(x, int) and isinstance(y, int) and x >= y:
            raise ScheduleInterpretError(
                f"{item}: the ending point should be greater than the starting point."
            )


@dataclass(frozen=True)
class Descript:
    """Applies a parsed and interpreted schedule to a Scheduler.

    This class coordinates the parsing, interpretation, and application
    of schedule specifications. The flow is:
    1. Parse: dict -> ScheduleSpec (AST)
    2. Interpret: ScheduleSpec -> ParameterLoopNest
    2.5. Instantiate: ParameterLoopNest -> LoopNest
    3. Validate: LoopNest.check()
    4. Apply: LoopNest -> Scheduler
    """

    abstract_dims: list[str]
    abstract_dim_sizes: dict[str, int] = field(default_factory=dict)
    abstract_p_dims: list[str] = field(default_factory=list)
    abstract_r_dims: list[str] = field(default_factory=list)
    abstract_matrix: list[str] = field(default_factory=list)
    partial_tiles: bool = False
    partial_unrolls: bool = False

    def __post_init__(self):
        for m in self.abstract_matrix:
            if m in ansor_tile:
                raise ScheduleParseError(f"Forbidden abstract matrix name: {m}")

    def loop_nest(
        self, node_name: str, spec: dict[str, dict[str, Any]] | tup_list | str
    ):
        if isinstance(spec, str):
            yaml_parser = YAMLParser()
            spec = yaml_parser.parse(spec)
        else:
            spec = pre_parse(spec)
        constraints: Any = [v for k, v in spec if k == "constraints"]
        constraints = constraints[0] if constraints else None
        spec = [(k, v) for k, v in spec if k != "constraints"]
        # Parse the specification into an AST
        parser = ScheduleParser()
        ast = parser.parse(spec)

        # Interpret the AST into a LoopNest
        interpreter = ScheduleInterpreter(
            abstract_dims=self.abstract_dims,
            abstract_dim_sizes=self.abstract_dim_sizes,
            abstract_p_dims=self.abstract_p_dims,
            abstract_r_dims=self.abstract_r_dims,
            abstract_matrix=self.abstract_matrix,
            partial_tiles=self.partial_tiles,
            partial_unrolls=self.partial_unrolls,
        )
        loop_nest = interpreter.interpret(ast, root=node_name)
        root = loop_nest.root_node
        if constraints and root:
            root.constraints += constraints
        return loop_nest

    def apply_sample(
        self,
        loop_nest: ParameterLoopNest | LoopNest,
        sample: dict[str, int],
        scheduler: Scheduler,
    ):
        if isinstance(loop_nest, ParameterLoopNest):
            # Apply the sample
            loop_nest = loop_nest.apply_sample(sample)

        # Validate the loop nest
        loop_nest.check()
        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest, scheduler)

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
            scheduler: The scheduler on which the schedule is applied.

        Raises:
            ScheduleParseError: If the spec cannot be parsed.
            ScheduleInterpretError: If the spec cannot be interpreted.
            ScheduleValidationError: If the resulting schedule is invalid.
        """
        loop_nest = self.loop_nest(node_name, spec)
        self.apply_sample(loop_nest, sample, scheduler)

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
