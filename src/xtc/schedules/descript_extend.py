#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from copy import deepcopy
from dataclasses import dataclass, field
import re

import strictyaml
from typing_extensions import override

from xtc.itf.schd.scheduler import Scheduler

from xtc.schedules.descript import (
    Annotations,
    AxisDecl,
    BufferDecl,
    Descript,
    FusionDecl,
    LoopNest,
    LoopNestSlice,
    PackDecl,
    ScheduleInterpretError,
    ScheduleInterpreter,
    ScheduleItem,
    ScheduleParseError,
    ScheduleParser,
    ScheduleSpec,
    SplitDecl,
    TileDecl,
)


@dataclass
class LoopNestSliceExtend(LoopNestSlice):
    axis_orders: list[str] = field(default_factory=list)
    axes: dict[str, dict] = field(default_factory=dict)
    packs: dict[str, list] = field(default_factory=dict)
    buffers: dict[str, list] = field(default_factory=dict)
    fusions: dict[str, list] = field(default_factory=dict)
    variables: set[str] = field(default_factory=set)
    constraints: set[str] = field(default_factory=set)
    vectorize_bool: set[tuple[str, str]] = field(default_factory=set)
    parallelize_bool: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class LoopNestExtend(LoopNest):
    @override
    def build_slice(self, root: str) -> LoopNestSliceExtend:
        slice = LoopNestSliceExtend(
            root=root, tiles={a: {} for a in self.abstract_dims}
        )
        self.slices = [slice] + self.slices
        return slice

    def apply_sample(self, sample: dict[str, Any]):
        for schedule in self.slices:
            for dim, axes in schedule.splits.items():
                for level, size in axes.items():
                    if isinstance(size, str):
                        schedule.splits[dim][level] = sample[size]
            for dim, axes in schedule.tiles.items():
                for level, size in axes.items():
                    if isinstance(size, str):
                        schedule.tiles[dim][level] = sample[size]
            for axis, size in schedule.unroll.items():
                if isinstance(size, str):
                    val = sample[size]
                    if val is None:
                        for s__ in schedule.tiles.values():
                            for level, size in s__.items():
                                if axis == level:
                                    val = size
                                    break
                            if val is not None:
                                break
                    schedule.unroll[axis] = val
            if isinstance(schedule, LoopNestSliceExtend):
                for axis, loop in schedule.vectorize_bool:
                    axis = sample.get(axis, False)
                    if axis is None or axis:
                        schedule.vectorize.append(loop)
                for axis, loop in schedule.parallelize_bool:
                    axis = sample.get(axis, False)
                    if axis is None or axis:
                        schedule.parallelize.append(loop)
                for dim, packs in schedule.packs.items():
                    for i, (flag, input, pad) in enumerate(packs):
                        sample_flag = False
                        if isinstance(flag, str):
                            flag = sample.get(flag, False)
                            sample_flag = True
                        if not flag:
                            schedule.packs[dim].pop(i)
                            continue
                        if isinstance(input, str):
                            input = sample.get(input, input)
                            sample_flag = True
                        if sample_flag:
                            schedule.packs[dim][i] = (flag, input, pad)
                for dim, buffs in schedule.buffers.items():
                    for i, (flag, pad) in enumerate(buffs):
                        sample_flag = False
                        if isinstance(flag, str):
                            flag = sample.get(flag, False)
                            sample_flag = True
                        if not flag:
                            schedule.buffers[dim].pop(i)
                            continue
                        if sample_flag:
                            schedule.buffers[dim][i] = (flag, pad)
                for dim, axes in schedule.axes.items():
                    d_holder = f"order_{dim}"
                    s = sample.get(d_holder, None)
                    if s:
                        sch = {}
                        for a in s:
                            sch[a] = axes[a]
                        schedule.axes[dim] = sch


def descript_extend_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    abstract_axis_sizes: dict[str, int],
    spec: dict[str, dict],
    abstract_matrix: list[str] = [],
    sample: dict[str, Any] = {},
    partial_tiles: bool = False,
    partial_unrolls: bool = False,
):
    descript = DescriptExtend(
        abstract_axis=abstract_axis,
        abstract_axis_sizes=abstract_axis_sizes,
        abstract_matrix=abstract_matrix,
        partial_tiles=partial_tiles,
        partial_unrolls=partial_unrolls,
    )
    descript.apply(node_name=node_name, spec=spec, sample=sample, scheduler=scheduler)


class ScheduleParserExtend(ScheduleParser):
    _SPLIT_PATTERN = re.compile(r"^(.*)\[(-\w+|\w*)?:(-\w+|\w*)?\]$")
    _SPLIT_MIDDLE_PATTERN = re.compile(r"^(.*)\[:(\w*):\]$")

    @override
    def __init__(self, abstract_axis: list[str], abstract_matrix: list[str]):
        self.abstract_matrix = abstract_matrix
        super().__init__(abstract_axis)

    @override
    def _parse_declaration(self, declaration: str, value: Any) -> ScheduleItem:
        if "fusion" == declaration:
            return self._parse_fusion()
        if "pack" == declaration:
            return self._parse_pack(value)
        if "buffer" == declaration:
            return self._parse_buffer(value)
        if declaration in self.abstract_matrix:
            return self._parse_matrix(declaration, value)

        return super()._parse_declaration(declaration, value)

    @override
    def _parse_split(self, declaration: str, value: dict) -> SplitDecl:
        axis_name, start, end, size = self._parse_split_syntax_extend(declaration)

        body = self.parse(value)
        return SplitDecl(axis=axis_name, start=start, end=end, body=body, size=size)

    @override
    def _parse_annotations(self, value: dict[str, Any], context: str) -> Annotations:
        """Parse annotation dict into Annotations object."""

        unroll_factor: int | str | None = None
        unroll_specified = False
        vectorize = False
        parallelize = False
        partial = False
        full = False

        for key, param in value.items():
            if key == "unroll":
                if param is True or param is None:
                    unroll_factor = None
                    unroll_specified = True
                elif param is False:
                    pass
                elif isinstance(param, int) or isinstance(param, str):
                    unroll_factor = param
                    unroll_specified = True
                else:
                    raise ScheduleParseError(
                        f'`{{"unroll" = {param}}}`: unroll parameter should be True, False, None, or an integer.'
                    )
            elif key == "vectorize":
                if param is True or param is None:
                    vectorize = True
                elif param is False:
                    pass
                elif isinstance(param, str):
                    vectorize = param
                else:
                    raise ScheduleParseError(
                        f'`{{"vectorize" = {param}}}`: parameterized vectorization not implemented.'
                    )
            elif key == "parallelize":
                if isinstance(param, str):
                    parallelize = param
                elif param is not None:
                    raise ScheduleParseError(
                        f'`{{"parallelize" = {param}}}`: parameterized parallelization not implemented.'
                    )
                else:
                    parallelize = True
            elif key == "partial":
                if full:
                    raise ScheduleParseError("Tile cannot be full and partial.")
                partial = True
            elif key == "full":
                if partial:
                    raise ScheduleParseError("Tile cannot be partial and full.")
                full = True
            else:
                raise ScheduleParseError(f"Unknown annotation on {context}: {key}")

        return Annotations(
            unroll_factor=unroll_factor,
            unroll_specified=unroll_specified,
            vectorize=vectorize,
            parallelize=parallelize,
            partial=partial,
            full=full,
        )

    def _parse_split_syntax_extend(
        self, declaration: str
    ) -> tuple[str, int | str | None, int | str | None, int | str | None]:
        """Parse the syntax of a split declaration."""
        match = self._SPLIT_PATTERN.match(declaration)
        if not match:
            match = self._SPLIT_MIDDLE_PATTERN.match(declaration)
            if not match:
                raise ScheduleParseError(f"Wrong format {declaration}")
            prefix, z = match.groups()
            z = int(z) if z.isnumeric() else z
            return prefix, None, None, z

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if x_str.isnumeric() else x_str
        y = int(y_str) if y_str.isnumeric() else y_str
        x = x if x else None
        y = y if y else None
        return prefix, x, y, None

    def _parse_fusion(self) -> FusionDecl:
        return FusionDecl()

    def _parse_pack(self, value: Any) -> PackDecl:
        assert len(value) == 3
        param, input, pad = value
        return PackDecl(param, input, pad)

    def _parse_buffer(self, value: Any) -> BufferDecl:
        assert len(value == 2)
        param, pad = value
        return BufferDecl(param, pad)

    def _parse_matrix(self, declaration: str, value: Any) -> PackDecl | BufferDecl:
        param = value.get("bufferize", False)
        if not (param is None or param):
            raise ScheduleParseError(
                f"Declared matrix {declaration} without bufferization."
            )
        pad = value.get("pad", False)
        if declaration == self.abstract_matrix[-1]:
            return BufferDecl(param, pad)
        return PackDecl(param, declaration, pad)


class ScheduleInterpreterExtend(ScheduleInterpreter):
    @override
    def __init__(
        self,
        abstract_axis: list[str],
        abstract_axis_sizes: dict[str, int],
        abstract_matrix: list[str],
        partial_tiles: bool = False,
        partial_unrolls: bool = False,
    ):
        self.abstract_matrix = abstract_matrix
        self.abstract_axis_sizes = abstract_axis_sizes
        self.partial_tiles = partial_tiles
        self.partial_unrolls = partial_unrolls
        super().__init__(abstract_axis)

    @override
    def interpret(self, spec: ScheduleSpec, root: str) -> LoopNestExtend:
        return self._interpret_spec(spec, root, head=[])

    @override
    def _interpret_spec(
        self,
        spec: ScheduleSpec,
        root: str,
        head: list[str],
        tile_sizes: dict[str, list[int | str]] | None = None,
    ) -> LoopNestExtend:
        """Interpret a schedule spec recursively."""
        loop_nest = LoopNestExtend(abstract_dims=self.abstract_axis)
        slice = loop_nest.build_slice(root)

        # Track state during interpretation
        last_split: list[tuple[int | str, int | str]] = []
        previous_cut: dict[str, int | str | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = list(head)
        axes_sizes: dict[str, list[int | str]] = {}
        sizes: dict[str, int | str] = {}
        if tile_sizes:
            axes_sizes = tile_sizes
        else:
            axes_sizes = {a: [v] for a, v in self.abstract_axis_sizes.items()}

        for item in spec.items:
            if isinstance(item, SplitDecl):
                self._interpret_split(
                    item=item,
                    slice=slice,
                    loop_nest=loop_nest,
                    root=root,
                    interchange=interchange,
                    previous_cut=previous_cut,
                    axes_sizes=axes_sizes,
                    last_split=last_split,
                )
            elif isinstance(item, TileDecl):
                loop_name = self._interpret_tile(
                    item=item,
                    slice=slice,
                    interchange=interchange,
                    axes_sizes=axes_sizes,
                    sizes=sizes,
                )
                self._apply_annotations(item.annotations, loop_name, sizes, slice)
            elif isinstance(item, AxisDecl):
                loop_name = self._interpret_axis(item, interchange)
                self._apply_annotations(item.annotations, loop_name, sizes, slice)
            elif isinstance(item, FusionDecl):
                ...
            elif isinstance(item, PackDecl):
                self._interpret_pack(slice, interchange[-1], item)
            elif isinstance(item, BufferDecl):
                self._interpret_buffer(slice, interchange[-1], item)

        if len(last_split) > 0:
            a, b = last_split[0]
            if isinstance(a, int) and not isinstance(b, int):
                a, b = b, a
            a, b = str(a), str(b)
            for c in slice.constraints:
                slice.constraints.remove(c)
                slice.constraints.add(c.replace(a, b))

        # Check that all splits are complete
        for axis, cut in previous_cut.items():
            if (
                cut is not None
                and isinstance(cut, int)
                and cut not in [0, axes_sizes[axis][-1]]
            ):
                raise ScheduleInterpretError(
                    f"Splitting of {axis} unachieved (stops at {cut})."
                )

        slice.interchange = interchange
        return loop_nest

    @override
    def _interpret_split(
        self,
        item: SplitDecl,
        slice: LoopNestSlice,
        loop_nest: LoopNest,
        root: str,
        interchange: list[str],
        previous_cut: dict[str, int | str | None],
        axes_sizes: dict[str, list[int | str]] = {},
        last_split: list[tuple[int | str, int | str]] = [],
    ):
        """Interpret a split declaration."""
        if not isinstance(slice, LoopNestSliceExtend):
            return super()._interpret_split(
                item, slice, loop_nest, root, interchange, previous_cut
            )
        axis_name = item.axis
        self._check_axis_existence(axis_name)
        x = item.start
        y = item.end
        z = item.size

        # The only declaration where y (the cut) is None is the
        # last one, so it cannot be the previous one.
        cut = previous_cut[axis_name]
        current_size = axes_sizes[axis_name][-1]

        # When x (the starting point of the slice) is not specified,
        # it is the previous cut
        if x is None:
            x = cut
        inner_size = self._check_splitting_intervals(item, cut, x)
        assert x is not None

        # Save the cutting points of the new dimensions
        if axis_name not in slice.splits:
            slice.splits[axis_name] = {}
        new_dim_index = len(slice.splits[axis_name])
        new_dim_name = f"{axis_name}[{new_dim_index}]"
        new_root_name = f"{root}/{new_dim_name}"
        interchange.append(new_dim_name)

        if z is None:
            # Update the previous cut
            previous_cut[axis_name] = y
            slice.splits[axis_name][new_dim_name] = x
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
                    inner_size.replace("/", "").replace("[", "_").replace("]", "_")
                )
                slice.constraints.add(f"{inner_size} <= {y}")
                if isinstance(x, str):
                    slice.constraints.add(f"{x} <= {y}")
                slice.constraints.add(f"{inner_size} + {x} == {y}")
        else:
            inner_size = z
            x = cut
            y = current_size
            assert x is not None
            slice.splits[axis_name][new_dim_name] = x
            if isinstance(z, int) and isinstance(x, int):
                previous_cut[axis_name] = x + z
                if not isinstance(y, int):
                    slice.constraints.add(f"{z + x} <= {y}")
            elif isinstance(x, int) and x == 0:
                previous_cut[axis_name] = z
                if not isinstance(y, int):
                    slice.constraints.add(f"{z} <= {y}")
            else:
                new_cut = root[1:] + new_dim_name
                new_cut = new_cut.replace("/", "").replace("[", "_").replace("]", "_")
                previous_cut[axis_name] = new_cut
                if len(last_split) > 0:
                    a, b = last_split[0]
                    slice.constraints.add(f"{a} <= {b}")
                last_split.append((new_cut, y))
                slice.constraints.add(f"{z} + {x} == {new_cut}")

        # Recursively interpret the nested schedule
        inner_nest = self._interpret_spec(
            spec=item.body,
            root=new_root_name,
            head=[axis_name],
            tile_sizes=deepcopy(axes_sizes),
        )
        loop_nest.slices += inner_nest.slices

    @override
    def _interpret_tile(
        self,
        item: TileDecl,
        slice: LoopNestSlice,
        interchange: list[str],
        sizes: dict[str, int | str],
        axes_sizes: dict[str, list[int | str]] = {},
    ) -> str:
        """Interpret a tile declaration. Returns the loop name."""
        self._check_axis_existence(item.axis)
        tile_num = len(slice.tiles[item.axis])
        loop_name = f"{item.axis}{tile_num}"
        if isinstance(item.size, int) and item.size <= 0:
            raise ScheduleInterpretError(
                f"`{item}`: tile sizes should be strictly positive."
            )
        slice.tiles[item.axis][loop_name] = item.size
        size_list = axes_sizes[item.axis]
        sizes[loop_name] = item.size
        assert isinstance(size_list, list)
        old_size = size_list[-1]
        interchange.append(loop_name)
        if isinstance(item.size, str):
            assert isinstance(slice, LoopNestSliceExtend)
            slice.variables.add(item.size)
            partial = item.annotations.partial
            full = item.annotations.full
            if partial or (not full and self.partial_tiles):
                slice.constraints.add(f"{item.size} <= {old_size}")
            else:
                s = (
                    ", ".join(map(str, size_list))
                    if len(size_list) > 1
                    else str(size_list[0])
                )
                s = f"{item.size} || {{{s}}}"
                slice.constraints.add(s)
        size_list.append(item.size)
        return loop_name

    @override
    def _apply_annotations(
        self,
        annotations: Annotations,
        loop_name: str,
        sizes: dict[str, int | str],
        slice: LoopNestSlice,
    ) -> None:
        """Apply annotations to a loop in the slice."""
        assert isinstance(slice, LoopNestSliceExtend)
        if annotations.unroll_specified:
            unroll_factor = annotations.unroll_factor
            if unroll_factor is None:
                # None means "unroll fully" - use the loop size
                if loop_name not in sizes:
                    raise ScheduleInterpretError(
                        f"{loop_name}'s size being unknown, an unroll factor is needed."
                    )
                unroll_factor = sizes[loop_name]
            elif isinstance(unroll_factor, str):
                slice.variables.add(unroll_factor)
                if self.partial_unrolls:
                    slice.constraints.add(f"{unroll_factor} <= {sizes[loop_name]}")
                else:
                    slice.constraints.add(f"{unroll_factor} || {sizes[loop_name]}")
            elif unroll_factor <= 0:
                raise ScheduleInterpretError(
                    f'`{{"unroll" = {unroll_factor}}}`: unroll parameter should be strictly positive.'
                )
            slice.unroll[loop_name] = unroll_factor

        vectorize = annotations.vectorize
        if vectorize:
            slice.vectorize.append(loop_name)
            if isinstance(vectorize, str):
                slice.variables.add(vectorize)
                slice.constraints.add(f"{vectorize} in {{0, 1}}")

        parallelize = annotations.parallelize
        if parallelize:
            slice.parallelize.append(loop_name)
            if isinstance(parallelize, str):
                slice.variables.add(parallelize)
                slice.constraints.add(f"{parallelize} in {{0, 1}}")

    def _interpret_pack(
        self, slice: LoopNestSliceExtend, loop_name: str, item: PackDecl
    ):
        param, input, pad = item.param, item.input, item.pad
        if isinstance(param, str):
            slice.variables.add(param)
            slice.constraints.add(f"{param} in {{0, 1}}")
        if isinstance(pad, str):
            slice.variables.add(pad)
            slice.constraints.add(f"{pad} in {{0, 1}}")
        if loop_name in slice.packs:
            slice.packs[loop_name].append((param, input, pad))
        else:
            slice.packs[loop_name] = [(param, input, pad)]

    def _interpret_buffer(
        self, slice: LoopNestSliceExtend, loop_name: str, item: BufferDecl
    ):
        param, pad = item.param, item.pad
        if isinstance(param, str):
            slice.variables.add(param)
            slice.constraints.add(f"{param} in {{0, 1}}")
        if isinstance(pad, str):
            slice.variables.add(pad)
            slice.constraints.add(f"{pad} in {{0, 1}}")
        slice.buffers[loop_name].append((param, pad))


@dataclass(frozen=False)
class DescriptExtend(Descript):
    abstract_axis_sizes: dict[str, int]
    abstract_matrix: list[str]
    partial_tiles: bool = False
    partial_unrolls: bool = False
    _loop_nest: None | LoopNestExtend = None

    @override
    def apply(
        self,
        node_name: str,
        spec: str | dict[str, dict[str, Any]],
        scheduler: Scheduler,
        sample: dict[str, Any] = {},
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

        if isinstance(spec, str):
            spec = self.parse_yaml(spec)

        # Parse the specification into an AST
        parser = ScheduleParserExtend(self.abstract_axis, self.abstract_matrix)
        ast = parser.parse(spec)

        # Interpret the AST into a LoopNest
        interpreter = ScheduleInterpreterExtend(
            self.abstract_axis, self.abstract_axis_sizes, self.abstract_matrix
        )
        loop_nest = interpreter.interpret(ast, root=node_name)

        if sample != {}:
            loop_nest.apply_sample(sample)

        # Validate the loop nest
        loop_nest.check()
        for slice in loop_nest.slices:
            assert isinstance(slice, LoopNestSliceExtend)

        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest, scheduler)

    def loop_nest(
        self, node_name: str, spec: str | dict[str, dict[str, Any]]
    ) -> LoopNestExtend:
        if self._loop_nest:
            return self._loop_nest

        if isinstance(spec, str):
            spec = self.parse_yaml(spec)

        parser = ScheduleParserExtend(self.abstract_axis, self.abstract_matrix)
        ast = parser.parse(spec)

        # Interpret the AST into a LoopNest
        interpreter = ScheduleInterpreterExtend(
            abstract_axis=self.abstract_axis,
            abstract_axis_sizes=self.abstract_axis_sizes,
            abstract_matrix=self.abstract_matrix,
            partial_tiles=self.partial_tiles,
            partial_unrolls=self.partial_unrolls,
        )
        self._loop_nest = interpreter.interpret(ast, root=node_name)
        return self._loop_nest

    def apply_sample(
        self, loop_nest: LoopNestExtend, scheduler: Scheduler, sample: dict[str, Any]
    ):
        loop_nest = deepcopy(loop_nest)
        if sample != {}:
            loop_nest.apply_sample(sample)

        # Validate the loop nest
        loop_nest.check()

        # Apply the schedule to the scheduler
        self._apply_loop_nest(loop_nest, scheduler)

    def parse_yaml(self, spec: str) -> dict[str, dict]:
        dspec = strictyaml.load(spec).data
        assert isinstance(dspec, dict)
        return self._parse_yaml(dspec)

    def _parse_yaml(self, spec: dict[str, dict]) -> dict[str, dict]:
        out_dict = {}
        for a, v in spec.items():
            if a in self.abstract_matrix:
                assert isinstance(v, str)
                out_dict[a] = self._split_yaml(v)
            else:
                if isinstance(v, str):
                    d = self._split_yaml(v)
                else:
                    assert isinstance(v, dict)
                    d = v
                size = d.get("size", None)
                if size:
                    d.pop("size")
                    a = f"{a}#{size}"
                if ":" in a:
                    out_dict[a] = self._parse_yaml(d)
                    continue
                out_dict[a] = {}
                for axis_arg, arg_val in d.items():
                    out_dict[a][axis_arg] = arg_val
        return out_dict

    def _split_yaml(self, s: str) -> dict[str, Any]:
        d = {}
        for s in s.split():
            if "=" not in s:
                d[s] = None
            else:
                x, y = s.split("=")
                try:
                    tmp = eval(y)
                except (NameError, SyntaxError):
                    tmp = y
                d[x] = tmp
        return d
