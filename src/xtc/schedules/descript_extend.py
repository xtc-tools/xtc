#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from copy import deepcopy
from dataclasses import dataclass, field
import re
import strictyaml
from typing_extensions import override

from xtc.itf.schd.scheduler import Scheduler

from xtc.schedules.descript import Descript, LoopNest, LoopNestSlice, correct_type


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
    descript.apply(node_name=node_name, spec=spec, scheduler=scheduler, sample=sample)


@dataclass(frozen=True)
class DescriptExtend(Descript):
    abstract_axis_sizes: dict[str, int]
    abstract_matrix: list[str]
    partial_tiles: bool = False
    partial_unrolls: bool = False

    @override
    def apply(
        self,
        node_name: str,
        spec: dict[str, dict] | str,
        scheduler: Scheduler,
        sample: dict[str, Any] = {},
    ):
        if isinstance(spec, str):
            dict_spec = self.parse_yaml(spec)
        else:
            dict_spec = spec
        flat_schedules = self._flatten_schedule(root=node_name, spec=dict_spec, head=[])
        variables = set()
        constraints = set()
        for schedule in flat_schedules.slices:
            if isinstance(schedule, LoopNestSliceExtend):
                variables.update(schedule.variables)
                constraints.update(schedule.constraints)

        flat_schedules = self.apply_sample(flat_schedules, sample)
        self.apply_scheduler(flat_schedules, scheduler)

    def parse_yaml(self, spec: str) -> dict[str, dict]:
        dspec = strictyaml.load(spec).data
        assert isinstance(dspec, dict)
        return self._parse_yaml(dspec)

    def _parse_yaml(self, spec: dict[str, dict]) -> dict[str, dict]:
        out_dict = {}
        for level, d_level in spec.items():
            level_dict = {}
            if not isinstance(d_level, dict):
                continue
            for a, v in d_level.items():
                if a == "explore":
                    assert isinstance(v, str)
                    if v == "":
                        tmp = None
                    else:
                        try:
                            tmp = eval(v)
                        except NameError:
                            tmp = v
                    level_dict["explore_axis_order"] = tmp
                elif a in self.abstract_matrix:
                    assert isinstance(v, str)
                    level_dict[a] = self._split_yaml(v)
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
                        level_dict[a] = self._parse_yaml(d)
                        continue
                    level_dict[a] = {}
                    for axis_arg, arg_val in d.items():
                        level_dict[a][axis_arg] = arg_val
                out_dict[level] = level_dict
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

    def flatten_schedule(self, node_name: str, spec: dict[str, dict] | str):
        if isinstance(spec, str):
            dict_spec = self.parse_yaml(spec)
        else:
            dict_spec = spec
        flat_schedules = self._flatten_schedule(root=node_name, spec=dict_spec, head=[])
        variables = []
        constraints = []
        axes = {}
        orders = {}
        for schedule in flat_schedules.slices:
            if isinstance(schedule, LoopNestSliceExtend):
                variables += schedule.variables
                constraints += schedule.constraints
                for axis, order in schedule.axes.items():
                    axes[f"order_{axis}"] = order
                axis_orders = schedule.axis_orders
                for axis in axis_orders:
                    orders[axis] = schedule.axes[axis]

        variables = list(dict.fromkeys(variables))
        constraints = list(dict.fromkeys(constraints))
        return (flat_schedules, variables, constraints, axes, orders)

    def apply_sample(
        self, flat_schedules: LoopNestExtend, sample: dict[str, Any]
    ) -> LoopNestExtend:
        flat_schedules = deepcopy(flat_schedules)
        flat_schedules.apply_sample(sample)
        return flat_schedules

    def apply_scheduler(self, flat_schedules: LoopNestExtend, scheduler: Scheduler):
        flat_schedules.check()
        for schedule in flat_schedules.slices:
            assert isinstance(schedule, LoopNestSliceExtend)
            root = schedule.root
            interchange = []

            for d, s in schedule.axes.items():
                s = list(s.values())
                for s in s:
                    interchange += s

                p = schedule.packs.get(d, None)
                if p:
                    for _, input, pad in p:
                        scheduler.pack_at(s[-1], input, pad=pad)

                b = schedule.buffers.get(d, None)
                if b:
                    scheduler.buffer_at(s[-1])

            for d, s in schedule.splits.items():
                s = correct_type(s)
                scheduler.split(d, s, root=root)

            for d, s in schedule.tiles.items():
                s = correct_type(s)
                scheduler.tile(d, s, root=root)

            scheduler.interchange(interchange, root=root)
            scheduler.vectorize(schedule.vectorize, root=root)
            scheduler.parallelize(schedule.parallelize, root=root)
            s = correct_type(schedule.unroll)
            scheduler.unroll(s, root=root)

    @override
    def _flatten_schedule(
        self,
        root: str,
        spec: dict[str, dict],
        head: list[str],
        tile_sizes: dict[str, int | str] | None = None,
        sched_sizes: dict[str, list] | None = None,
    ) -> LoopNestExtend:
        recursive_scheds = LoopNestExtend(abstract_dims=self.abstract_axis)
        sched = recursive_scheds.build_slice(root)
        # sched: SchedDict = {
        #     "root": root,
        #     "fusions": {},
        #     "packs": {},
        #     "buffers": {},
        #     "axis_orders": [],
        #     "axes": {},
        #     "splits": {},
        #     "tiles": {a: {} for a in self.abstract_axis},
        #     "interchange": [],
        #     "vectorize": [],
        #     "parallelize": [],
        #     "unroll": {},
        #     "variables": [],
        #     "constraints": [],
        # }
        # State of the schedule
        if tile_sizes:
            axes_sizes: dict[str, int | str] = tile_sizes
        else:
            axes_sizes = {a: v for a, v in self.abstract_axis_sizes.items()}
        if sched_sizes is None:
            sched_sizes = {}
            for a, v in axes_sizes.items():
                sched_sizes[a] = [str(v)]
        sizes: dict[str, int | str | None] = {}
        previous_cut: dict[str, int | str | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = head
        # constraints: list[str] = []
        # variables: list[str] = []
        # Processing the schedule
        for tree_declaration, tree_val in spec.items():
            assert isinstance(tree_val, dict)
            tree_interchange = {}
            tree_packs = []
            tree_fusion = []
            tree_buff = []
            last_split = None
            for declaration, val in tree_val.items():
                if declaration == "fusion":
                    tree_fusion.append(val)
                    continue
                elif declaration == "pack":
                    for val_ in val:
                        if len(val_) != 3:
                            raise Exception(f"Packing {val_} should have 3 parameters.")
                        param, input, pad = val_
                        tree_packs.append((param, input, pad))
                        if isinstance(param, str):
                            sched.variables.add(param)
                            sched.constraints.add(f"{param} in {{0, 1}}")
                        if isinstance(input, str):
                            input = self.abstract_matrix.index(input)
                        if isinstance(pad, str):
                            sched.variables.add(pad)
                            sched.constraints.add(f"{pad} in {{0, 1}}")
                    continue
                elif declaration in "buffer":
                    for val_ in val:
                        if len(val_) != 2:
                            raise Exception(
                                f"Bufferisation {val_} should have 2 parameters."
                            )
                        param, pad = val_
                        tree_buff.append((param, pad))
                        if isinstance(param, str):
                            sched.variables.add(param)
                            sched.constraints.add(f"{param} in {{0, 1}}")
                        if isinstance(pad, str):
                            sched.variables.add(pad)
                            sched.constraints.add(f"{pad} in {{0, 1}}")
                    continue
                elif declaration == "explore_axis_order":
                    sched.axis_orders.append(tree_declaration)
                    continue
                elif declaration in self.abstract_matrix:
                    matrix_index = self.abstract_matrix.index(declaration)
                    param = val.get("bufferize", False)
                    pad = val.get("pad", False)
                    if param is None or param:
                        if matrix_index == len(self.abstract_matrix) - 1:
                            tree_buff.append((param, pad))
                        else:
                            tree_packs.append((param, matrix_index, pad))
                        if isinstance(param, str):
                            sched.variables.add(param)
                            sched.constraints.add(f"{param} in {{0, 1}}")
                        if isinstance(pad, str):
                            sched.variables.add(pad)
                            sched.constraints.add(f"{pad} in {{0, 1}}")
                    continue
                elif ":" in declaration:
                    axis_name, x, y, z = self.parse_split_declaration(declaration)
                    self._check_axis_existence(axis_name)

                    # The only declaration where y (the cut) is None is the
                    # last one, so it cannot be the previous one.
                    cut = previous_cut[axis_name]

                    current_size = axes_sizes[axis_name]
                    # Update the previous cut
                    # Save the cutting points of the new dimensions
                    if axis_name not in sched.splits:
                        sched.splits[axis_name] = {}
                    new_dim_index = len(sched.splits[axis_name])
                    new_dim_name = f"{axis_name}[{new_dim_index}]"
                    new_axes_root_name = f"{root}/{new_dim_name}"
                    if axis_name in tree_interchange:
                        tree_interchange[axis_name].append(new_dim_name)
                    else:
                        tree_interchange[axis_name] = [new_dim_name]

                    if z is None:
                        previous_cut[axis_name] = y
                        # When x (the starting point of the slice), is not
                        # specified, it is the previous cut
                        if x is None:
                            x = cut
                        assert isinstance(x, int | str)
                        sched.splits[axis_name][new_dim_name] = x

                        # assert isinstance(x, int)
                        inner_size = self._extended_check_splitting_intervals(
                            declaration, axis_name, cut, x, y
                        )
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
                                inner_size.replace("/", "")
                                .replace("[", "_")
                                .replace("]", "_")
                            )
                            sched.constraints.add(f"{inner_size} <= {y}")
                            if isinstance(x, str):
                                sched.constraints.add(f"{x} <= {y}")
                            sched.constraints.add(f"{inner_size} + {x} == {y}")
                    else:
                        inner_size = z
                        x = cut
                        y = current_size
                        if isinstance(z, int) and isinstance(x, int):
                            previous_cut[axis_name] = x + z
                            if not isinstance(y, int):
                                sched.constraints.add(f"{z + x} <= {y}")
                        elif isinstance(x, int) and x == 0:
                            previous_cut[axis_name] = z
                            if not isinstance(y, int):
                                sched.constraints.add(f"{z} <= {y}")
                        else:
                            new_cut = root[1:] + new_dim_name
                            new_cut = (
                                new_cut.replace("/", "")
                                .replace("[", "_")
                                .replace("]", "_")
                            )
                            previous_cut[axis_name] = new_cut
                            if last_split is not None:
                                a, b = last_split
                                sched.constraints.add(f"{a} <= {b}")
                            last_split = (new_cut, y)
                            sched.constraints.add(f"{z} + {x} == {new_cut}")

                    axes_sizes[axis_name] = inner_size

                    # Fetch the schedule associated with the new dimension
                    next_schedule = val
                    assert isinstance(next_schedule, dict)
                    inner_scheds = self._flatten_schedule(
                        spec=next_schedule,
                        root=new_axes_root_name,
                        tile_sizes=axes_sizes.copy(),
                        head=[axis_name],
                        sched_sizes=deepcopy(sched_sizes),
                    )
                    axes_sizes[axis_name] = current_size

                    recursive_scheds.slices += inner_scheds.slices
                    continue

                elif "#" in declaration:
                    axis_name, tile_size = declaration.split("#")
                    self._check_axis_existence(axis_name)
                    assert isinstance(tile_size, str)
                    if tile_size.isdecimal():
                        loop_size = int(tile_size)
                    else:
                        loop_size = tile_size
                        sched.variables.add(tile_size)
                    if not loop_size:
                        raise Exception(
                            f"Invalid tile size: '{tile_size}' in {declaration}"
                        )

                    if isinstance(loop_size, str):
                        partial = "partial" in val
                        full = "full" in val
                        if partial and full:
                            raise Exception(
                                f"Tile {declaration} cannot be partial and full"
                            )
                        if partial or (not full and self.partial_tiles):
                            sched.constraints.add(
                                f"{loop_size} <= {axes_sizes[axis_name]}"
                            )
                        else:
                            s = (
                                ", ".join(sched_sizes[axis_name])
                                if len(sched_sizes[axis_name]) > 1
                                else sched_sizes[axis_name][0]
                            )
                            s = f"{loop_size} || {{{s}}}"
                            sched.constraints.add(s)
                    sched_sizes[axis_name].insert(0, str(loop_size))
                    axes_sizes[axis_name] = loop_size
                    tile_num = len(sched.tiles[axis_name])
                    loop_name = f"{axis_name}{tile_num}"
                    sched.tiles[axis_name][loop_name] = loop_size
                    sizes[loop_name] = loop_size
                    if axis_name in tree_interchange:
                        raise Exception(
                            f"axis {axis_name} already is used in level {tree_declaration}."
                        )
                    tree_interchange[axis_name] = [loop_name]
                elif declaration in self.abstract_axis:
                    loop_name = declaration
                    axis_name = loop_name
                    if loop_name in tree_interchange:
                        raise Exception(
                            f"""
                            Axis {declaration} is scheduled twice (or more).
                            """
                        )
                    tree_interchange[loop_name] = [loop_name]
                else:
                    raise Exception(
                        f"""
                        Axis {declaration} is not a defined axis.
                        Known axis are: {self.abstract_axis}")
                        """
                    )

                self.annotate(
                    loop_name=loop_name,
                    sizes=sizes,
                    annotations=val,
                    sched=sched,
                )
            sched.axes[tree_declaration] = tree_interchange
            if len(tree_packs) > 0:
                sched.packs[tree_declaration] = tree_packs
            if len(tree_fusion) > 0:
                sched.fusions[tree_declaration] = tree_fusion
            if len(tree_buff) > 0:
                sched.buffers[tree_declaration] = tree_buff
            for v in tree_interchange.values():
                interchange += v

            if last_split is not None:
                a, b = last_split
                if isinstance(a, int) and not isinstance(b, int):
                    a, b = b, a
                a, b = str(a), str(b)
                for c in sched.constraints:
                    sched.constraints.remove(c)
                    sched.constraints.add(c.replace(a, b))
                last_split = None

        # Check if the last cut of each axis is either 0 or None.
        # None correspond to "until the end of the loop". 0 is the
        # default value, if it has 0 then it means the axis isn't splitted.
        # Any other value means the split is let in a partial state.
        for axis, cut in previous_cut.items():
            if cut is not None and isinstance(cut, int) and cut != 0:
                raise Exception(
                    f"Splitting on axis {axis} should end but stops at {cut}"
                )

        sched.interchange = interchange
        return recursive_scheds

    def _extended_check_splitting_intervals(
        self,
        declaration: str,
        axis_name: str,
        cut: int | str | None,
        x: int | str | None,
        y: int | str | None,
    ) -> int | str | None:
        if cut is None:
            raise Exception(
                f"""
                {declaration} is defined on an already covered axis.
                This might be caused by a missing endpoint: {axis_name}
                """
            )

        assert isinstance(x, int | str)

        if isinstance(cut, int) and isinstance(x, int):
            if x > cut:
                raise Exception(
                    f"""
                    Splitting doesn't cover the whole axis
                    (jumps from {cut} to {x} on axis {axis_name})
                    """
                )
            elif x < cut:
                raise Exception(
                    f"""
                    Splitting are overlapping on axis {axis_name}
                    (covered until {cut} but restart at {x})
                    """
                )
        else:
            if x != cut:
                raise Exception(
                    f"""
                    Splitting should use the same variables between an end and a start
                    ({cut} and {x} on axis {axis_name})
                    """
                )
        assert x == cut
        if y is None:
            return None

        if isinstance(x, int):
            if isinstance(y, int):
                if x >= y:
                    raise Exception(
                        f"""
                        Starting point in the splitting cannot be greater or equal to
                        the ending point in: {declaration}
                        """
                    )
                else:
                    return y - x
            if x == 0:
                return y
        return None

    def annotate(
        self,
        loop_name: str,
        sizes: dict[str, int | str | None],
        annotations: dict[str, Any],
        sched: LoopNestSliceExtend,
    ):
        for instr, param in annotations.items():
            assert isinstance(instr, str)
            match instr:
                case "unroll":
                    if param is None and loop_name in sizes:
                        ufactor = sizes[loop_name]
                    else:
                        ufactor = param
                        if isinstance(param, str):
                            sched.variables.add(param)
                            leq = "<=" if self.partial_unrolls else "||"
                            sched.constraints.add(f"{ufactor} {leq} {sizes[loop_name]}")
                    assert isinstance(ufactor, int | str)
                    sched.unroll[loop_name] = ufactor

                case "vectorize":
                    if isinstance(param, str):
                        sched.variables.add(param)
                        sched.constraints.add(f"{param} in {{0, 1}}")
                        sched.vectorize_bool.add((param, loop_name))
                        continue
                    if param is None:
                        sched.vectorize.append(loop_name)
                        continue
                    raise Exception(
                        "Vectorize should not have a parameter (Feature not implemented)"
                    )

                case "parallelize":
                    if isinstance(param, str):
                        sched.variables.add(param)
                        sched.constraints.add(f"{param} in {{0, 1}}")
                        sched.parallelize_bool.add((param, loop_name))
                        continue
                    if param is None:
                        sched.parallelize.append(loop_name)
                        continue
                    if param is not None:
                        raise Exception(
                            "Parallelize should not have a parameter (Feature not implemented)"
                        )
                case "partial":
                    continue
                case "full":
                    continue
                case _:
                    raise Exception(f"Unknown annotation on {loop_name}: {instr}")

    def parse_split_declaration(
        self,
        declaration: str,
    ) -> Tuple[str, int | str | None, int | str | None, int | str | None]:
        pattern = r"^(.*)\[(?:(-\w+|\w*)?):(?:(-\w+|\w*)?)\]$"
        match = re.match(pattern, declaration)
        if not match:
            pattern = r"^(.*)\[:(\w*):\]$"
            match = re.match(pattern, declaration)
            if not match:
                raise Exception(f"Wrong format {declaration}")
            prefix, z = match.groups()
            z = int(z) if z.isnumeric() else z
            return prefix, None, None, z

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if x_str.isnumeric() else x_str
        y = int(y_str) if y_str.isnumeric() else y_str
        x = x if x else None
        y = y if y else None
        return prefix, x, y, None
