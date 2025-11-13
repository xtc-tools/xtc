#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from copy import deepcopy
from dataclasses import dataclass
import re
import strictyaml
from typing_extensions import override

from xtc.itf.schd.scheduler import Scheduler

from xtc.schedules.descript import Descript, SchedDict


def descript_extend_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    abstract_axis_sizes: dict[str, int],
    spec: dict[str, dict],
    abstract_matrix: list[str] = [],
    sample: dict[str, Any] = {},
):
    descript = DescriptExtend(
        abstract_axis=abstract_axis,
        abstract_axis_sizes=abstract_axis_sizes,
        abstract_matrix=abstract_matrix,
    )
    descript.apply(node_name=node_name, spec=spec, scheduler=scheduler, sample=sample)


@dataclass(frozen=True)
class DescriptExtend(Descript):
    abstract_axis_sizes: dict[str, int]
    abstract_matrix: list[str]

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
        for schedule in flat_schedules:
            variables.update(schedule["variables"])
            constraints.update(schedule["constraints"])

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
        for schedule in flat_schedules:
            variables += schedule["variables"]
            constraints += schedule["constraints"]
            for axis, order in schedule["axes"].items():
                axes[f"order_{axis}"] = order
            axis_orders = schedule["axis_orders"]
            for axis in axis_orders:
                orders[axis] = schedule["axes"][axis]

        for axis in self.abstract_axis:
            all_axis_constraints = []
            for schedule in flat_schedules:
                for sched in schedule["sizes"][axis]:
                    if len(sched) > 1:
                        all_axis_constraints.append(sched)
            axis_constraints = []
            i = 0
            while i < len(all_axis_constraints):
                sched = all_axis_constraints[i]
                if isinstance(sched[0], int):
                    axis_constraints.append(sched)
                    all_axis_constraints.pop(i)
                else:
                    i += 1
            flag_flag = True
            while len(all_axis_constraints) > 0 and flag_flag:
                i = 0
                axis_constraints_acc = []
                flag_flag = False
                while i < len(all_axis_constraints):
                    sched = all_axis_constraints[i]
                    flag = False
                    for constraint in axis_constraints:
                        if sched[0] == constraint[-1]:
                            axis_constraints_acc.append(constraint + sched[1:])
                            flag = True
                    if flag:
                        all_axis_constraints.pop(i)
                        flag_flag = True
                    else:
                        i += 1
                if flag_flag:
                    axis_constraints = axis_constraints_acc

            axis_constraints += all_axis_constraints
            axis_constraints.reverse()
            for constraint in axis_constraints:
                if constraint[0] == 1:
                    for size in constraint[1:]:
                        if isinstance(size, str):
                            constraints.append(f"{size} in {{1}}")
                else:
                    constraint.reverse()
                    constraint_str = ""
                    var_flag = False
                    if isinstance(constraint[0], str):
                        constraint_str = "1 || "
                    for size in constraint[:-1]:
                        var_flag = var_flag or isinstance(size, str)
                        constraint_str += f"{size} || "
                    constraint_str += str(constraint[-1])
                    if var_flag:
                        constraints.insert(0, constraint_str)

        variables = list(dict.fromkeys(variables))
        constraints = list(dict.fromkeys(constraints))
        return (flat_schedules, variables, constraints, axes, orders)

    def apply_sample(
        self, flat_schedules: list[SchedDict], sample: dict[str, Any]
    ) -> list[SchedDict]:
        flat_schedules = deepcopy(flat_schedules)
        for schedule in flat_schedules:
            for k in ["splits", "tiles"]:
                for dim, axes in schedule[k].items():
                    for level, size in axes.items():
                        if isinstance(size, str):
                            schedule[k][dim][level] = sample[size]
            for k in ["vectorize", "parallelize"]:
                for i, axes in enumerate(schedule[k]):
                    if isinstance(axes, Tuple):
                        axes, loop = axes
                        axes = sample.get(axes, False)
                        if axes is None or axes:
                            schedule[k][i] = loop
                        else:
                            schedule[k].pop(i)
            for axis, size in schedule["unroll"].items():
                if isinstance(size, str):
                    val = sample[size]
                    if val is None:
                        for s__ in schedule["tiles"].values():
                            for level, size in s__.items():
                                if axis == level:
                                    val = size
                                    break
                            if val is not None:
                                break
                    schedule["unroll"][axis] = val
            for dim, packs in schedule["packs"].items():
                for i, (flag, input, pad) in enumerate(packs):
                    sample_flag = False
                    if isinstance(flag, str):
                        flag = sample.get(flag, False)
                        sample_flag = True
                    if not flag:
                        schedule["packs"][dim].pop(i)
                        continue
                    if isinstance(input, str):
                        input = sample.get(input, input)
                        sample_flag = True
                    if sample_flag:
                        schedule["packs"][dim][i] = (flag, input, pad)
            for dim, buffs in schedule["buffers"].items():
                for i, (flag, pad) in enumerate(buffs):
                    sample_flag = False
                    if isinstance(flag, str):
                        flag = sample.get(flag, False)
                        sample_flag = True
                    if not flag:
                        schedule["buffers"][dim].pop(i)
                        continue
                    if sample_flag:
                        schedule["buffers"][dim][i] = (flag, pad)
            for dim, axes in schedule["axes"].items():
                d_holder = f"order_{dim}"
                s = sample.get(d_holder, None)
                if s:
                    sch = {}
                    for a in s:
                        sch[a] = axes[a]
                    schedule["axes"][dim] = sch
        return flat_schedules

    def apply_scheduler(self, flat_schedules: list[SchedDict], scheduler: Scheduler):
        self._check_flattened_schedule(flat_schedules)
        for schedule in flat_schedules:
            root = schedule["root"]
            interchange = []

            for d, s in schedule["axes"].items():
                s = list(s.values())
                for s in s:
                    interchange += s

                p = schedule["packs"].get(d, None)
                if p:
                    for _, input, pad in p:
                        scheduler.pack_at(s[-1], input, pad=pad)

                b = schedule["buffers"].get(d, None)
                if b:
                    scheduler.buffer_at(s[-1])

            for d, s in schedule["splits"].items():
                scheduler.split(d, s, root=root)

            for d, s in schedule["tiles"].items():
                scheduler.tile(d, s, root=root)

            scheduler.interchange(interchange, root=root)
            scheduler.vectorize(schedule["vectorize"], root=root)
            scheduler.parallelize(schedule["parallelize"], root=root)
            scheduler.unroll(schedule["unroll"], root=root)

    @override
    def _flatten_schedule(
        self,
        root: str,
        spec: dict[str, dict],
        head: list[str],
        tile_sizes: dict[str, int | str] | None = None,
    ) -> list[SchedDict]:
        recursive_scheds: list[SchedDict] = []
        sched: SchedDict = {
            "root": root,
            "fusions": {},
            "packs": {},
            "buffers": {},
            "axis_orders": [],
            "axes": {},
            "splits": {},
            "sizes": {},
            "tiles": {a: {} for a in self.abstract_axis},
            "interchange": [],
            "vectorize": [],
            "parallelize": [],
            "unroll": {},
            "variables": [],
            "constraints": [],
        }
        # State of the schedule
        if tile_sizes:
            axes_sizes: dict[str, int | str] = tile_sizes
        else:
            axes_sizes = {a: v for a, v in self.abstract_axis_sizes.items()}
        sched_sizes = {}
        for a, v in axes_sizes.items():
            sched["sizes"][a] = []
            sched_sizes[a] = [v]
        sizes: dict[str, int | str | None] = {}
        previous_cut: dict[str, int | str | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = head
        constraints: list[str] = []
        variables: list[str] = []
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
                            variables.append(param)
                            constraints.append(f"{param} in {{0, 1}}")
                        if isinstance(input, str):
                            input = self.abstract_matrix.index(input)
                        if isinstance(pad, str):
                            variables.append(pad)
                            constraints.append(f"{pad} in {{0, 1}}")
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
                            variables.append(param)
                            constraints.append(f"{param} in {{0, 1}}")
                        if isinstance(pad, str):
                            variables.append(pad)
                            constraints.append(f"{pad} in {{0, 1}}")
                    continue
                elif declaration == "explore_axis_order":
                    sched["axis_orders"].append(tree_declaration)
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
                            variables.append(param)
                            constraints.append(f"{param} in {{0, 1}}")
                        if isinstance(pad, str):
                            variables.append(pad)
                            constraints.append(f"{pad} in {{0, 1}}")
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
                    if axis_name not in sched["splits"]:
                        sched["splits"][axis_name] = {}
                    new_dim_index = len(sched["splits"][axis_name])
                    new_dim_name = f"{axis_name}[{new_dim_index}]"
                    new_axes_root_name = f"{root}/{new_dim_name}"
                    if axis_name in tree_interchange:
                        tree_interchange[axis_name].append(new_dim_name)
                    else:
                        tree_interchange[axis_name] = [new_dim_name]

                    if z is None:
                        previous_cut[axis_name] = y
                        sched["splits"][axis_name][new_dim_name] = x
                        # When x (the starting point of the slice), is not
                        # specified, it is the previous cut
                        if x is None:
                            x = cut

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
                            constraints.append(f"{inner_size} <= {y}")
                            if isinstance(x, str):
                                constraints.append(f"{x} <= {y}")
                                # constraints.append(f"1 || {x} || {y}")
                                # sched_sizes[axis_name].append(x)
                            constraints.append(f"{inner_size} + {x} == {y}")
                    else:
                        inner_size = z
                        x = cut
                        y = current_size
                        if isinstance(z, int) and isinstance(x, int):
                            previous_cut[axis_name] = x + z
                            if not isinstance(y, int):
                                constraints.append(f"{z + x} <= {y}")
                        elif isinstance(x, int) and x == 0:
                            previous_cut[axis_name] = z
                            if not isinstance(y, int):
                                constraints.append(f"{z} <= {y}")
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
                                constraints.append(f"{a} <= {b}")
                            last_split = (new_cut, y)
                            constraints.append(f"{z} + {x} == {new_cut}")

                    axes_sizes[axis_name] = inner_size

                    # Fetch the schedule associated with the new dimension
                    next_schedule = val
                    assert isinstance(next_schedule, dict)
                    inner_scheds = self._flatten_schedule(
                        spec=next_schedule,
                        root=new_axes_root_name,
                        tile_sizes=axes_sizes.copy(),
                        head=[axis_name],
                    )
                    axes_sizes[axis_name] = current_size

                    recursive_scheds += inner_scheds
                    continue
                elif "#" in declaration:
                    axis_name, tile_size = declaration.split("#")
                    self._check_axis_existence(axis_name)
                    assert isinstance(tile_size, str)
                    if tile_size.isdecimal():
                        loop_size = int(tile_size)
                    else:
                        loop_size = tile_size
                        variables.append(tile_size)
                    if not loop_size:
                        raise Exception(
                            f"Invalid tile size: '{tile_size}' in {declaration}"
                        )

                    axes_sizes[axis_name] = loop_size
                    sched_sizes[axis_name].append(loop_size)
                    tile_num = len(sched["tiles"][axis_name])
                    loop_name = f"{axis_name}{tile_num}"
                    sched["tiles"][axis_name][loop_name] = loop_size
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
                    axis_name=axis_name,
                    sizes=sizes,
                    annotations=val,
                    sched=sched,
                    sched_sizes=sched_sizes[axis_name],
                )
            sched["axes"][tree_declaration] = tree_interchange
            if len(tree_packs) > 0:
                sched["packs"][tree_declaration] = tree_packs
            if len(tree_fusion) > 0:
                sched["fusions"][tree_declaration] = tree_fusion
            if len(tree_buff) > 0:
                sched["buffers"][tree_declaration] = tree_buff
            for v in tree_interchange.values():
                interchange += v

            if last_split is not None:
                a, b = last_split
                if isinstance(a, int) and not isinstance(b, int):
                    a, b = b, a
                a, b = str(a), str(b)
                for i in range(len(constraints)):
                    c = constraints[i]
                    constraints[i] = c.replace(a, b)
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

        sched["interchange"] = interchange
        sched["variables"] = variables + sched["variables"]
        sched["constraints"] = constraints + sched["constraints"]
        for a in self.abstract_axis:
            flag = True
            for sched_ in sched["sizes"][a]:
                if set(sched_sizes[a]) <= set(sched_):
                    flag = False
                    break
            if flag:
                sched["sizes"][a] = [sched_sizes[a]] + sched["sizes"][a]
        return [sched] + recursive_scheds

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
        axis_name: str,
        sizes: dict[str, int | str | None],
        annotations: dict[str, Any],
        sched: dict[str, Any],
        sched_sizes: list[int | str],
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
                            sched["variables"].append(param)
                            sched["sizes"][axis_name].append(sched_sizes + [ufactor])
                    sched["unroll"][loop_name] = ufactor

                case "vectorize":
                    if isinstance(param, str):
                        sched["variables"].append(param)
                        sched["constraints"].append(f"{param} in {{0, 1}}")
                        sched["vectorize"].append((param, loop_name))
                        continue
                    if param is None:
                        sched["vectorize"].append(loop_name)
                        continue
                    raise Exception(
                        "Vectorize should not have a parameter (Feature not implemented)"
                    )

                case "parallelize":
                    if isinstance(param, str):
                        sched["variables"].append(param)
                        sched["constraints"].append(f"{param} in {{0, 1}}")
                        sched["parallelize"].append((param, loop_name))
                        continue
                    if param is None:
                        sched["parallelize"].append(loop_name)
                        continue
                    if param is not None:
                        raise Exception(
                            "Parallelize should not have a parameter (Feature not implemented)"
                        )

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
