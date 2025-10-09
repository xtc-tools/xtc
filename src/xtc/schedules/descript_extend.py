#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from dataclasses import dataclass
import re
from typing_extensions import override

from xtc.itf.schd.scheduler import Scheduler

from xtc.schedules.descript import Descript, SchedDict


def descript_extend_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    spec: dict[str, dict],
    sample: dict[str, Any] = {},
):
    descript = DescriptExtend(abstract_axis=abstract_axis)
    descript.apply(node_name=node_name, spec=spec, scheduler=scheduler, sample=sample)


@dataclass(frozen=True)
class DescriptExtend(Descript):
    @override
    def apply(
        self,
        node_name: str,
        spec: dict[str, dict],
        scheduler: Scheduler,
        sample: dict[str, Any] = {},
    ):
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec, head=[])
        variables = set()
        constraints = set()
        for schedule in flat_schedules:
            variables.update(schedule["variables"])
            constraints.update(schedule["constraints"])

        flat_schedules = self.apply_sample(flat_schedules, sample)
        self.apply_scheduler(flat_schedules, scheduler)

    def flatten_schedule(self, node_name: str, spec: dict[str, dict]):
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec, head=[])
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
        variables = list(dict.fromkeys(variables))
        constraints = list(dict.fromkeys(constraints))
        return (flat_schedules, variables, constraints, axes, orders)

    def apply_sample(
        self, flat_schedules: list[SchedDict], sample: dict[str, Any]
    ) -> list[SchedDict]:
        flat_schedules = flat_schedules.copy()
        for schedule in flat_schedules:
            for k in ["splits", "tiles"]:
                for d, s in schedule[k].items():
                    for d_, s_ in s.items():
                        if isinstance(s_, str):
                            schedule[k][d][d_] = sample[s_]
            for k in ["vectorize", "parallelize"]:
                for i, s in enumerate(schedule[k]):
                    if isinstance(s, Tuple):
                        s, loop = s
                        s = sample.get(s, False)
                        if s is None or s:
                            schedule[k][i] = loop
                        else:
                            schedule[k].pop(i)
            for d, s in schedule["unroll"].items():
                if isinstance(s, str):
                    val = sample[s]
                    if val is None:
                        for s__ in schedule["tiles"].values():
                            for d_, s_ in s__.items():
                                if d == d_:
                                    val = s_
                                    break
                            if val is not None:
                                break
                    schedule["unroll"][d] = val
            for d, axes in schedule["axes"].items():
                d_holder = f"order_{d}"
                s = sample.get(d_holder, None)
                if s:
                    sch = {}
                    for a in s:
                        sch[a] = axes[a]
                    schedule["axes"][d] = sch
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

            for d, s in schedule["splits"].items():
                scheduler.split(d, s, root=root)

            for d, s in schedule["tiles"].items():
                scheduler.tile(d, s, root=root)

            # print(interchange)
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
            "axis_orders": [],
            "axes": {},
            "splits": {},
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
            axes_sizes = {a: f"[{a}]" for a in self.abstract_axis}
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
            for declaration, val in tree_val.items():
                if declaration == "fusion":
                    # sched["fusions"][tree_declaration] = val
                    tree_fusion.append(val)
                    continue
                elif declaration == "pack":
                    for val_ in val:
                        if len(val_) != 3:
                            raise Exception(f"Packing {val_} should have 3 parameters.")
                        param, input, pack = val_
                        tree_packs.append((param, input, pack))
                        if isinstance(param, str):
                            variables.append(param)
                            constraints.append(f"0 <= {param} <= 1")
                        if isinstance(input, str):
                            raise Exception("Packing input cannot be a variable.")
                        if isinstance(pack, str):
                            variables.append(pack)
                            constraints.append(f"0 <= {pack} <= 1")
                    continue
                elif declaration == "explore_axis_order":
                    sched["axis_orders"].append(tree_declaration)
                    continue
                elif ":" in declaration:
                    axis_name, x, y = self.parse_split_declaration(declaration)
                    self._check_axis_existence(axis_name)

                    # The only declaration where y (the cut) is None is the
                    # last one, so it cannot be the previous one.
                    cut = previous_cut[axis_name]

                    # When x (the starting point of the slice), is not
                    # specified, it is the previous cut
                    if x is None:
                        x = cut

                    # print(declaration, axis_name, cut, x, y)
                    lam, inner_size = self._extended_check_splitting_intervals(
                        declaration, axis_name, cut, x, y
                    )
                    current_size = axes_sizes[axis_name]
                    # Update the previous cut
                    previous_cut[axis_name] = y
                    # Save the cutting points of the new dimensions
                    if axis_name not in sched["splits"]:
                        sched["splits"][axis_name] = {}
                    new_dim_index = len(sched["splits"][axis_name])
                    new_dim_name = f"{axis_name}[{new_dim_index}]"
                    new_axes_root_name = f"{root}/{new_dim_name}"
                    sched["splits"][axis_name][new_dim_name] = x
                    if axis_name in tree_interchange:
                        tree_interchange[axis_name].append(new_dim_name)
                    else:
                        tree_interchange[axis_name] = [new_dim_name]
                    inner_size = inner_size if inner_size else f"{current_size} - {x}"
                    inner_size_holder = f"{axis_name}_{new_dim_index}_"
                    constraints.append(f"{inner_size_holder} == {inner_size}")
                    axes_sizes[axis_name] = inner_size_holder

                    if lam:
                        if isinstance(y, str):
                            variables.append(y)
                        constraints.append(lam)
                        constraints.append(f"1 || {y} || {current_size}")

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
                        constraints.append(
                            f"1 || {tile_size} || {axes_sizes[axis_name]}"
                        )
                    if not loop_size:
                        raise Exception(
                            f"Invalid tile size: '{tile_size}' in {declaration}"
                        )

                    axes_sizes[axis_name] = loop_size
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
            sched["axes"][tree_declaration] = tree_interchange
            if len(tree_packs) > 0:
                sched["packs"][tree_declaration] = tree_packs
            if len(tree_fusion) > 0:
                sched["fusions"][tree_declaration] = tree_fusion
            for v in tree_interchange.values():
                interchange += v

        # Check if the last cut of each axis is either 0 or None.
        # None correspond to "until the end of the loop". 0 is the
        # default value, if it has 0 then it means the axis isn't splitted.
        # Any other value means the split is let in a partial state.
        for axis, cut in previous_cut.items():
            if cut is not None and cut != 0:
                raise Exception(
                    f"Splitting on axis {axis} should end but stops at {cut}"
                )

        sched["interchange"] = interchange
        sched["variables"] = variables + sched["variables"]
        sched["constraints"] = constraints + sched["constraints"]
        return [sched] + recursive_scheds

    def _extended_check_splitting_intervals(
        self,
        declaration: str,
        axis_name: str,
        cut: int | str | None,
        x: int | str | None,
        y: int | str | None,
    ) -> Tuple[str | None, int | str | None]:
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

        if y is None:
            return (None, None)

        constraint = f"{x} < {y}"
        if isinstance(y, int):
            if isinstance(x, int):
                if x >= y:
                    raise Exception(
                        f"""
                        Starting point in the splitting cannot be greater or equal to
                        the ending point in: {declaration}
                        """
                    )
                else:
                    return (None, y - x)
            else:
                return (constraint, f"{y} - {x}")
        if isinstance(x, int) and x == 0:
            return (constraint, f"{y}")
        return (constraint, f"{y} - {x}")

    def annotate(
        self,
        loop_name: str,
        sizes: dict[str, int | str | None],
        annotations: dict[str, Any],
        sched: dict[str, Any],
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
                            sched["constraints"].append(
                                f"1 || {param} || {sizes[loop_name]}"
                            )
                    sched["unroll"][loop_name] = ufactor

                case "vectorize":
                    if isinstance(param, str):
                        sched["variables"].append(param)
                        sched["constraints"].append(f"0 <= {param} <= 1")
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
                        sched["constraints"].append(f"0 <= {param} <= 1")
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
    ) -> Tuple[str, int | str | None, int | str | None]:
        pattern = r"^(.*)\[(?:(-\w+|\w*)?):(?:(-\w+|\w*)?)\]$"
        match = re.match(pattern, declaration)
        if not match:
            raise Exception(f"Wrong format {declaration}")

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if x_str.isnumeric() else x_str
        y = int(y_str) if y_str.isnumeric() else y_str
        x = x if x else None
        y = y if y else None
        return prefix, x, y
