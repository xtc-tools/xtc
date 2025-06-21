#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from dataclasses import dataclass
import re
from xtc.itf.schd.scheduler import Scheduler


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    spec: dict[str, dict],
):
    descript = Descript(scheduler=scheduler, abstract_axis=abstract_axis)
    descript.apply(node_name=node_name, spec=spec)


@dataclass(frozen=True)
class Descript:
    scheduler: Scheduler
    abstract_axis: list[str]

    def apply(self, node_name: str, spec: dict[str, dict]):
        flattened_schedule = self._flatten_schedule(root=node_name, spec=spec)

        for interchange in flattened_schedule["interchanges"]:
            self.scheduler.interchange(interchange)
        for d, s in flattened_schedule["splits"].items():
            self.scheduler.split(d, s)
        for d, s in flattened_schedule["tiles"].items():
            self.scheduler.tile(d, s)
        self.scheduler.vectorize(flattened_schedule["vectorize"])
        self.scheduler.parallelize(flattened_schedule["parallelize"])
        self.scheduler.unroll(flattened_schedule["unroll"])

    def _flatten_schedule(
        self,
        root: str,
        spec: dict[str, dict],
    ) -> dict[str, Any]:
        sched = {
            "splits": {a: {} for a in self.abstract_axis},
            "tiles": {a: {} for a in self.abstract_axis},
            "interchanges": [],
            "vectorize": [],
            "parallelize": [],
            "unroll": {},
        }
        # State of the schedule
        sizes: dict[str, int | None] = {}
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = [root]
        # Processing the schedule
        for declaration, val in spec.items():
            # Splits
            if ":" in declaration:
                axis_name, x, y = parse_split_declaration(declaration)
                # The only declaration where y (the cut) is None is the
                # last one, so it cannot be the previous one.
                assert previous_cut[axis_name] is not None
                # When x (the starting point of the slice), is not
                # specified, it is the previous cut
                if x is None:
                    x = previous_cut[axis_name]
                assert x is not None
                # Update the previous cut
                previous_cut[axis_name] = y
                # Save the cutting points of the new dimensions
                new_dim_index = len(sched["splits"][axis_name])
                new_dim_name = f"{root}/{axis_name}[{new_dim_index}]"
                sched["splits"][axis_name][new_dim_name] = x
                interchange.append(new_dim_name)
                # Fetch the schedule associated with the new dimension
                next_schedule = val
                assert isinstance(next_schedule, dict)
                inner_sched = self._flatten_schedule(
                    spec=next_schedule, root=new_dim_name
                )
                sched = merge_flat_schedules(sched, inner_sched)
                continue

            # Tiles
            if "#" in declaration:
                axis_name, tile_size = declaration.split("#")
                loop_size = int(tile_size)
                tile_num = len(sched["tiles"][axis_name])
                loop_name = f"{root}/{axis_name}{tile_num}"
                sched["tiles"][axis_name][loop_name] = loop_size

            # Initial dimensions
            elif declaration in self.abstract_axis:
                axis_name = declaration
                loop_size = 1
                tile_num = len(sched["tiles"][axis_name])
                loop_name = f"{root}/{axis_name}{tile_num}"
                sched["tiles"][axis_name][loop_name] = loop_size
            else:
                raise Exception(f"Unknown declaration: {declaration}")

            sizes[loop_name] = loop_size
            # Build the interchange
            interchange.append(loop_name)
            # Annotations
            assert isinstance(val, dict)
            for instr, param in val.items():
                assert isinstance(instr, str)
                assert isinstance(param, int | None)
                match instr:
                    case "unroll":
                        ufactor = sizes[loop_name] if param is None else param
                        assert isinstance(ufactor, int)
                        sched["unroll"][loop_name] = ufactor
                    case "vectorize":
                        sched["vectorize"].append(loop_name)
                    case "parallelize":
                        sched["parallelize"].append(loop_name)
                    case _:
                        raise Exception(f"Unknown annotation on {loop_name}: {instr}")

        sched["interchanges"] = [interchange] + sched["interchanges"]
        return sched


def parse_split_declaration(declaration: str) -> Tuple[str, int | None, int | None]:
    pattern = r"^(.*)\[(?:(\d*)?):(?:(\d*)?)\]$"
    match = re.match(pattern, declaration)
    if not match:
        raise ValueError("Wrong format.")
    prefix, x_str, y_str = match.groups()
    x = int(x_str) if x_str else None
    y = int(y_str) if y_str else None
    return prefix, x, y


def merge_flat_schedules(
    sched1: dict[str, Any],
    sched2: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "splits": sched1["splits"],  # tmp
        "tiles": sched1["tiles"],  # tmp
        "interchanges": sched1["interchanges"] + sched2["interchanges"],
        "vectorize": list(set(sched1["vectorize"] + sched2["vectorize"])),
        "parallelize": list(set(sched1["parallelize"] + sched2["parallelize"])),
        "unroll": sched1["unroll"] | sched2["unroll"],
    }
    for d in sched2["splits"]:
        for t in sched2["splits"][d]:
            result["splits"][d][t] = sched2["splits"][d][t]

    for d in sched2["tiles"]:
        for t in sched2["tiles"][d]:
            result["tiles"][d][t] = sched2["tiles"][d][t]
    return result
