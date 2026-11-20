#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Focused generic regressions for the checked loop-nest analysis API."""

from __future__ import annotations

import pytest

from xtc.schedules.exceptions import ScheduleValidationError
from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.loop_nest_analysis import (
    LogicalLoopContext,
    LoopNestValidationError,
    analyze_loop_nest,
    check_loop_nest,
    iter_loop_nodes,
)
from xtc.schedules.parameter_loop_nest import (
    ParameterLoopNest,
    ParameterLoopNestNode,
    ParameterSplitOrigin,
)


def test_split_child_requires_incoming_axis_in_local_interchange() -> None:
    nest = LoopNest(
        ["i", "j"],
        LoopNestNode(
            root="C",
            interchange=["i", "left", "right"],
            splits={"i": {"left": 0, "right": 4}},
            children=[
                LoopNestNode(root="left", interchange=["j"]),
                LoopNestNode(root="right", interchange=["i"]),
            ],
        ),
    )

    message = "Split child C/left must include incoming axis i in its interchange."
    with pytest.raises(LoopNestValidationError, match=message) as caught:
        check_loop_nest(nest)

    assert nest.root_node is not None
    assert caught.value.nest is nest
    assert caught.value.node is nest.root_node.children[0]
    dump = nest.pretty_print(detailed=True)
    assert str(caught.value) == f"Loop nest:\n{dump}\n\n{message}"


def test_loop_nest_validation_error_context_is_optional() -> None:
    node = LoopNestNode(root="C", interchange=["i"])
    assert str(LoopNestValidationError("Invalid.")) == "Invalid."
    assert str(LoopNestValidationError("Invalid.", node=node)) == (
        "Loop node:\nloop i  <- C\n  ...\n\nInvalid."
    )


def test_loop_nest_validation_error_survives_rendering_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_pretty_print(_nest: LoopNest, *, detailed: bool) -> str:
        assert detailed
        raise TypeError("Bad field.")

    monkeypatch.setattr(LoopNest, "pretty_print", broken_pretty_print)
    error = LoopNestValidationError("Validation failed.", nest=LoopNest([]))
    assert str(error) == (
        "Loop nest:\n<pretty-print failed: TypeError>\n\nValidation failed."
    )


def test_split_child_incoming_axis_need_not_be_first_in_interchange() -> None:
    nest = LoopNest(
        ["i", "j"],
        LoopNestNode(
            root="C",
            splits={"i": {"left": 0, "right": 4}},
            children=[
                LoopNestNode(root="left", interchange=["j", "i"]),
                LoopNestNode(root="right", interchange=["i", "j"]),
            ],
        ),
    )

    assert check_loop_nest(nest) is None


def test_check_requires_locally_declared_tile_in_interchange() -> None:
    nest = LoopNest(
        ["i"],
        LoopNestNode(root="C", tiles={"i": {"tile": 8}}, interchange=["i"]),
    )

    with pytest.raises(
        ScheduleValidationError,
        match="Tile tile on axis i must appear in the interchange at C",
    ):
        nest.check()


def test_check_rejects_child_tile_matching_enclosing_loop() -> None:
    nest = LoopNest(
        ["i", "k"],
        LoopNestNode(
            root="C",
            tiles={"i": {"outer": 64, "inner": 8}},
            interchange=["i", "outer", "inner", "k", "left", "right"],
            splits={"k": {"left": 0, "right": 4}},
            children=[
                LoopNestNode(
                    root=branch,
                    tiles={"i": {"i64": 64, "i8": 8}},
                    interchange=["k", "i64", "i8"],
                )
                for branch in ("left", "right")
            ],
        ),
    )

    with pytest.raises(
        ScheduleValidationError,
        match="Tile i64 on axis i is outside its logical scope at C/left",
    ):
        nest.check()


def test_empty_nest_full_analysis() -> None:
    nest = LoopNest([])
    assert check_loop_nest(nest) is None
    assert nest.check() is None
    analysis = analyze_loop_nest(nest)
    assert analysis.node_info == {}
    assert analysis.extents == {}
    assert analysis.tile_entries == {}
    assert analysis.loop_contexts == {}
    assert analysis.shared_prefixes == {}


@pytest.mark.parametrize(
    ("tiles", "interchange", "diagnostic"),
    [
        ({}, ["i", "j", "typo"], "Undefined interchange loop typo at C"),
        (
            {"i": {"tile": 4}},
            ["i", "tile", "tile", "j"],
            "Duplicate interchange loop tile at C",
        ),
        (
            {"i": {"t": 4}, "j": {"t": 8}},
            ["i", "j", "t"],
            "Duplicate loop name t at C",
        ),
        ({"i": {"j": 4}}, ["i", "j"], "Duplicate loop name j at C"),
    ],
)
def test_check_rejects_unknown_or_colliding_local_names(
    tiles: dict[str, dict[str, int]], interchange: list[str], diagnostic: str
) -> None:
    nest = LoopNest(
        ["i", "j"], LoopNestNode(root="C", tiles=tiles, interchange=interchange)
    )
    with pytest.raises(ScheduleValidationError, match=diagnostic):
        nest.check()


def test_shared_prefix_decision_preserves_branch_local_refinements() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#16": 16}},
        interchange=["i", "i#16", "j", "left", "right"],
        splits={"i": {"left": 0, "right": 6}},
        children=[
            LoopNestNode(
                root="left",
                tiles={"j": {"j8": 8, "j4": 4}},
                interchange=["i", "j8", "j4"],
            ),
            LoopNestNode(
                root="right",
                tiles={"j": {"j8": 8, "j2": 2}},
                interchange=["i", "j8", "j2"],
            ),
        ],
    )
    nest = LoopNest(["i", "j"], root)
    analysis = analyze_loop_nest(nest)

    assert [path for path, _ in iter_loop_nodes(nest)] == ["C", "C/left", "C/right"]
    assert analysis.shared_prefixes == {
        "C": {"j": (8,)},
        "C/left": {},
        "C/right": {},
    }
    assert analysis.extents["C/left"] == {"i": 6, "j": 8}
    assert analysis.extents["C/right"] == {"i": 10, "j": 8}
    assert analysis.tile_entries["C/left"] == {"j": [(8, 4)]}
    assert analysis.tile_entries["C/right"]["j"] == [(8, 2)]
    assert analysis.loop_contexts["C/left"]["j8"] == LogicalLoopContext("j", 8, 4)
    assert analysis.loop_contexts["C/right"]["j8"] == LogicalLoopContext("j", 8, 2)


def test_scoped_duplicate_tile_names_keep_branch_local_metadata() -> None:
    root = LoopNestNode(
        root="C",
        splits={"i": {"left": 0, "right": 4}},
        interchange=["left", "right"],
        children=[
            LoopNestNode(
                root="left",
                tiles={"j": {"j0": 4}},
                interchange=["i", "j", "j0"],
            ),
            LoopNestNode(
                root="right",
                tiles={"j": {"j0": 8}},
                interchange=["i", "j", "j0"],
            ),
        ],
    )
    nest = LoopNest(["i", "j"], root)

    analysis = analyze_loop_nest(nest)
    assert analysis.node_info["C/left"].tiles_info["j0"] == ("j", 4)
    assert analysis.node_info["C/right"].tiles_info["j0"] == ("j", 8)
    assert root.pretty_print().splitlines()[0] == "split(i, 0, 4)"


def test_pretty_print_keeps_tile_names_local() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i0": 16}},
        splits={"i": {"left": 0, "right": 6}},
        interchange=["i", "i0", "left", "right"],
        children=[
            LoopNestNode(
                root="left",
                tiles={"i": {"i0": 4}},
                interchange=["i", "i0"],
            ),
            LoopNestNode(
                root="right",
                tiles={"i": {"i0": 4}},
                interchange=["i", "i0"],
            ),
        ],
    )

    assert LoopNest(["i"], root).pretty_print() == (
        "loop i\n"
        "  tile(i, 16)\n"
        "    split(i, 0, 6)\n"
        "      loop i\n"
        "        tile(i, 4)\n"
        "          ...\n"
        "    split(i, 6, ...)\n"
        "      loop i\n"
        "        tile(i, 4)\n"
        "          ..."
    )


@pytest.mark.parametrize(
    ("extent", "stride", "trips"), [(50, 32, 2), (64, 8, 8), (None, 8, None)]
)
def test_checked_context_uses_quotient_trip_count(
    extent: int | None, stride: int, trips: int | None
) -> None:
    tiles = {"inner": stride}
    interchange = ["i"]
    target = "i"
    if extent is not None:
        tiles = {"outer": extent, **tiles}
        interchange.append("outer")
        target = "outer"
    interchange.append("inner")
    root = LoopNestNode(
        root="C", tiles={"i": tiles}, interchange=interchange, unroll={target: 2}
    )
    context = analyze_loop_nest(LoopNest(["i"], root)).loop_contexts["C"][target]

    assert context == LogicalLoopContext("i", extent, stride)
    assert context.trip_count == trips


@pytest.mark.parametrize("parent_extent", [16, None])
def test_sampled_partition_derives_tail_from_parent_range(
    parent_extent: int | None,
) -> None:
    template = ParameterLoopNest(["i"])
    root = template.build_root_node("C")
    root.splits = {"i": {"left": 0, "right": "cut"}}
    root.interchange = ["i", "left", "right"]
    if parent_extent is not None:
        root.tiles = {"i": {"tile": "extent"}}
        root.interchange.insert(1, "tile")
    root.add_child(
        ParameterLoopNestNode(
            root="left",
            tiles={},
            interchange=["i"],
            split_origin=ParameterSplitOrigin("i", 0, "cut"),
        )
    )
    root.add_child(
        ParameterLoopNestNode(
            root="right",
            tiles={},
            interchange=["i"],
            split_origin=ParameterSplitOrigin("i", "cut", "extent"),
        )
    )
    nest = template.apply_sample({"cut": 6, "extent": 16})
    analysis = analyze_loop_nest(nest)
    assert nest.root_node is not None
    assert nest.root_node.splits == {"i": {"left": 0, "right": 6}}
    assert analysis.extents["C/left"]["i"] == 6
    # Symbolic checking's terminal bound is not a concrete range declaration.
    expected_tail = parent_extent - 6 if parent_extent is not None else None
    assert analysis.extents["C/right"]["i"] == expected_tail
    assert analysis.loop_contexts["C/right"]["i"].nominal_extent == expected_tail
