#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import pytest

from xtc.schedules.exceptions import ScheduleValidationError
from xtc.schedules.loop_nest import (
    LoopNest,
    LoopNestNode,
)
from xtc.schedules.loop_nest_analysis import analyze_loop_nest
from xtc.schedules.loop_nest_builder import LoopNestBuilder
from xtc.schedules.plain_schedule import PlainNodeScheduler


def test_empty_loop_nest_pretty_print() -> None:
    assert LoopNest([]).pretty_print() == ""


def test_loop_nest_node_constructor_initializes_nested_children() -> None:
    supplied_children = [
        LoopNestNode(root="first", interchange=["i"]),
        LoopNestNode(root="second", interchange=["i"]),
    ]
    root = LoopNestNode(
        root="root",
        splits={"i": {"middle": 0, "tail": 8}},
        interchange=["i", "middle", "tail"],
        children=[
            LoopNestNode(
                root="middle",
                splits={"i": {"first": 0, "second": 3}},
                interchange=["i", "first", "second"],
                children=supplied_children,
            ),
            LoopNestNode(root="tail", interchange=["i"]),
        ],
    )
    LoopNest(["i"], root).check()

    middle = root.children[0]
    assert [child.root for child in middle.children] == ["first", "second"]
    assert middle.parent is root
    assert all(child.parent is middle for child in middle.children)
    assert middle.children is not supplied_children


def test_loop_nest_node_constructor_does_not_alias_empty_children_list() -> None:
    supplied_children: list[LoopNestNode] = []
    node = LoopNestNode(root="root", children=supplied_children)

    assert node.children is not supplied_children


def test_constructor_rejects_owned_child_without_partial_attachment() -> None:
    child = LoopNestNode(root="child", interchange=["i"])
    parent = LoopNestNode(
        root="parent",
        splits={"i": {"child": 0, "tail": 4}},
        interchange=["i", "child", "tail"],
        children=[
            child,
            LoopNestNode(root="tail", interchange=["i"]),
        ],
    )
    fresh = LoopNestNode(root="fresh", interchange=["i"])

    with pytest.raises(AssertionError, match="must not already have a parent"):
        LoopNestNode(root="other", children=[fresh, child])

    assert fresh.parent is None
    assert child.parent is parent
    assert parent.children[0] is child
    LoopNest(["i"], parent).check()


def test_loop_nest_node_add_child_works_after_constructor_initialization() -> None:
    initial = LoopNestNode(root="initial", interchange=["i"])
    root = LoopNestNode(
        root="root",
        splits={"i": {"initial": 0, "later": 4}},
        interchange=["i", "initial", "later"],
        children=[initial],
    )
    later = LoopNestNode(root="later", interchange=["i"])

    root.add_child(later)
    LoopNest(["i"], root).check()

    assert root.children == [initial, later]
    assert initial.parent is root
    assert later.parent is root


def _partition(axis: str, starts: dict[str, int]) -> LoopNestNode:
    return LoopNestNode(
        root="C",
        splits={axis: starts},
        interchange=[axis, *starts],
        children=[LoopNestNode(root=name, interchange=[axis]) for name in starts],
    )


def test_detailed_pretty_print_labels_root_and_split_children() -> None:
    root = LoopNestNode(
        root="C",
        interchange=["i", "left", "right"],
        vectorize=["i"],
        splits={"j": {"left": 0, "right": 4}},
        children=[LoopNestNode(root="left"), LoopNestNode(root="C/right")],
    )

    assert LoopNest(["i", "j"], root).pretty_print(detailed=True) == (
        "loop i  // vectorized  <- C\n"
        "  split(j, 0, 4)  <- C/left\n"
        "    ...\n"
        "  split(j, 4, ...)  <- C/right\n"
        "    ..."
    )


def test_detailed_pretty_print_labels_a_body_only_node() -> None:
    assert LoopNestNode(root="root").pretty_print(2, detailed=True) == "  ...  <- root"


def test_pretty_print_distinguishes_missing_child_from_body() -> None:
    root = _partition("i", {"first": 0, "second": 4})
    root.children.pop()

    assert LoopNest(["i"], root).pretty_print() == (
        "loop i\n"
        "  split(i, 0, 4)\n"
        "    loop i\n"
        "      ...\n"
        "  split(i, 4, ...)\n"
        "    <missing child>"
    )


def test_pretty_print_follows_interchange_without_sorting_cuts() -> None:
    root = _partition("i", {"first": 0, "second": 8, "third": 4})
    root.interchange[:] = ["i", "third", "j", "first", "second"]
    for factor, child in enumerate(root.children, start=2):
        child.unroll["i"] = factor

    assert LoopNest(["i", "j"], root).pretty_print() == (
        "loop i\n"
        "  split(i, 4, ...)\n"
        "    loop i  // unroll(4)\n"
        "      ...\n"
        "  loop j\n"
        "    split(i, 0, 8)\n"
        "      loop i  // unroll(2)\n"
        "        ...\n"
        "    split(i, 8, 4)\n"
        "      loop i  // unroll(3)\n"
        "        ..."
    )


@pytest.mark.parametrize("reuse_child", [False, True], ids=["cycle", "reused-child"])
def test_pretty_print_stops_at_revisited_nodes(reuse_child: bool) -> None:
    root = _partition("i", {"first": 0, "second": 4, "tail": 8})
    revisited = root.children[0] if reuse_child else root
    root.children[1] = revisited

    output = LoopNest(["i"], root).pretty_print()

    assert f"    <cycle or reused node {revisited.root!r}>" in output
    assert output.endswith("  split(i, 8, ...)\n    loop i\n      ...")


@pytest.mark.parametrize("in_child", [False, True])
def test_check_rejects_tile_declaration_order_despite_sorted_interchange(
    in_child: bool,
) -> None:
    node = LoopNestNode(
        root="bad",
        tiles={"i": {"small": 4, "large": 16}},
        interchange=["i", "large", "small"],
    )
    root = (
        LoopNestNode(
            root="C",
            splits={"i": {"bad": 0, "tail": 32}},
            interchange=["i", "bad", "tail"],
            children=[node, LoopNestNode(root="tail", interchange=["i"])],
        )
        if in_child
        else node
    )
    with pytest.raises(
        ScheduleValidationError,
        match="Tiles on axis i must have non-increasing sizes in declaration order",
    ):
        LoopNest(["i"], root).check()


def test_check_accepts_nonincreasing_tile_declarations_per_axis() -> None:
    root = LoopNestNode(
        root="C",
        tiles={
            "i": {"outer_i": 16, "same_i": 16, "inner_i": 4},
            "j": {"outer_j": 32, "inner_j": 8},
        },
        interchange=["i", "j", "outer_i", "same_i", "inner_i", "outer_j", "inner_j"],
    )
    LoopNest(["i", "j"], root).check()


def test_builder_preserves_invalid_tile_declaration_order_until_check() -> None:
    scheduler = PlainNodeScheduler("C", "C", ["i"])
    tiles = {"small": 4, "large": 16}
    scheduler.tile("i", tiles)
    scheduler.interchange(["i", "large", "small"])

    nest = LoopNestBuilder.from_plain_node_schedule(scheduler.get_plain_schedule())
    assert nest.root_node is not None
    assert list(nest.root_node.tiles["i"].items()) == list(tiles.items())
    assert nest.root_node.interchange == ["i", "large", "small"]
    with pytest.raises(ScheduleValidationError, match="in declaration order"):
        nest.check()


def test_builder_preserves_invalid_split_declaration_order_until_check() -> None:
    scheduler = PlainNodeScheduler("C", "C", ["i"])
    starts = {"first": 0, "middle": 8, "last": 4}
    scheduler.split("i", starts)
    scheduler.interchange(["i", *starts])
    for branch in starts:
        scheduler.interchange(["i"], root=f"./{branch}")

    nest = LoopNestBuilder.from_plain_node_schedule(scheduler.get_plain_schedule())
    assert nest.root_node is not None
    assert list(nest.root_node.splits["i"].items()) == list(starts.items())
    assert [child.root for child in nest.root_node.children] == list(starts)
    with pytest.raises(ScheduleValidationError, match="strictly increasing cuts"):
        nest.check()


def test_check_rejects_more_than_one_partition_axis_even_if_one_is_empty() -> None:
    root = LoopNestNode(root="C", splits={"i": {"left": 0}, "j": {}})
    with pytest.raises(ScheduleValidationError):
        LoopNest(abstract_dims=["i", "j"], root_node=root).check()


@pytest.mark.parametrize("branches", [{}, {"only": 0}], ids=["empty", "unary"])
def test_partition_needs_two_branches(branches: dict[str, int]) -> None:
    root = _partition("i", branches)
    with pytest.raises(
        ScheduleValidationError, match="must declare at least two branches"
    ):
        LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_three_ordered_partition_branches() -> None:
    root = _partition("i", {"first": 0, "middle": 3, "tail": 8})
    nest = LoopNest(abstract_dims=["i"], root_node=root)
    nest.check()

    analysis = analyze_loop_nest(nest)
    assert analysis.extents == {
        "C/first": {"i": 3},
        "C/middle": {"i": 5},
        "C/tail": {"i": None},
        "C": {"i": None},
    }
    assert analysis.loop_contexts["C/tail"]["i"].nominal_extent is None


@pytest.mark.parametrize(
    "starts",
    (
        {"first": 1, "tail": 5},
        {"first": 0, "tail": 0},
        {"first": 0, "tail": 5, "last": 4},
    ),
)
def test_check_rejects_invalid_partition_cut_order(starts: dict[str, int]) -> None:
    root = LoopNestNode(
        root="C",
        splits={"i": starts},
        interchange=list(starts),
        children=[LoopNestNode(root=name, interchange=["i"]) for name in starts],
    )
    with pytest.raises(ScheduleValidationError):
        LoopNest(abstract_dims=["i"], root_node=root).check()


@pytest.mark.parametrize("cut", [3, 6])
def test_check_rejects_unroll_on_partition_target(cut: int) -> None:
    root = _partition("i", {"left": 0, "right": cut})
    root.tiles["i"] = {"tile": 16}
    root.interchange.insert(1, "tile")
    root.unroll["tile"] = 4

    with pytest.raises(
        ScheduleValidationError, match="Cannot unroll partition target tile at C"
    ):
        LoopNest(["i"], root).check()


@pytest.mark.parametrize("target", ["i", "tile"])
def test_check_rejects_vectorization_on_non_leaf_node(target: str) -> None:
    root = _partition("i", {"left": 0, "right": 6})
    root.tiles["i"] = {"tile": 48}
    root.interchange[:] = ["i", "tile"]
    root.vectorize.append(target)

    with pytest.raises(
        ScheduleValidationError,
        match="Vector annotations are only allowed in leaf nodes",
    ):
        LoopNest(["i"], root).check()


@pytest.mark.parametrize(
    ("order", "vectors", "inner"),
    [(["i", "j", "k"], ["i", "k"], "j"), (["i", "j"], ["i", "j"], "k")],
    ids=["explicit-gap", "implicit-inner-loop"],
)
def test_check_rejects_nonvector_loop_below_vector_loop(
    order: list[str], vectors: list[str], inner: str
) -> None:
    root = _partition("i", {"left": 0, "right": 4})
    left, right = root.children
    left.interchange[:] = order
    left.vectorize[:] = vectors
    right.interchange.append("k")

    with pytest.raises(
        ScheduleValidationError,
        match=f"Inner loop {inner} isn't vectorized but an outer one is",
    ):
        LoopNest(["i", "j", "k"], root).check()


def test_check_rejects_partition_cut_beyond_known_tile_bound() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#8": 8}},
        splits={"i": {"first": 0, "tail": 10}},
        interchange=["i", "i#8", "first", "tail"],
        children=[
            LoopNestNode(
                root="first",
                interchange=["i"],
            ),
            LoopNestNode(
                root="tail",
                interchange=["i"],
            ),
        ],
    )
    with pytest.raises(ScheduleValidationError):
        LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_rejects_declared_partition_child_correspondence_errors() -> None:
    malformed = (
        # Missing child.
        LoopNestNode(
            root="C",
            splits={"i": {"a": 0, "b": 4}},
            children=[LoopNestNode(root="a")],
        ),
        # Extra child.
        LoopNestNode(
            root="C",
            splits={"i": {"a": 0}},
            children=[
                LoopNestNode(root="a"),
                LoopNestNode(root="extra"),
            ],
        ),
        # Child declaration order differs.
        LoopNestNode(
            root="C",
            splits={"i": {"a": 0, "b": 4}},
            children=[
                LoopNestNode(root="b"),
                LoopNestNode(root="a"),
            ],
        ),
        # Duplicate branch roots.
        LoopNestNode(
            root="C",
            splits={"i": {"a": 0, "b": 4}},
            children=[
                LoopNestNode(root="a"),
                LoopNestNode(root="a"),
            ],
        ),
    )
    for root in malformed:
        with pytest.raises(ScheduleValidationError):
            LoopNest(abstract_dims=["i"], root_node=root).check()


@pytest.mark.parametrize(
    ("malformation", "diagnostic"),
    [
        ("parent", "Invalid parent link"),
        ("cycle", "reused or participates in a cycle"),
        ("reuse", "reused or participates in a cycle"),
    ],
)
def test_check_rejects_malformed_topology(malformation: str, diagnostic: str) -> None:
    first = LoopNestNode(root="first")
    second = LoopNestNode(root="second")
    root = LoopNestNode(
        root="C",
        splits={"i": {"first": 0, "second": 4}},
        children=[first, second],
    )
    if malformation == "parent":
        object.__setattr__(first, "parent", None)
    elif malformation == "cycle":
        object.__setattr__(first, "children", [root])
    else:
        object.__setattr__(root, "children", [first, first])

    with pytest.raises(ScheduleValidationError, match=diagnostic):
        LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_rejects_children_without_declared_split() -> None:
    child = LoopNestNode(root="child", interchange=["i"])
    root = LoopNestNode(root="C", children=[child])
    with pytest.raises(ScheduleValidationError, match="declares no partition"):
        LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_nested_partitions_on_different_axes() -> None:
    outer = LoopNestNode(
        root="C",
        interchange=["i"],
        splits={"i": {"i0": 0, "i1": 8}},
        children=[
            LoopNestNode(
                root="i0",
                interchange=["i", "j"],
                splits={"j": {"j0": 0, "j1": 4}},
                children=[
                    LoopNestNode(root="j0", interchange=["j"]),
                    LoopNestNode(root="j1", interchange=["j"]),
                ],
            ),
            LoopNestNode(root="i1", interchange=["i"]),
        ],
    )
    LoopNest(abstract_dims=["i", "j"], root_node=outer).check()


def test_check_accepts_nested_partitions_on_same_axis() -> None:
    inner_children = [
        LoopNestNode(root="inner", interchange=["i"]),
        LoopNestNode(root="tail", interchange=["i"]),
    ]
    root = LoopNestNode(
        root="C",
        interchange=["i"],
        splits={"i": {"outer": 0, "last": 10}},
        children=[
            LoopNestNode(
                root="outer",
                interchange=["i"],
                splits={"i": {"inner": 0, "tail": 4}},
                children=inner_children,
            ),
            LoopNestNode(root="last", interchange=["i"]),
        ],
    )
    analysis = analyze_loop_nest(LoopNest(abstract_dims=["i"], root_node=root))
    assert analysis.extents["C/outer/tail"] == {"i": 6}
    assert analysis.extents["C/last"] == {"i": None}


def test_check_accepts_preexisting_inner_tile_refinement() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#8": 8}},
        interchange=["i", "i#8"],
        splits={"i": {"first": 0, "tail": 3}},
        children=[
            LoopNestNode(
                root="first",
                interchange=["i"],
            ),
            LoopNestNode(
                root="tail",
                interchange=["i"],
            ),
        ],
    )
    LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_branch_local_tile_differences_and_shared_prefix() -> None:
    root = LoopNestNode(
        root="C",
        interchange=["i"],
        splits={"i": {"a": 0, "b": 8}},
        children=[
            LoopNestNode(
                root="a",
                tiles={"j": {"j#4": 4}},
                interchange=["i", "j", "j#4"],
            ),
            LoopNestNode(
                root="b",
                tiles={"j": {"j#4": 4, "j#2": 2}},
                interchange=["i", "j", "j#4", "j#2"],
            ),
        ],
    )
    LoopNest(abstract_dims=["i", "j"], root_node=root).check()


def test_check_rejects_mismatched_shared_tile_requirements() -> None:
    root = LoopNestNode(
        root="C",
        interchange=["j"],
        splits={"i": {"a": 0, "b": 8}},
        children=[
            LoopNestNode(
                root="a",
                tiles={"j": {"j#2": 2}},
                interchange=["i", "j#2"],
            ),
            LoopNestNode(
                root="b",
                tiles={"j": {"j#4": 4}},
                interchange=["i", "j#4"],
            ),
        ],
    )
    with pytest.raises(
        ScheduleValidationError, match="Conflicting tile requirements for shared axis j"
    ):
        LoopNest(abstract_dims=["i", "j"], root_node=root).check()


def test_check_rejects_missing_shared_tile_and_out_of_scope_nested_split() -> None:
    missing = LoopNestNode(
        root="C",
        interchange=["j"],
        splits={"i": {"a": 0, "b": 8}},
        children=[
            LoopNestNode(
                root="a",
                tiles={"j": {"j#2": 2}},
                interchange=["i", "j#2"],
            ),
            LoopNestNode(
                root="b",
                interchange=["i"],
            ),
        ],
    )
    with pytest.raises(
        ScheduleValidationError, match="Conflicting tile requirements for shared axis j"
    ):
        LoopNest(abstract_dims=["i", "j"], root_node=missing).check()

    out_of_scope = LoopNestNode(
        root="C",
        interchange=["j"],
        splits={"i": {"a": 0, "b": 8}},
        children=[
            LoopNestNode(
                root="a",
                interchange=["i"],
                splits={"j": {"j0": 0, "j1": 4}},
                children=[
                    LoopNestNode(root="j0", interchange=["j"]),
                    LoopNestNode(root="j1", interchange=["j"]),
                ],
            ),
            LoopNestNode(
                root="b",
                interchange=["i"],
            ),
        ],
    )
    with pytest.raises(ScheduleValidationError, match="loop is outside that scope"):
        LoopNest(abstract_dims=["i", "j"], root_node=out_of_scope).check()


def test_check_validates_partition_cut_bound_when_split_names_are_omitted() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#8": 8}},
        splits={"i": {"first": 0, "tail": 10}},
        interchange=["i", "i#8"],
        children=[
            LoopNestNode(
                root="first",
                interchange=["i"],
            ),
            LoopNestNode(
                root="tail",
                interchange=["i"],
            ),
        ],
    )
    with pytest.raises(ScheduleValidationError, match="Partition cut .*active extent"):
        LoopNest(abstract_dims=["i"], root_node=root).check()


@pytest.mark.parametrize(
    ("outer", "inner", "factor"),
    [(64, 8, 4), (64, 8, 8), (64, 8, 10), (64, 10, 10), (16, 16, 4)],
)
def test_check_accepts_unroll_factors_above_and_below_trip_count(
    outer: int, inner: int, factor: int
) -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"outer": outer, "inner": inner}},
        interchange=["i", "outer", "inner"],
        unroll={"outer": factor},
    )
    LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_unroll_with_unknown_extent() -> None:
    root = LoopNestNode(root="C", interchange=["i"], unroll={"i": 128})
    LoopNest(["i"], root).check()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("unroll", "vectorization"),
        ("unroll", "parallelization"),
        ("vectorization", "parallelization"),
    ],
)
def test_check_rejects_conflicting_annotations_on_same_loop(
    first: str, second: str
) -> None:
    kinds = {first, second}
    root = LoopNestNode(
        root="C",
        interchange=["i", "j"],
        unroll={"j": 2} if "unroll" in kinds else {},
        vectorize=["j"] if "vectorization" in kinds else [],
        parallelize=["j"] if "parallelization" in kinds else [],
    )

    with pytest.raises(
        ScheduleValidationError,
        match=f"Cannot combine {first} and {second} on loop j at C",
    ):
        LoopNest(["i", "j"], root).check()


def test_check_allows_parallel_annotations_between_other_loops() -> None:
    order = ["i", "j", "k", "l"]
    root = LoopNestNode(
        root="C",
        interchange=order,
        parallelize=["k", "i"],
        unroll={"j": 2},
        vectorize=["l"],
    )
    LoopNest(order, root).check()


def test_check_rejects_undefined_and_out_of_scope_annotations() -> None:
    undefined = LoopNestNode(root="C", interchange=["i"], vectorize=["missing"])
    with pytest.raises(ScheduleValidationError):
        LoopNest(abstract_dims=["i"], root_node=undefined).check()

    # Incoming partition loops are available for branch-local annotations.
    incoming_partition_loop = LoopNestNode(
        root="C",
        interchange=["i"],
        splits={"i": {"a": 0, "b": 4}},
        children=[
            LoopNestNode(
                root="a",
                interchange=["i"],
                vectorize=["i"],
            ),
            LoopNestNode(
                root="b",
                interchange=["i"],
            ),
        ],
    )
    LoopNest(abstract_dims=["i"], root_node=incoming_partition_loop).check()

    # A logical loop inherited from the parent is not in the split branch's
    # local scope, even if the child repeats its name.
    out_of_scope = LoopNestNode(
        root="C",
        interchange=["j"],
        splits={"i": {"a": 0, "b": 4}},
        children=[
            LoopNestNode(
                root="a",
                interchange=["i", "j"],
                vectorize=["j"],
            ),
            LoopNestNode(
                root="b",
                interchange=["i"],
            ),
        ],
    )
    with pytest.raises(ScheduleValidationError, match="outside its logical scope"):
        LoopNest(abstract_dims=["i", "j"], root_node=out_of_scope).check()


def test_check_uses_nested_tail_extent_not_absolute_endpoint() -> None:
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#34": 34}},
        splits={"i": {"prefix": 0, "tail": 12}},
        interchange=["i", "i#34", "prefix", "tail"],
        children=[
            LoopNestNode(root="prefix", interchange=["i"]),
            LoopNestNode(
                root="tail",
                splits={"i": {"inner": 0, "local-tail": 23}},
                interchange=["i", "inner", "local-tail"],
                children=[
                    LoopNestNode(root="inner", interchange=["i"]),
                    LoopNestNode(root="local-tail", interchange=["i"]),
                ],
            ),
        ],
    )
    with pytest.raises(ScheduleValidationError, match="active extent 22"):
        LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_nondividing_generic_tile_refinement() -> None:
    # Generic validation permits this chain; backend primitive restrictions are
    # checked separately by the backend that lowers the schedule.
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#6": 6, "i#4": 4}},
        interchange=["i", "i#6", "i#4"],
    )
    LoopNest(abstract_dims=["i"], root_node=root).check()


def test_check_accepts_qualified_child_roots() -> None:
    root = LoopNestNode(
        root="C",
        interchange=["i", "C/a"],
        splits={"i": {"C/a": 0, "C/other": 4}},
        children=[
            LoopNestNode(
                root="C/a",
                interchange=["i", "C/a/b", "C/a/other"],
                splits={"i": {"C/a/b": 0, "C/a/other": 2}},
                children=[
                    LoopNestNode(root="C/a/b", interchange=["i"]),
                    LoopNestNode(root="C/a/other", interchange=["i"]),
                ],
            ),
            LoopNestNode(root="C/other", interchange=["i"]),
        ],
    )
    LoopNest(abstract_dims=["i"], root_node=root).check()


def _tree_snapshot(root: LoopNestNode) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    pending = [root]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        result.append(
            (
                node,
                node.parent,
                tuple(node.children),
                {axis: dict(tiles) for axis, tiles in node.tiles.items()},
                {axis: dict(splits) for axis, splits in node.splits.items()},
                tuple(node.interchange),
                tuple(node.vectorize),
            )
        )
        pending.extend(node.children)
    return tuple(result)


@pytest.mark.parametrize("qualified_collision", [False, True])
def test_check_rejects_ambiguous_analysis_paths(qualified_collision: bool) -> None:
    if qualified_collision:
        children = [
            LoopNestNode(root="branch/leaf"),
            LoopNestNode(
                root="branch",
                splits={"i": {"leaf": 0, "tail": 2}},
                children=[
                    LoopNestNode(root="leaf"),
                    LoopNestNode(root="tail", interchange=["i"]),
                ],
            ),
        ]
    else:
        children = [
            LoopNestNode(
                root="branch",
                tiles={"j": {"j0": 2}},
            ),
            LoopNestNode(
                root="branch",
                tiles={"j": {"j0": 4}},
            ),
        ]
    root = LoopNestNode(
        root="C",
        interchange=["i", "j", "branch", "branch/leaf"],
        splits={"i": {"branch/leaf": 0, "branch": 4}},
        children=children,
    )
    with pytest.raises(ScheduleValidationError, match="Duplicate loop-nest node path"):
        LoopNest(abstract_dims=["i", "j"], root_node=root).check()


def test_check_does_not_mutate_tree_on_success_or_failure() -> None:
    valid = _partition("i", {"left": 0, "right": 4})
    valid_before = _tree_snapshot(valid)
    LoopNest(abstract_dims=["i"], root_node=valid).check()
    assert _tree_snapshot(valid) == valid_before

    invalid = LoopNestNode(
        root="C",
        splits={"i": {"left": 0}},
        children=[
            LoopNestNode(root="left"),
            LoopNestNode(root="right"),
        ],
    )
    invalid_before = _tree_snapshot(invalid)
    with pytest.raises(ScheduleValidationError):
        LoopNest(abstract_dims=["i"], root_node=invalid).check()
    assert _tree_snapshot(invalid) == invalid_before
