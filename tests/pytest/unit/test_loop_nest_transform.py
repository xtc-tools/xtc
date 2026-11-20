#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from collections.abc import Callable

import pytest

from xtc.schedules.exceptions import ScheduleValidationError
from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.loop_nest_analysis import analyze_loop_nest
from xtc.schedules.loop_nest_transform import (
    loop_nest_unroll_as_full,
    loop_nest_vectorize_as_pow2,
)


def _checked(nest: LoopNest) -> LoopNest:
    nest.check()
    return nest


def _snapshot(nest: LoopNest) -> tuple[object, ...]:
    return (
        tuple(nest.abstract_dims),
        tuple(
            (
                node.root,
                node.parent.root if node.parent else None,
                tuple(child.root for child in node.children),
                {axis: dict(values) for axis, values in node.tiles.items()},
                {axis: dict(values) for axis, values in node.splits.items()},
                tuple(node.interchange),
                tuple(node.vectorize),
                tuple(node.parallelize),
                dict(node.unroll),
                dict(node.buffer_at),
                dict(node.pack_at),
                dict(node.fuse_producer_at),
                tuple(node.fuse_consumer_at),
                dict(node.external_at),
            )
            for node in nest.nodes
        ),
    )


def _tile_sizes(node: LoopNestNode, axis: str = "i") -> list[int]:
    return list(node.tiles.get(axis, {}).values())


def _bounded_branch(width: int) -> LoopNest:
    branch = LoopNestNode(
        root="branch",
        interchange=["i"],
        vectorize=["i"],
    )
    tail = LoopNestNode(root="tail", interchange=["i"])
    root = LoopNestNode(
        root="C",
        interchange=["i"],
        splits={"i": {"branch": 0, "tail": width}},
        children=[branch, tail],
    )
    return LoopNest(abstract_dims=["i"], root_node=root)


def _tiled_nest(
    sizes: list[int],
    *,
    vectorize: list[str] | None = None,
    unroll: dict[str, int] | None = None,
) -> LoopNest:
    names = [f"i#{size}" for size in sizes]
    return LoopNest(
        abstract_dims=["i"],
        root_node=LoopNestNode(
            root="C",
            tiles={"i": dict(zip(names, sizes, strict=True))},
            interchange=["i", *names],
            vectorize=list(vectorize or []),
            unroll=dict(unroll or {}),
        ),
    )


@pytest.mark.parametrize(
    ("width", "lane", "groups"),
    [(1, None, None), (8, None, None), (6, 2, 3), (48, 16, 3), (7, 1, 7)],
    ids=["unit", "power-of-two", "six", "forty-eight", "odd"],
)
def test_bounded_vector_width_uses_known_lane_and_group_count(
    width: int, lane: int | None, groups: int | None
) -> None:
    source = _checked(_bounded_branch(width))
    before = _snapshot(source)

    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert result.root_node is not None
    branch = result.root_node.children[0]
    if lane is None:
        assert branch.tiles == {}
        assert branch.vectorize == ["i"]
        assert branch.unroll == {}
    else:
        assert _tile_sizes(branch) == [lane]
        lane_name = branch.vectorize[0]
        assert branch.tiles["i"][lane_name] == lane
        if groups == 1:
            assert branch.unroll == {}
        else:
            assert branch.unroll["i"] == groups
    assert _snapshot(source) == before


@pytest.mark.parametrize(("span", "lanes"), [(6, 2), (48, 16)])
def test_vectorization_explicit_tile_has_exact_lane_and_group_structure(
    span: int, lanes: int
) -> None:
    target = f"i#{span}"
    source = _tiled_nest([span], vectorize=[target])
    before = _snapshot(source)
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert result.root_node is not None
    root = result.root_node
    (lane,) = root.vectorize
    assert root.tiles["i"] == {target: span, lane: lanes}
    assert root.interchange == ["i", target, lane]
    assert root.unroll == {target: span // lanes}
    assert _snapshot(source) == before


@pytest.mark.parametrize(
    ("sizes", "include_base"),
    [([48, 8], False), ([3], True), ([7, 6, 5, 2], False)],
    ids=["dividing", "unknown-outer-extent", "non-dividing"],
)
def test_same_axis_vector_groups_are_unchanged(
    sizes: list[int], include_base: bool
) -> None:
    targets = (["i"] if include_base else []) + [f"i#{size}" for size in sizes]
    source = _tiled_nest(sizes, vectorize=targets)
    before = _snapshot(source)
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert _snapshot(source) == _snapshot(result) == before
    assert _snapshot(loop_nest_vectorize_as_pow2(result)) == before


def test_tail_vectorization_uses_inherited_parent_tile_and_actual_tail_extent() -> None:
    tail = LoopNestNode(
        root="tail",
        interchange=["i"],
        vectorize=["i"],
    )
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#34": 34}},
        interchange=["i", "i#34"],
        splits={"i": {"first": 0, "tail": 12}},
        children=[
            LoopNestNode(root="first", interchange=["i"]),
            tail,
        ],
    )
    result = _checked(
        loop_nest_vectorize_as_pow2(
            _checked(LoopNest(abstract_dims=["i"], root_node=root))
        )
    )
    assert result.root_node is not None
    normalized_tail = result.root_node.children[1]
    lane_name = normalized_tail.vectorize[0]
    assert normalized_tail.tiles["i"][lane_name] == 2
    assert normalized_tail.unroll["i"] == 11


def test_vector_lane_name_collision_gets_fresh_name() -> None:
    source = _bounded_branch(6)
    assert source.root_node is not None
    tail = source.root_node.children[1]
    tail.tiles["i"] = {"__v_i": 1}
    tail.interchange.append("__v_i")

    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert result.root_node is not None
    branch = result.root_node.children[0]
    lane = branch.vectorize[0]
    assert lane == "__v_i_1"
    assert branch.tiles["i"][lane] == 2
    assert branch.unroll["i"] == 3


def test_vectorization_with_unknown_extent_is_unchanged() -> None:
    source = LoopNest(
        ["i"], LoopNestNode(root="C", interchange=["i"], vectorize=["i"])
    )
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert _snapshot(result) == _snapshot(source)


def test_unroll_base_axis_inserts_extent_above_existing_tile() -> None:
    source = _checked(
        LoopNest(
            abstract_dims=["i"],
            root_node=LoopNestNode(
                root="C",
                tiles={"i": {"i#8": 8}},
                interchange=["i", "i#8"],
                unroll={"i": 4},
            ),
        )
    )
    result = _checked(loop_nest_unroll_as_full(source))
    assert result.root_node is not None
    root = result.root_node
    assert root.tiles["i"]["__u_i"] == 32
    assert root.tiles["i"]["i#8"] == 8
    assert root.unroll == {"__u_i": 4}
    assert root.interchange.index("__u_i") < root.interchange.index("i#8")


def test_split_child_bounds_are_resolved_per_branch() -> None:
    first = LoopNestNode(
        root="first",
        interchange=["i"],
        vectorize=["i"],
    )
    second = LoopNestNode(
        root="second",
        interchange=["i"],
        vectorize=["i"],
    )
    root = LoopNestNode(
        root="C",
        tiles={"i": {"i#50": 50}},
        interchange=["i", "i#50"],
        splits={"i": {"first": 0, "second": 6, "tail": 34}},
        children=[
            first,
            second,
            LoopNestNode(root="tail", interchange=["i"]),
        ],
    )
    result = _checked(
        loop_nest_vectorize_as_pow2(
            _checked(LoopNest(abstract_dims=["i"], root_node=root))
        )
    )
    assert result.root_node is not None
    first_result, second_result = result.root_node.children[:2]
    assert first_result.tiles["i"][first_result.vectorize[0]] == 2
    assert first_result.unroll["i"] == 3
    assert second_result.tiles["i"][second_result.vectorize[0]] == 4
    assert second_result.unroll["i"] == 7


@pytest.mark.parametrize(
    ("transform", "source"),
    [
        (loop_nest_vectorize_as_pow2, _bounded_branch(48)),
        (loop_nest_vectorize_as_pow2, _tiled_nest([6], vectorize=["i#6"])),
        (loop_nest_unroll_as_full, _tiled_nest([8], unroll={"i#8": 4})),
        (
            loop_nest_unroll_as_full,
            LoopNest(
                abstract_dims=["i"],
                root_node=LoopNestNode(root="C", interchange=["i"], unroll={"i": 4}),
            ),
        ),
    ],
    ids=["vector-bounded", "vector-explicit", "unroll-tiled", "unroll-base"],
)
def test_each_transform_is_idempotent_for_common_variants(
    transform: Callable[[LoopNest], LoopNest], source: LoopNest
) -> None:
    source = _checked(source)
    result = _checked(transform(source))
    repeated = _checked(transform(result))
    assert _snapshot(repeated) == _snapshot(result)


def test_no_partition_clone_owns_mutable_result_fields() -> None:
    source = _checked(
        LoopNest(
            abstract_dims=["i"],
            root_node=LoopNestNode(
                root="C",
                tiles={"i": {"i#8": 8}},
                interchange=["i", "i#8", "leaf", "tail"],
                parallelize=["i#8"],
                buffer_at={"i#8": None},
                splits={"i": {"leaf": 0, "tail": 4}},
                children=[
                    LoopNestNode(root="leaf", interchange=["i"]),
                    LoopNestNode(root="tail", interchange=["i"]),
                ],
            ),
        )
    )
    before = _snapshot(source)
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert result.root_node is not None
    assert source.root_node is not None
    assert result.root_node is not source.root_node
    assert result.root_node.children[0] is not source.root_node.children[0]
    assert result.root_node.children[0].parent is result.root_node
    result.root_node.interchange.append("result-only")
    result.root_node.tiles["i"]["result-only"] = 3
    result.root_node.parallelize.append("result-only")
    result.root_node.buffer_at["result-only"] = "i"
    result.root_node.children[0].interchange.append("child-only")
    assert _snapshot(source) == before


@pytest.mark.parametrize(
    ("sizes", "target", "factor", "expected"),
    [
        ([8], "i#8", 8, [8]),
        ([50], "i#50", 32, [50, 32]),
        ([64, 8], "i#64", 2, [64, 16, 8]),
        ([64, 8], "i#64", 10, [64, 8]),
        ([64, 10], "i#64", 2, [64, 20, 10]),
        ([64, 10], "i#64", 10, [64, 10]),
    ],
)
def test_full_unroll_uses_trip_counts(
    sizes: list[int], target: str, factor: int, expected: list[int]
) -> None:
    source = _checked(_tiled_nest(sizes, unroll={target: factor}))
    before = _snapshot(source)
    result = _checked(loop_nest_unroll_as_full(source))
    assert result.root_node is not None
    assert _tile_sizes(result.root_node) == expected
    ((unrolled, preserved_factor),) = result.root_node.unroll.items()
    assert preserved_factor == factor
    if sizes == expected:
        assert unrolled == target
        assert _snapshot(result) == before
    else:
        assert unrolled != target
        assert analyze_loop_nest(result).loop_contexts["C"][unrolled].trip_count == factor
    assert _snapshot(source) == before


def test_unroll_moves_placements_to_generated_inner_loop() -> None:
    source = _checked(
        LoopNest(
            ["i", "j"],
            LoopNestNode(
                root="C",
                interchange=["i", "j"],
                unroll={"i": 2},
                buffer_at={"i": "local", "j": None},
                pack_at={"i": (0, "global", True)},
                fuse_producer_at={"i": 1},
                fuse_consumer_at=["i"],
                external_at={"i": "my_kernel"},
            ),
        )
    )
    before = _snapshot(source)
    result = loop_nest_unroll_as_full(source)
    root = result.root_node
    assert root is not None
    ((inner, factor),) = root.unroll.items()
    assert factor == 2
    assert root.interchange == ["i", inner, "j"]
    assert root.buffer_at == {inner: "local", "j": None}
    assert root.pack_at == {inner: (0, "global", True)}
    assert root.fuse_producer_at == {inner: 1}
    assert root.fuse_consumer_at == [inner]
    assert root.external_at == {inner: "my_kernel"}
    assert _snapshot(source) == before
    assert _snapshot(loop_nest_unroll_as_full(result)) == _snapshot(result)


def test_multiple_vector_axes_are_unchanged() -> None:
    source = _checked(
        LoopNest(
            abstract_dims=["i", "j", "k"],
            root_node=LoopNestNode(
                root="C",
                tiles={"i": {"i6": 6}, "j": {"j6": 6}},
                interchange=["i", "j", "i6", "j6", "k"],
                vectorize=["i6", "j6", "k"],
            ),
        )
    )
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert _snapshot(result) == _snapshot(source)


def test_vector_group_does_not_prevent_normalizing_single_target_in_sibling() -> None:
    source = LoopNest(
        ["i", "k"],
        LoopNestNode(
            root="C",
            interchange=["i", "k"],
            splits={"k": {"left": 0, "right": 6}},
            children=[
                LoopNestNode(
                    root=name,
                    interchange=["k", "i8", "i6"],
                    tiles={"i": {"i8": 8, "i6": 6}},
                    vectorize=vectors,
                )
                for name, vectors in [("left", ["i8", "i6"]), ("right", ["i6"])]
            ],
        ),
    )
    before = _snapshot(source)
    result = _checked(loop_nest_vectorize_as_pow2(source))
    assert result.root_node is not None
    left, right = result.root_node.children
    assert left.vectorize == ["i8", "i6"]
    assert _tile_sizes(left) == [8, 6]
    assert _tile_sizes(right) == [8, 6, 2]
    assert right.unroll == {"i6": 3}
    assert analyze_loop_nest(result).extents == analyze_loop_nest(source).extents
    assert _snapshot(source) == before
    assert _snapshot(loop_nest_vectorize_as_pow2(result)) == _snapshot(result)


def test_canonical_composition_recognizes_shared_vector_groups_as_fully_unrolled() -> (
    None
):
    source = _checked(
        LoopNest(
            abstract_dims=["i", "j"],
            root_node=LoopNestNode(
                root="C",
                interchange=["j", "i"],
                unroll={"j": 2},
                splits={"i": {"first": 0, "tail": 5}},
                children=[
                    LoopNestNode(
                        root=name,
                        tiles={"j": {"j48": 48}},
                        interchange=["i", "j48"],
                        vectorize=["j48"],
                    )
                    for name in ["first", "tail"]
                ],
            ),
        )
    )
    before = _snapshot(source)
    vectorized = _checked(loop_nest_vectorize_as_pow2(source))
    result = _checked(loop_nest_unroll_as_full(vectorized))
    root = result.root_node
    assert root is not None
    assert root.tiles == {"j": {"__u_j": 96}}
    assert root.unroll == {"__u_j": 2}
    for child in root.children:
        assert child.tiles == {"j": {"j48": 48, "__v_j48": 16}}
        assert child.unroll == {"j48": 3}
        assert child.vectorize == ["__v_j48"]
    assert _snapshot(source) == before


def test_empty_nest_is_cloned_without_root() -> None:
    source = LoopNest(abstract_dims=["i"], root_node=None)
    for transform in (loop_nest_vectorize_as_pow2, loop_nest_unroll_as_full):
        result = transform(source)
        assert result is not source
        assert result.root_node is None
        assert result.abstract_dims == source.abstract_dims


def test_oversized_unroll_preserves_rebased_branch() -> None:
    source = LoopNest(
        ["i"],
        LoopNestNode(
            root="C",
            interchange=["i"],
            splits={"i": {"first": 0, "tail": 6}},
            children=[
                LoopNestNode(root="first", interchange=["i"], unroll={"i": 8}),
                LoopNestNode(root="tail", interchange=["i"]),
            ],
        ),
    )
    analysis = analyze_loop_nest(source)
    assert analysis.loop_contexts["C/first"]["i"].trip_count == 6
    result = loop_nest_unroll_as_full(source)
    assert _snapshot(result) == _snapshot(source)


def _unroll_above_branch_tiles(right_size: int | None, factor: int = 2) -> LoopNest:
    def branch(name: str, size: int | None) -> LoopNestNode:
        return LoopNestNode(
            root=name,
            interchange=["k", "tile"] if size is not None else ["k"],
            tiles={"i": {"tile": size}} if size is not None else {},
        )

    return LoopNest(
        ["i", "k"],
        LoopNestNode(
            root="C",
            interchange=["i", "k", "left", "right"],
            splits={"k": {"left": 0, "right": 1}},
            unroll={"i": factor},
            children=[branch("left", 8), branch("right", right_size)],
        ),
    )


def test_unroll_uses_common_branch_tile_stride() -> None:
    source = _unroll_above_branch_tiles(8)
    analysis = analyze_loop_nest(source)
    assert analysis.loop_contexts["C"]["i"].next_inner_stride == 8

    result = loop_nest_unroll_as_full(source)
    root = result.root_node
    assert root is not None
    assert _tile_sizes(root) == [16]
    ((unrolled, factor),) = root.unroll.items()
    normalized = analyze_loop_nest(result)
    assert factor == normalized.loop_contexts["C"][unrolled].trip_count == 2
    assert [_tile_sizes(child) for child in root.children] == [[8], [8]]


def test_unit_unroll_leaves_branch_tiles_unchanged() -> None:
    result = loop_nest_unroll_as_full(_unroll_above_branch_tiles(8, factor=1))
    root = result.root_node
    assert root is not None
    assert not root.tiles
    assert not root.unroll
    assert [_tile_sizes(child) for child in root.children] == [[8], [8]]


@pytest.mark.parametrize("right_size", [4, None], ids=["different", "missing"])
def test_check_rejects_incompatible_shared_tiles(right_size: int | None) -> None:
    with pytest.raises(ScheduleValidationError, match="Conflicting tile requirements"):
        _unroll_above_branch_tiles(right_size).check()


def test_branch_unroll_names_shared_tile_not_enclosing_dimension() -> None:
    nest = _unroll_above_branch_tiles(8)
    assert nest.root_node is not None
    nest.root_node.unroll.clear()
    right = nest.root_node.children[1]
    right.tiles["i"]["inner"] = 2
    right.interchange.extend(["inner", "i"])
    right.unroll["i"] = 2
    with pytest.raises(
        ScheduleValidationError, match="Loop i is outside its logical scope at C/right"
    ):
        nest.check()

    right.interchange.pop()
    right.unroll.clear()
    right.unroll["tile"] = 2
    result = loop_nest_unroll_as_full(nest)
    assert result.root_node is not None
    rewritten = result.root_node.children[1]
    assert _tile_sizes(rewritten) == [8, 4, 2]
    ((unrolled, factor),) = rewritten.unroll.items()
    context = analyze_loop_nest(result).loop_contexts["C/right"][unrolled]
    assert context.trip_count == factor == 2


@pytest.mark.parametrize(
    ("sizes", "target", "expected_sizes"),
    [([16], "i", [64, 16]), ([60, 16], "i#60", [60, 16])],
    ids=["unknown-outer-extent", "partial-last-trip"],
)
def test_unroll_above_partition_preserves_branch_extents(
    sizes: list[int], target: str, expected_sizes: list[int]
) -> None:
    source = _tiled_nest(sizes, unroll={target: 4})
    assert source.root_node is not None
    root = source.root_node
    root.splits["i"] = {"left": 0, "right": 6}
    for branch in root.splits["i"]:
        root.add_child(LoopNestNode(root=branch, interchange=["i"]))

    result = loop_nest_unroll_as_full(source)

    assert result.root_node is not None
    assert _tile_sizes(result.root_node) == expected_sizes
    analysis = analyze_loop_nest(result)
    ((unrolled, factor),) = result.root_node.unroll.items()
    assert factor == analysis.loop_contexts["C"][unrolled].trip_count == 4
    assert analysis.extents["C/left"]["i"] == 6
    assert analysis.extents["C/right"]["i"] == 10
