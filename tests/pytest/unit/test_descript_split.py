#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import pytest

from xtc.schedules.descript import Descript
from xtc.schedules.exceptions import ScheduleInterpretError


@pytest.mark.parametrize("split", ["i[0:4]", "i[:4:]"])
def test_explicit_split_axis_keeps_its_position_and_uses_branch_extent(
    split: str,
) -> None:
    nest = (
        Descript(abstract_dims=["i", "j"], abstract_dim_sizes={"i": 12})
        .loop_nest(
            "C",
            {split: {"j": {}, "i": {"unroll": True}}, "i[4:]": {}},
        )
        .apply_sample({})
    )
    nest.check()

    assert nest.root_node is not None
    branch = nest.root_node.children[0]
    assert branch.interchange == ["j", "i"]
    assert branch.unroll == {"i": 4}


def test_full_unroll_uses_branch_tile_trip_count() -> None:
    nest = (
        Descript(abstract_dims=["i"])
        .loop_nest(
            "C",
            {"i[:5]": {"i": {"unroll": True}, "i#3": {}}, "i[5:]": {}},
        )
        .apply_sample({})
    )
    nest.check()

    assert nest.root_node is not None
    assert nest.root_node.children[0].unroll == {"i": 2}


def test_omitted_split_axis_is_prepended_once() -> None:
    nest = Descript(abstract_dims=["i", "j"], abstract_dim_sizes={"i": 12}).loop_nest(
        "C", {"i[0:4]": {"j": {}}, "i[4:]": {}}
    )

    assert nest.root_node is not None
    assert nest.root_node.children[0].interchange == ["i", "j"]


def test_full_unroll_with_unknown_branch_extent_needs_factor() -> None:
    with pytest.raises(ScheduleInterpretError, match="size being unknown"):
        Descript(abstract_dims=["i"]).loop_nest(
            "C", {"i[:4]": {}, "i[4:]": {"i": {"unroll": True}}}
        )
