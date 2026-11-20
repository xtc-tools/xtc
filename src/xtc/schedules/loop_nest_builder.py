#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any

from .loop_names import basename, make_loop_name, parent_name, rooted_name
from .loop_nest import LoopNest, LoopNestNode
from .plain_schedule import PlainNodeSchedule


class LoopNestBuilder:
    @classmethod
    def from_plain_node_schedule(cls, node_sched: PlainNodeSchedule) -> LoopNest:
        top_name = node_sched.node_name
        root_node = cls._get_loop_node(top_name, node_sched.node_name, node_sched)
        return LoopNest(abstract_dims=node_sched.dims[:], root_node=root_node)

    @classmethod
    def _get_single_node(
        cls,
        top_name: str,
        full_name: str,
        node_sched: PlainNodeSchedule,
    ) -> LoopNestNode:
        def is_node_axis(axis: str):
            return parent_name(rooted_name(axis, top_name)) == full_name

        def is_node_root(name: str):
            return rooted_name(name, top_name) == full_name

        def localize_axis_list(axis_list: list[str]) -> list[str]:
            return [basename(axis) for axis in axis_list if is_node_axis(axis)]

        def localize_axis_dict(axis_dict: dict[str, Any]) -> dict[str, Any]:
            return {
                basename(axis): v for axis, v in axis_dict.items() if is_node_axis(axis)
            }

        def localize_axis_tuples(
            axis_tuples: list[tuple[str, Any]],
        ) -> list[tuple[str, Any]]:
            return [
                (basename(axis), v) for axis, v in axis_tuples if is_node_axis(axis)
            ]

        perms = {
            full_name: v  # noqa: B035
            for p, v in node_sched.permutation.items()
            if is_node_root(p)
        }
        interchange = [basename(p) for p in perms[full_name]]
        splits = {
            basename(axis): {basename(k): v for k, v in node_sched.splits[axis].items()}
            for axis in node_sched.splits
            if is_node_axis(axis)
        }
        tiles = {
            basename(axis): {
                basename(tile_name): size for tile_name, size in axis_tiles.items()
            }
            for axis, axis_tiles in node_sched.tiles.items()
            if is_node_axis(axis)
        }
        vectorize = localize_axis_list(node_sched.vectorization)
        parallelize = localize_axis_list(node_sched.parallelization)
        unroll = localize_axis_dict(node_sched.unrolling)
        # TODO: loop nest supports only one buffer per axis
        buffer_at = {
            basename(axis): v[0]
            for axis, v in node_sched.write_buffers.items()
            if is_node_axis(axis)
        }
        # TODO: loop nest supports only one pack per axis
        pack_at = {
            basename(axis): v[0]
            for axis, v in node_sched.packed_buffers.items()
            if is_node_axis(axis)
        }
        # TODO: loop nest supports only one fuse per axis
        fuse_producer_at = dict(localize_axis_tuples(node_sched.fused_producers))
        # TODO: loop nest supports only one fuse consumer per axis
        fuse_consumer_at = localize_axis_list(node_sched.fused_consumers)
        external_at = localize_axis_dict(node_sched.externals)

        return LoopNestNode(
            root=basename(full_name),
            interchange=interchange,
            tiles=tiles,
            vectorize=vectorize,
            parallelize=parallelize,
            unroll=unroll,
            buffer_at=buffer_at,
            pack_at=pack_at,
            fuse_producer_at=fuse_producer_at,
            fuse_consumer_at=fuse_consumer_at,
            external_at=external_at,
            splits=splits,
        )

    @classmethod
    def _get_loop_node(
        cls,
        top_name: str,
        full_name: str,
        node_sched: PlainNodeSchedule,
    ) -> LoopNestNode:
        node = cls._get_single_node(top_name, full_name, node_sched)
        for branches in node.splits.values():
            for branch in branches:
                child = cls._get_loop_node(
                    top_name, make_loop_name(full_name, branch), node_sched
                )
                node.add_child(child)
        return node
