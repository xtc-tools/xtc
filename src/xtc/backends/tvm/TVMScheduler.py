#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import TextIO, TypeAlias
from io import StringIO
import numpy as np
from copy import deepcopy

from xtc.utils.math import pow2divisor
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.plain_schedule import PlainNodeSchedule, PlainNodeScheduler
from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.loop_nest_builder import LoopNestBuilder
import xtc.backends.tvm as backend
import xtc.itf as itf

from .TVMOps import TVMOperation

# Actual backend Schedule implementation is a mapping
# from op name to the TVM schedule string
ScheduleImpl: TypeAlias = dict[str, str]


class TVMScheduleEmitter(ABC):
    @abstractmethod
    def emit(self, scheduler: "TVMScheduler"): ...


class TVMScheduleEmitterTIR(TVMScheduleEmitter):
    def __init__(
        self,
        op: TVMOperation,
        obj_var: str = "obj",
        sch_var: str = "sch",
        outf: TextIO = sys.stdout,
    ):
        self._op = op
        self._obj_var = obj_var
        self._sch_var = sch_var
        self._outf = outf

    def _dump_schedule(self, sched: LoopNest):
        root = sched.root_node
        if root is None:
            return
        block = "O"
        sch = self._sch_var
        outf = self._outf
        print(f'{block} = {sch}.get_sblock("{self._op.name}")', file=outf)
        dims = sched.abstract_dims
        self._dump_schedule_node(block, dims, sched, root)

    def _dump_schedule_node(
        self, block: str, dims: list[str], sched: LoopNest, node: LoopNestNode
    ):
        assert node is not None, "unexpected undefined node"
        sch = self._sch_var
        outf = self._outf
        print(f"{', '.join(dims)}, = {sch}.get_loops({block})", file=outf)
        if node.fuse_consumer_at:
            print(f"O_F0 = {sch}.get_consumers({block})[0]", file=outf)
        if node.pack_at:
            inputs = list({inp[0]: None for inp in node.pack_at.values()})
            for inp_idx in inputs:
                print(
                    f'I_R{inp_idx} = {sch}.cache_read({block}, {inp_idx}, "global")',
                    file=outf,
                )
        if node.buffer_at:
            print(f'O_W0 = {sch}.cache_write({block}, 0, "global")', file=outf)
        if node.fuse_producer_at:
            producers = list({idx: None for idx in node.fuse_producer_at.values()})
            for prod_idx in producers:
                print(
                    f"I_F{prod_idx} = {sch}.get_producers({block})[{prod_idx}]",
                    file=outf,
                )
        axes = []
        for t_axis in dims:
            t_tiles = node.tiles.get(t_axis, {})
            t_names = [t_axis] + list(t_tiles)
            axes += t_names
            if not t_tiles:
                continue
            tile_sizes = list(t_tiles.values())
            factors = [
                outer // inner for outer, inner in zip(tile_sizes, tile_sizes[1:])
            ]
            factors.append(tile_sizes[-1])
            t_factors = ["None"] + [str(factor) for factor in factors]
            print(
                f"{', '.join(t_names)}, = {sch}.split({t_axis}, factors=[{', '.join(t_factors)}])",
                file=outf,
            )
        outer = [axis for axis in node.interchange if axis in axes]
        reorder = outer + axes
        if node.splits:
            split_axis = list(node.splits)[0]
            reorder = outer + [split_axis] + axes
        reorder = list(dict.fromkeys(reorder))
        inner = [axis for axis in reorder if axis not in outer]
        print(f"{sch}.reorder({', '.join(reorder)})", file=outf)
        if node.buffer_at:
            for axis in node.buffer_at:
                print(f"{sch}.reverse_compute_at(O_W0, {axis})", file=outf)
        if node.pack_at:
            for axis, (inp_idx, mtype, pad) in node.pack_at.items():
                print(f"{sch}.compute_at(I_R{inp_idx}, {axis})", file=outf)
                dim, factor, offset = tvm_cache_read_factor_offset(
                    self._op, inp_idx, pad
                )
                if factor != 0:
                    print(
                        f"{sch}.storage_align(I_R{inp_idx}, 0, ",
                        f"axis={dim}, factor={factor}, offset={offset})",
                        file=outf,
                    )
        if node.fuse_producer_at:
            for axis, prod_idx in node.fuse_producer_at.items():
                print(f"{sch}.compute_at(I_F{prod_idx}, {axis})", file=outf)
        if node.fuse_consumer_at:
            for axis in node.fuse_consumer_at:
                print(f"{sch}.reverse_compute_at(O_F0, {axis})", file=outf)
        for u_axis, u_factor in node.unroll.items():
            print(f"{sch}.unroll({u_axis})", file=outf)
        for v_axis in node.vectorize:
            print(f"{sch}.vectorize({v_axis})", file=outf)
        if node.parallelize:
            if len(node.parallelize) > 1:
                print(
                    f"{node.parallelize[-1]} = {sch}.fuse({', '.join(node.parallelize)})",
                    file=outf,
                )
            print(
                f"{sch}.parallel({node.parallelize[-1]})",
                file=outf,
            )
        for external_axis, symbol in node.external_at.items():
            print(
                f"{sch} = externalize_tile_below({sch}, {block}, "
                f"{external_axis}, {symbol!r})",
                file=outf,
            )
        if node.splits:
            split_axis = list(node.splits)[0]
            splits = node.splits[split_axis]
            split_factors = [str(x) for x in list(splits.values())[1:]]
            print(
                f"{sch}, {block}_SPLITS = loop_partition_rebased("
                f"{sch}, {split_axis}, [{', '.join(split_factors)}, None])",
                file=outf,
            )
            for idx, split in enumerate(splits):
                split_block = f"{block}_{split_axis}_{idx}"
                print(
                    f"{split_block} = {block}_SPLITS[{idx}]",
                    file=outf,
                )
                self._dump_schedule_node(split_block, inner, sched, node.children[idx])

    @override
    def emit(self, scheduler: "TVMScheduler"):
        sched = scheduler.get_loop_nest()
        sched = tvm_update_loopnest_for_codegen(sched)
        sched.check()
        self._dump_schedule(sched)


def tvm_cache_read_factor_offset(
    op: TVMOperation, input_idx: int, pad: bool
) -> tuple[int, int, int]:
    if not pad:
        return 0, 0, 0
    input_spec = op.np_inputs_spec()[input_idx]
    if len(input_spec["shape"]) < 2:
        return 0, 0, 0
    # Assume for CPU common number of sets and line size for L1
    # Except to minimize conflicts by setting the inner axis
    # size to a factor of num_sets and adding +1
    num_sets, line_size = 64, 64
    elt_size = np.dtype(input_spec["dtype"]).itemsize
    elts_per_line = line_size // elt_size
    return -2, elts_per_line * num_sets, elts_per_line


def tvm_update_loopnest_for_codegen(sched: LoopNest) -> LoopNest:
    def _update_loopnode(node: LoopNestNode) -> LoopNestNode:
        axis_dim = {axis: dim for dim, tiles in node.tiles.items() for axis in tiles}
        adjusted_tiles = {}
        adjusted_unrolling = {
            k: v for k, v in node.unroll.items() if k not in node.vectorize
        }
        adjusted_unrolls = list(adjusted_unrolling)
        adjusted_vectorization = node.vectorize[:]
        adjusted_permutation = node.interchange[:]
        dims_to_update = set(
            [
                axis_dim.get(axis, axis)
                for axis in adjusted_unrolls + adjusted_vectorization
            ]
        )
        tiled_dims = list(node.tiles)
        additional_dims = sorted(dims_to_update - set(tiled_dims))
        for dim in tiled_dims + additional_dims:
            dim_tiles = node.tiles.get(dim, {})
            adjusted_dim_tiles = {}
            for axis, size in {dim: 0, **dim_tiles}.items():
                if size:
                    adjusted_dim_tiles.update({axis: size})
                if axis in adjusted_unrolling:
                    assert axis not in adjusted_vectorization
                    assert axis in adjusted_permutation
                    unroll = adjusted_unrolling[axis]
                    if size == 0 or unroll < size:
                        axis_idx = adjusted_unrolls.index(axis)
                        new_axis = f"__u_{axis}"
                        adjusted_dim_tiles.update({new_axis: unroll})
                        adjusted_unrolls[axis_idx] = new_axis
                        del adjusted_unrolling[axis]
                        adjusted_unrolling.update({new_axis: unroll})
                        adjusted_permutation.insert(
                            adjusted_permutation.index(axis) + 1,
                            new_axis,
                        )
                elif axis in adjusted_vectorization:
                    assert axis not in adjusted_unrolling
                    assert axis in adjusted_permutation
                    if size > 0:
                        pow2 = pow2divisor(size)
                        unroll = size // pow2
                        if unroll > 1:
                            axis_idx = adjusted_vectorization.index(axis)
                            new_axis = f"__v_{axis}"
                            adjusted_dim_tiles.update({new_axis: pow2})
                            adjusted_vectorization[axis_idx] = new_axis
                            adjusted_unrolls.append(axis)
                            adjusted_unrolling.update({axis: unroll})
                            adjusted_permutation.insert(
                                adjusted_permutation.index(axis) + 1,
                                new_axis,
                            )
            if adjusted_dim_tiles:
                adjusted_tiles[dim] = adjusted_dim_tiles
        adjusted_unrolling = {u: adjusted_unrolling[u] for u in adjusted_unrolls}
        updated_node = LoopNestNode(
            root=node.root,
            tiles=adjusted_tiles,
            splits=deepcopy(node.splits),
            interchange=adjusted_permutation,
            vectorize=adjusted_vectorization,
            parallelize=deepcopy(node.parallelize),
            unroll=adjusted_unrolling,
            buffer_at=deepcopy(node.buffer_at),
            pack_at=deepcopy(node.pack_at),
            fuse_producer_at=deepcopy(node.fuse_producer_at),
            fuse_consumer_at=deepcopy(node.fuse_consumer_at),
            external_at=deepcopy(node.external_at),
            split_origin=deepcopy(node.split_origin),
        )
        for child in node.children:
            updated_child = _update_loopnode(child)
            updated_node.add_child(updated_child)
        return updated_node

    root = sched.root_node
    if root is not None:
        root = _update_loopnode(root)
    return LoopNest(
        abstract_dims=sched.abstract_dims,
        root_node=root,
    )


class TVMScheduler(itf.schd.Scheduler):
    def __init__(
        self,
        backend: "backend.TVMBackend",
        nodes: list[str] | None = None,
        default_node: str | None = None,
    ) -> None:
        self._backend = backend
        if nodes is None:
            self._scheduled_ops = backend._ops
        else:
            self._scheduled_ops = {name: backend._ops[name] for name in nodes}
        assert len(self._scheduled_ops) > 0
        if default_node is None:
            candidate_ops = list(self._scheduled_ops.values())
        else:
            assert default_node in self._scheduled_ops
            candidate_ops = [
                v for k, v in self._scheduled_ops.items() if k == default_node
            ]
        self._op = candidate_ops[-1]
        self._plain_sch = PlainNodeScheduler(
            self._op.name,
            self._op.name,  # TODO: ident
            list(self._op.operator.dims()),
        )

    @property
    def _default_node_name(self) -> str:
        return self._op.name

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        io = StringIO()
        emitter = TVMScheduleEmitterTIR(op=self._op, outf=io)
        emitter.emit(self)
        sched = io.getvalue()
        assert self._op.name is not None
        schedule_impl = {self._op.name: sched}
        return TVMSchedule(scheduler=self, schedule_impl=schedule_impl)

    @override
    def set_dims(self, dims: list[str]) -> None:
        self._plain_sch.set_dims(dims)

    @override
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.split(dim, segments, root)

    @override
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.tile(dim, tiles, root)

    @override
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.interchange(permutation, root)

    @override
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        assert mtype is None or mtype == "global"
        self._plain_sch.buffer_at(axis, mtype, root)

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        assert mtype is None or mtype == "global"
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        self._plain_sch.pack_at(axis, input_idx, mtype, pad, root)

    @override
    def vectorize(
        self,
        # widths are ignored: TVM has no masked vectorize
        axes: list[str] | dict[str, int | None],
        root: str = DEFAULT_ROOT,
    ) -> None:
        self._plain_sch.vectorize(axes, root)

    @override
    def external_at(self, axis: str, symbol: str, root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.external_at(axis, symbol, root)

    @override
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.parallelize(axes, root)

    @override
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.unroll(unrolls, root)

    @override
    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        self._plain_sch.fuse_producer_at(axis, input_idx, root)

    @override
    def fuse_consumer_at(self, axis: str, root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.fuse_consumer_at(axis, root)

    @override
    def define_memory_mesh(self, axes: dict[str, int]) -> None:
        # TODO: not implemented for now
        pass

    @override
    def define_processor_mesh(self, axes: dict[str, int]) -> None:
        # TODO: not implemented for now
        pass

    @override
    def distribute(
        self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT
    ) -> None:
        # TODO: not implemented for now
        pass

    @override
    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ) -> None:
        # TODO: not implemented for now
        pass

    def _get_plain_schedule(self) -> PlainNodeSchedule:
        return self._plain_sch.get_plain_schedule()

    @override
    def get_loop_nest(self) -> LoopNest:
        return LoopNestBuilder.from_plain_node_schedule(self._get_plain_schedule())

    @override
    def __str__(self) -> str:
        return str(self._get_plain_schedule())


class TVMSchedule(itf.schd.Schedule):
    def __init__(self, scheduler: "TVMScheduler", schedule_impl: ScheduleImpl) -> None:
        self._scheduler = scheduler
        self._schedule_impl = schedule_impl

    @property
    def schedule_impl(self) -> ScheduleImpl:
        return self._schedule_impl

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @override
    def __str__(self) -> str:
        return "\n".join(self._schedule_impl.values())
