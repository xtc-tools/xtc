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
_GPU_DIM = {0: "x", 1: "y", 2: "z"}
# Storage scopes accepted by tvm cache_read/cache_write
_TVM_MEM_SCOPES = {"global", "shared", "local", "warp", "shared.dyn"}


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
        self._dump_cast_schedule(root, sched)
        print(
            f"{sch} = decompose_reduction_initializers({sch})",
            file=outf,
        )

    def _dump_tile_splits(self, t_axes: list[str], node: LoopNestNode) -> list[str]:
        """Emit the tile splits of the axes and return all the axis names."""
        sch = self._sch_var
        outf = self._outf
        axes: list[str] = []
        for t_axis in t_axes:
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
        return axes

    def _dump_cast_schedule(self, node: LoopNestNode, sched: LoopNest) -> None:
        """Mirror the schedule on the cast stage of mixed type computation.

        The cast stage has the spatial dims of the operation only, without
        gpu thread bindings it stays on the host side where the tvm
        pipeline cannot lower its casts and the compilation fails. Replay
        the schedule of the main block on the cast stage: tiles, loop
        order, block/thread mapping, split, unroll, vectorize and
        parallelization, restricted to the spatial dims of the cast stage.
        """
        # TODO: Currently has_cast_stage depend of the attribute acc_dtype
        # if it is added to operator that does not do the cast it will do an error
        if not self._op.operator.has_cast_stage():
            return
        block = "O_cast"
        sch = self._sch_var
        outf = self._outf
        spatial_dims = list(self._op.operator.dims("P"))
        print(f'{block} = {sch}.get_sblock("{self._op.name}_cast")', file=outf)
        self._dump_schedule_node(
            block, spatial_dims, sched, node, cast_mode=True, cast_root=True
        )

    def _dump_schedule_node(
        self,
        block: str,
        dims: list[str],
        sched: LoopNest,
        node: LoopNestNode,
        cast_mode: bool = False,
        cast_root: bool = False,
    ):
        assert node is not None, "unexpected undefined node"
        sch = self._sch_var
        outf = self._outf
        print(f"{', '.join(dims)}, = {sch}.get_loops({block})", file=outf)
        if node.fuse_consumer_at and not cast_mode:
            print(f"O_F0 = {sch}.get_consumers({block})[0]", file=outf)
        if node.pack_at and not cast_mode:
            scope_of = {
                inp_idx: (mtype if mtype is not None else "global")
                for inp in node.pack_at.values()
                for inp_idx, mtype, _, _ in inp
            }
            for inp_idx, scope in scope_of.items():
                print(
                    f'I_R{inp_idx} = {sch}.cache_read({block}, {inp_idx}, "{scope}")',
                    file=outf,
                )
        if node.buffer_at and not cast_mode:
            wscope = next(iter(node.buffer_at.values()))
            wscope = wscope if wscope is not None else "global"
            print(f'O_W0 = {sch}.cache_write({block}, 0, "{wscope}")', file=outf)
        if node.fuse_producer_at and not cast_mode:
            producers = list({idx: None for idx in node.fuse_producer_at.values()})
            for prod_idx in producers:
                print(
                    f"I_F{prod_idx} = {sch}.get_producers({block})[{prod_idx}]",
                    file=outf,
                )
        axes = self._dump_tile_splits(dims, node)
        outer = [axis for axis in node.interchange if axis in axes]
        reorder = outer + axes
        split_axis: str | None = None
        if node.splits:
            maybe_axis = list(node.splits)[0]
            if not cast_mode or maybe_axis in axes:
                split_axis = maybe_axis
                reorder = outer + [split_axis] + axes
        reorder = list(dict.fromkeys(reorder))
        inner = [axis for axis in reorder if axis not in outer]
        print(f"{sch}.reorder({', '.join(reorder)})", file=outf)
        if cast_mode:
            # the reduction dims are not part of the cast stage, only
            # schedule the axes derived from its spatial dims
            cast_axes = set(axes)
            if split_axis is not None and split_axis in axes:
                cast_axes.add(split_axis)
            for i, gpu_axis in enumerate(node.gpu_block):
                if gpu_axis in cast_axes:
                    print(
                        f"{sch}.bind({gpu_axis}, 'blockIdx.{_GPU_DIM[i]}')",
                        file=outf,
                    )
            if node.gpu_block:
                for i, gpu_axis in enumerate(node.gpu_thread):
                    if gpu_axis in cast_axes:
                        print(
                            f"{sch}.bind({gpu_axis}, 'threadIdx.{_GPU_DIM[i]}')",
                            file=outf,
                        )
            for u_axis, _ in node.unroll.items():
                if u_axis in cast_axes:
                    print(f"{sch}.unroll({u_axis})", file=outf)
            for v_axis in node.vectorize:
                if v_axis in cast_axes:
                    print(f"{sch}.vectorize({v_axis})", file=outf)
            if node.parallelize:
                parallelize = [axis for axis in node.parallelize if axis in cast_axes]
                if parallelize:
                    if len(parallelize) > 1:
                        print(
                            f"{parallelize[-1]} = {sch}.fuse({', '.join(parallelize)})",
                            file=outf,
                        )
                    print(f"{sch}.parallel({parallelize[-1]})", file=outf)
        else:
            for i, gpu_axis in enumerate(node.gpu_block):
                print(f"{sch}.bind({gpu_axis}, 'blockIdx.{_GPU_DIM[i]}')", file=outf)
            # If there is no block there should be any gpu thread either
            if node.gpu_block:
                for i, gpu_axis in enumerate(node.gpu_thread):
                    print(
                        f"{sch}.bind({gpu_axis}, 'threadIdx.{_GPU_DIM[i]}')", file=outf
                    )
        if node.buffer_at and not cast_mode:
            for axis in node.buffer_at:
                print(f"{sch}.reverse_compute_at(O_W0, {axis})", file=outf)
        if node.pack_at and not cast_mode:
            for axis, v in node.pack_at.items():
                # FIXME: If the target is not nvgpu and is shared do not do the async copy
                pipeline_async = False
                for inp_idx, mtype, pad, alignment in v:
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
                    if mtype in ("shared", "shared.dyn") and alignment is not None:
                        self._dump_async_copy_distribution(
                            node, inp_idx, mtype, alignment
                        )
                        pipeline_async = True
                if pipeline_async:
                    self._dump_async_copy_annotations(node, axis)
        if node.fuse_producer_at and not cast_mode:
            for axis, prod_idx in node.fuse_producer_at.items():
                print(f"{sch}.compute_at(I_F{prod_idx}, {axis})", file=outf)
        if node.fuse_consumer_at and not cast_mode:
            for axis in node.fuse_consumer_at:
                print(f"{sch}.reverse_compute_at(O_F0, {axis})", file=outf)
        if not cast_mode:
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
            if cast_mode:
                continue
            print(
                f"{sch} = externalize_tile_below({sch}, {block}, "
                f"{external_axis}, {symbol!r})",
                file=outf,
            )
        if split_axis is not None:
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
                self._dump_schedule_node(
                    split_block,
                    inner,
                    sched,
                    node.children[idx],
                    cast_mode,
                )

    def _gpu_thread_extents(self, node: LoopNestNode) -> list[int]:
        # Iteration count of each gpu_thread axis, in bind order.
        extents: list[int] = []
        for axis in node.gpu_thread:
            extent: int | None = None
            for chain in node.tiles.values():
                names = list(chain)
                if axis not in names:
                    continue
                idx = names.index(axis)
                if idx + 1 < len(names):
                    extent = chain[axis] // chain[names[idx + 1]]
                else:
                    extent = chain[axis]
                break
            if extent is None:
                return []
            extents.append(extent)
        return extents

    def _dump_async_copy_distribution(
        self,
        node: LoopNestNode,
        input_idx: int,
        mtype: str | None,
        alignment: int | None,
    ) -> None:
        if mtype not in ("shared", "shared.dyn"):
            return
        if alignment is None:
            return
        if not node.gpu_thread:
            return
        extents = self._gpu_thread_extents(node)
        if not extents:
            return
        input_spec = self._op.np_inputs_spec()[input_idx]
        ndim = len(input_spec["shape"])
        if ndim < 1:
            return
        # cp.async only supports 4, 8 and 16-byte copies
        elt_size = np.dtype(input_spec["dtype"]).itemsize
        assert alignment * elt_size in (4, 8, 16), (
            f"pack_at alignment of input {input_idx} must copy 4, 8 or 16 "
            f"bytes ({alignment} elements of {input_spec['dtype']} = "
            f"{alignment * elt_size} bytes)"
        )
        # Vectorized compute accesses reinterpret the shared buffer element
        # type (FlattenBuffer), which leaves the copy store and load with
        # mismatched lanes and crashes the cp.async lowering, so keep the
        # copy scalar then.
        vectorize = not node.vectorize
        sch = self._sch_var
        print(
            f"{sch} = distribute_async_copy("
            f"{sch}, I_R{input_idx}, {ndim}, {extents}, {alignment}, "
            f"vectorize={vectorize})",
            file=self._outf,
        )

    def _dump_async_copy_annotations(self, node: LoopNestNode, axis: str) -> None:
        # Annotate the pack loop so InjectSoftwarePipeline (in the s_tir
        # pipeline) turns the async global->shared copies into a software
        # pipeline; InjectPTXAsyncCopy then lowers them to cp.async.
        packs = node.pack_at[axis]
        sch = self._sch_var
        outf = self._outf

        def is_async(mtype: str | None, alignment: int | None) -> bool:
            return mtype in ("shared", "shared.dyn") and alignment is not None

        if not any(is_async(mtype, alignment) for _, mtype, _, alignment in packs):
            # No shared, aligned copy: nothing to pipeline
            return
        if not node.interchange or axis == node.interchange[-1]:
            # No loop below the pack axis: the stage counts below would
            # not match the statements of the loop body
            return
        if (
            axis in node.buffer_at
            or axis in node.fuse_producer_at
            or axis in node.fuse_consumer_at
        ):
            return
        if any(baxis in node.gpu_thread for baxis in node.buffer_at):
            return
        stage = [
            0 if is_async(mtype, alignment) else 1 for _, mtype, _, alignment in packs
        ] + [1]
        order = list(range(len(packs) + 1))
        print(f'{sch}.annotate({axis}, "software_pipeline_stage", {stage})', file=outf)
        print(f'{sch}.annotate({axis}, "software_pipeline_order", {order})', file=outf)
        print(
            f'{sch}.annotate({axis}, "software_pipeline_async_stages", [0])',
            file=outf,
        )

    @override
    def emit(self, scheduler: "TVMScheduler"):
        sched = scheduler.get_loop_nest()
        sched = tvm_update_loopnest_for_codegen(sched)
        sched.check()
        self._dump_schedule(sched)


def tvm_cache_read_factor_offset(
    op: TVMOperation, input_idx: int, pad: bool
) -> tuple[int, int, int]:
    input_spec = op.np_inputs_spec()[input_idx]
    if not pad:
        return 0, 0, 0
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
            gpu_block=deepcopy(node.gpu_block),
            gpu_thread=deepcopy(node.gpu_thread),
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

    def _emit_schedule_code(self) -> str:
        io = StringIO()
        emitter = TVMScheduleEmitterTIR(op=self._op, outf=io)
        emitter.emit(self)
        return io.getvalue()

    @override
    def schedule(self) -> itf.schd.Schedule:
        assert self._op.name is not None
        schedule_impl = {self._op.name: self._emit_schedule_code()}
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
        assert mtype is None or mtype in _TVM_MEM_SCOPES
        self._plain_sch.buffer_at(axis, mtype, root)

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        alignment: int | None = None,
        root: str = DEFAULT_ROOT,
    ) -> None:
        assert mtype is None or mtype in _TVM_MEM_SCOPES
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        assert alignment is None or alignment > 0
        self._plain_sch.pack_at(axis, input_idx, mtype, pad, alignment, root)

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

    @override
    def gpu_block(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.gpu_block(axes, root)

    @override
    def gpu_thread(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.gpu_thread(axes, root)

    @override
    def gpu_lane(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        # TODO: not implemented for now
        pass

    @override
    def gpu_warp(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
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
