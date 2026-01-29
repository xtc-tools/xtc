#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
from typing_extensions import override
from typing import TextIO, TypeAlias
from io import StringIO
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from xtc.utils.math import pow2divisor
from xtc.itf.schd.scheduler import DEFAULT_ROOT
import xtc.backends.tvm as backend
import xtc.itf as itf

from .TVMOps import TVMOperation

# Actual backend Schedule implementation is a mapping
# from op name to the TVM schedule string
ScheduleImpl: TypeAlias = dict[str, str]


@dataclass(frozen=True)
class TVMPlainSchedule:
    dims: list[str]
    tiles: dict[str, dict[str, int]]
    permutation: list[str]
    parallelization: list[str]
    unrolling: dict[str, int]
    vectorization: list[str]
    write_caches: list[str]
    read_buffers: list[tuple[str, int, bool]]
    fused: list[tuple[str, int]]


class TVMScheduleEmitter:
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

    def _parallel_dims(self, sched: TVMPlainSchedule) -> list[str]:
        op_dims = self._op.operator.dims()
        return [sched.dims[op_dims.index(d)] for d in self._op.operator.dims("P")]

    def _reduction_dims(self, sched: TVMPlainSchedule) -> list[str]:
        op_dims = self._op.operator.dims()
        return [sched.dims[op_dims.index(d)] for d in self._op.operator.dims("R")]

    def _update_schedule_for_codegen(self, sch: TVMPlainSchedule) -> TVMPlainSchedule:
        unrolling = sch.unrolling
        vectorization = sch.vectorization
        adjusted_tiles = {}
        adjusted_unrolling = {
            k: v for k, v in sch.unrolling.items() if k not in vectorization
        }
        adjusted_unrolls = list(adjusted_unrolling)
        adjusted_vectorization = sch.vectorization[:]
        adjusted_permutation = sch.permutation[:]
        for dim, dim_tiles in sch.tiles.items():
            adjusted_dim_tiles = {}
            for axis, size in dim_tiles.items():
                adjusted_dim_tiles.update({axis: size})
                if axis in adjusted_unrolling:
                    assert axis not in adjusted_vectorization
                    unroll = unrolling[axis]
                    if unroll < size:
                        axis_idx = adjusted_unrolls.index(axis)
                        new_axis = f"__u_{axis}"
                        adjusted_dim_tiles.update({new_axis: unroll})
                        del adjusted_unrolls[axis_idx]
                        adjusted_unrolls.insert(axis_idx, new_axis)
                        adjusted_unrolling.update({new_axis: unroll})
                        adjusted_permutation.insert(
                            adjusted_permutation.index(axis) + 1,
                            new_axis,
                        )
                elif axis in adjusted_vectorization:
                    pow2 = pow2divisor(size)
                    unroll = size // pow2
                    if unroll > 1:
                        axis_idx = adjusted_vectorization.index(axis)
                        new_axis = f"__v_{axis}"
                        adjusted_dim_tiles.update({new_axis: pow2})
                        del adjusted_vectorization[axis_idx]
                        adjusted_vectorization.insert(axis_idx, new_axis)
                        adjusted_unrolls.append(axis)
                        adjusted_unrolling.update({axis: unroll})
                        adjusted_permutation.insert(
                            adjusted_permutation.index(axis) + 1,
                            new_axis,
                        )
            adjusted_tiles[dim] = adjusted_dim_tiles
        adjusted_unrolling = {u: adjusted_unrolling[u] for u in adjusted_unrolls}
        adjusted = TVMPlainSchedule(
            dims=deepcopy(sch.dims),
            tiles=adjusted_tiles,
            permutation=adjusted_permutation,
            parallelization=deepcopy(sch.parallelization),
            unrolling=adjusted_unrolling,
            vectorization=adjusted_vectorization,
            write_caches=deepcopy(sch.write_caches),
            read_buffers=deepcopy(sch.read_buffers),
            fused=deepcopy(sch.fused),
        )
        return adjusted

    def _full_packs(self, sched: TVMPlainSchedule) -> dict[str, tuple[int, int, int]]:
        def factor_offset(input_idx: int, pad: bool):
            if not pad:
                return 0, 0
            input_spec = self._op.np_inputs_spec()[input_idx]
            if len(input_spec["shape"]) < 2:
                return 0, 0
            # Assume for CPU common number of sets and line size for L1
            # Except to minimize conflicts by setting the inner axis
            # size to a factor of num_sets and adding +1
            num_sets, line_size = 64, 64
            elt_size = np.dtype(input_spec["dtype"]).itemsize
            elts_per_line = line_size // elt_size
            return elts_per_line * num_sets, elts_per_line

        packs = {}
        for axis, input_idx, pad in sched.read_buffers:
            factor, offset = factor_offset(input_idx, pad)
            packs[axis] = (input_idx, factor, offset)
        return packs

    def _full_fuses(self, sched: TVMPlainSchedule) -> dict[str, tuple[int]]:
        fuses = {}
        for axis, input_idx in sched.fused:
            fuses[axis] = (input_idx,)
        return fuses

    def _full_tilings(self, sched: TVMPlainSchedule) -> dict[str, tuple[str, str, int]]:
        order = sched.permutation
        tiles = sched.tiles
        tilings = {}
        for dim, dim_tiles in tiles.items():
            t_axes = list(dim_tiles.keys())
            t_sizes = list(dim_tiles.values())
            tilings[dim] = (dim, "", t_sizes[0])
            for idx in range(1, len(dim_tiles)):
                tilings[t_axes[idx]] = (dim, t_axes[idx - 1], t_sizes[idx])
        tilings = {axis: tilings[axis] for axis in order}
        return tilings

    def _write_buffer_tiling(
        self,
        sched: TVMPlainSchedule,
        axis: str,
        tilings: dict[str, tuple[str, str, int]],
    ) -> tuple[dict[str, tuple[str, str, int]], dict[str, tuple[str, str, int]]]:
        child = {
            parent: (dim, axis, factor)
            for axis, (dim, parent, factor) in tilings.items()
        }
        tiles_axis = list(tilings)
        tile_idx = tiles_axis.index(axis)
        outer_tiles = dict(list(tilings.items())[: tile_idx + 1])
        inner_tiles = dict(list(tilings.items())[tile_idx + 1 :])
        outers_dims = set()
        for axis, (dim, parent, factor) in list(outer_tiles.items()):
            outers_dims.add(dim)
            factor = child.get(axis, ("", "", 1))[2]
            outer_tiles[f"{dim}_"] = (dim, axis, factor)
        dims = set()
        parallel_dims = self._parallel_dims(sched)
        for axis, (dim, parent, factor) in list(inner_tiles.items()):
            if outers_dims and dim not in outers_dims:
                outers_dims.add(dim)
                if dim in parallel_dims:
                    outer_tiles[f"{dim}_"] = (dim, dim, 0)
            if dim not in dims:
                dims.add(dim)
                parent = ""
            inner_tiles[axis] = (dim, parent, factor)
        return outer_tiles, inner_tiles

    def _full_write_buffers(
        self, sched: TVMPlainSchedule
    ) -> dict[tuple[str, str, str], dict[str, tuple[str, str, int]]]:
        tilings = self._full_tilings(sched)
        reorder_idx = {axis: idx for idx, axis in enumerate(tilings)}
        write_axis = sorted(sched.write_caches, key=lambda axis: reorder_idx[axis])
        buffer_tilings = {}
        out = ("O", "", "")
        tiling = tilings
        for idx, axis in enumerate(write_axis):
            outer_tiling, inner_tiling = self._write_buffer_tiling(sched, axis, tiling)
            buffer_tilings[out] = outer_tiling
            out = (f"O_W{idx}", out[0], axis)
            tiling = inner_tiling
        buffer_tilings[out] = tiling
        return buffer_tilings

    def _emit_assign_axis(
        self, sched: TVMPlainSchedule, sch: str, tens: str, outf: TextIO
    ) -> None:
        parallel_dims = self._parallel_dims(sched)
        reduction_dims = self._reduction_dims(sched)
        if parallel_dims:
            print(f"{', '.join(parallel_dims)}, = {tens}.op.axis", file=outf)
        if reduction_dims:
            print(f"{', '.join(reduction_dims)}, = {tens}.op.reduce_axis", file=outf)

    def _emit_assign_tilings(
        self,
        sched: TVMPlainSchedule,
        sch: str,
        tens: str,
        tilings: dict[str, tuple[str, str, int]],
        outf: TextIO,
    ) -> None:
        for axis, (dim, parent, factor) in tilings.items():
            if not parent:
                if axis != dim:
                    print(f"{axis} = {dim}", file=outf)
                continue
            if factor > 0:
                print(
                    f"{parent}, {axis} = {sch}[{tens}].split({parent}, factor={factor})",
                    file=outf,
                )
            else:
                print(f"{axis} = {parent}", file=outf)
        print(f"{sch}[{tens}].reorder({', '.join(tilings)})", file=outf)

    def _dump_schedule(self, sched: TVMPlainSchedule):
        tilings = self._full_write_buffers(sched)
        packings = self._full_packs(sched)
        fuses = self._full_fuses(sched)
        obj = self._obj_var
        sch = self._sch_var
        outf = self._outf
        if packings:
            print(f"INPS = list({obj}.values())[:-1]", file=outf)
        for (tens, parent, axis), tiles in tilings.items():
            if not parent:
                print(f"{tens} = {obj}['{self._op.name}']", file=outf)
            else:
                print(f'{tens} = {sch}.cache_write({parent}, "local")', file=outf)
            for tile_axis in tiles:
                if tile_axis in packings:
                    inp_idx, _, _ = packings[tile_axis]
                    print(
                        f'I_R{inp_idx} = {sch}.cache_read(INPS[{inp_idx}], "local", [{tens}])',
                        file=outf,
                    )
                if tile_axis in fuses:
                    (inp_idx,) = fuses[tile_axis]
                    print(
                        f"I_F{inp_idx} = {tens}.op.input_tensors[{inp_idx}]", file=outf
                    )
        for idx, ((tens, parent, axis), tiles) in enumerate(tilings.items()):
            if parent:
                print(f"{sch}[{tens}].compute_at({sch}[{parent}], {axis})", file=outf)
            self._emit_assign_axis(sched, sch, tens, outf)
            self._emit_assign_tilings(sched, sch, tens, tiles, outf)
            for tile_axis in tiles:
                if tile_axis in packings:
                    inp_idx, factor, offset = packings[tile_axis]
                    print(
                        f"{sch}[I_R{inp_idx}].compute_at({sch}[{tens}], {tile_axis})",
                        file=outf,
                    )
                    if factor != 0:
                        print(
                            f"{sch}[I_R{inp_idx}].storage_align(I_R{inp_idx}.op.axis[-2], factor={factor}, offset={offset})",
                            file=outf,
                        )
                if tile_axis in fuses:
                    (inp_idx,) = fuses[tile_axis]
                    print(
                        f"{sch}[I_F{inp_idx}].compute_at({sch}[{tens}], {tile_axis})",
                        file=outf,
                    )
            for u_axis in sched.unrolling:
                if u_axis in tiles:
                    print(f"{sch}[{tens}].unroll({u_axis})", file=outf)
            for v_axis in sched.vectorization:
                if v_axis in tiles:
                    print(f"{sch}[{tens}].vectorize({v_axis})", file=outf)
            if sched.parallelization:
                if sched.parallelization[0] in tiles:
                    if len(sched.parallelization) > 1:
                        print(
                            f"{sched.parallelization[-1]} = {sch}[{tens}].fuse({', '.join(sched.parallelization)})",
                            file=outf,
                        )
                    print(
                        f"{sch}[{tens}].parallel({sched.parallelization[-1]})",
                        file=outf,
                    )

    def emit(self, sched: TVMPlainSchedule):
        # First adjust schedule to fix code gen limitations before emit
        sched = self._update_schedule_for_codegen(sched)
        self._dump_schedule(sched)


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
        self._abstract_dims = {d: d for d in self._op.operator.dims()}
        self._sizes = list(self._op.operator.dims_sizes().values())
        self.dims = list(self._op.operator.dims())
        self.tiles: dict[str, dict[str, int]] = {
            k: {k: v} for k, v in zip(self.dims, self._sizes)
        }
        self.permutation: list[str] = list(self.tiles.keys())
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.write_caches: list[str] = []
        self.read_buffers: list[tuple[str, int, bool]] = []
        self.fused: list[tuple[str, int]] = []
        self._update_loops()

    @property
    def dims_sizes(self) -> dict[str, int]:
        return {d: s for d, s in zip(self.dims, self._sizes)}

    @property
    def parallel_dims(self) -> list[str]:
        return [self._abstract_dims[d] for d in self._op.operator.dims("P")]

    @property
    def reduction_dims(self) -> list[str]:
        return [self._abstract_dims[d] for d in self._op.operator.dims("R")]

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        io = StringIO()
        emitter = TVMScheduleEmitter(op=self._op, outf=io)
        schedule = self._get_plain_schedule()
        emitter.emit(schedule)
        sched = io.getvalue()
        assert self._op.name is not None
        schedule_impl = {self._op.name: sched}
        return TVMSchedule(scheduler=self, schedule_impl=schedule_impl)

    def _update_loops(self):
        loops = dict()
        parallels = []
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
                if k in self.parallel_dims:
                    parallels.append(dim_name)
        self._working_dims = loops
        self._working_parallel_dims = parallels
        self.permutation = list(self._working_dims.keys())

    @override
    def set_dims(self, dims: list[str]) -> None:
        assert len(dims) == len(self.dims)
        self.dims = dims[:]
        self._abstract_dims = {d: a for d, a in zip(self._op.operator.dims(), dims)}
        self.tiles = {k: {k: v} for k, v in zip(self.dims, self._sizes)}
        self.permutation = list(self.tiles.keys())

    @override
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None: ...

    @override
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        all_tiles = {dim: self.dims_sizes[dim], **tiles}
        parent_tile_size = self.dims_sizes[dim]
        for name, size in all_tiles.items():
            assert size >= 1, f"unexpected tile size < 1 for axis {dim}"
            assert parent_tile_size >= size, (
                f"unexpected tile size < inner tile for axis {dim}"
            )
            self.tiles[dim][name] = size
            parent_tile_size = size
        self._update_loops()

    @override
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in self.permutation:
            assert axis in permutation, f"missing axis {axis} in interchange"
        for axis in permutation:
            assert axis in self.permutation, f"unexpected axis {axis} in interchange"
        self.permutation = permutation

    @override
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        assert mtype is None or mtype == "local"
        self.write_caches.append(axis)

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        assert mtype is None or mtype == "local"
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        self.read_buffers.append((axis, input_idx, pad))

    @override
    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in axes:
            assert axis in self._working_parallel_dims, f"non parallel axis {axis}"
        self.vectorization = axes

    @override
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in axes:
            assert axis in self._working_parallel_dims, f"non parallel axis {axis}"
        self.parallelization = axes

    @override
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        for axis, unroll in unrolls.items():
            assert unroll > 0, f"unroll < 1 not supported for axis {axis}"
        self.unrolling = unrolls

    @override
    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        self.fused.append((axis, input_idx))

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

    def _get_plain_schedule(self) -> TVMPlainSchedule:
        return TVMPlainSchedule(
            dims=deepcopy(self.dims),
            tiles=self.tiles,
            permutation=self.permutation,
            parallelization=deepcopy(self.parallelization),
            unrolling=self.unrolling,
            vectorization=self.vectorization,
            write_caches=deepcopy(self.write_caches),
            read_buffers=deepcopy(self.read_buffers),
            fused=deepcopy(self.fused),
        )

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
