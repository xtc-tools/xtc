#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from itertools import pairwise
from typing import TextIO, TypeAlias

import numpy as np
from typing_extensions import override

import xtc.backends.tvm as backend
import xtc.itf as itf
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.loop_nest import LoopNest, LoopNestNode
from xtc.schedules.loop_nest_analysis import (
    LoopNestAnalysis,
    analyze_loop_nest,
)
from xtc.schedules.loop_nest_builder import LoopNestBuilder
from xtc.schedules.loop_nest_transform import (
    loop_nest_unroll_as_full,
    loop_nest_vectorize_as_pow2,
)
from xtc.schedules.plain_schedule import PlainNodeSchedule, PlainNodeScheduler

from .TVMOps import TVMOperation

# Actual backend Schedule implementation is a mapping
# from op name to the TVM schedule string
ScheduleImpl: TypeAlias = dict[str, str]


class TVMScheduleEmitter(ABC):
    @abstractmethod
    def emit(self, scheduler: "TVMScheduler"): ...


@dataclass
class _LoopScope:
    """Complete TVM-visible chain and the loops transformable in this scope.

    get_loops stops at the nearest TVM scope block; enclosing loops beyond a
    partition wrapper are intentionally unavailable, not guessed or sliced out.
    Variables are rebound against the whole visible chain after reconstruction.
    Tile prefixes identify levels already materialized by an
    ancestor, independently of the local names used in a branch.
    """

    chain: list[str]
    available: list[str]
    base: dict[str, str]  # Bare dimension names in this scope.
    active: dict[str, str]  # Innermost levels for further tiling or splitting.
    prefixes: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    extents: dict[str, int | None] = field(default_factory=dict)


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

    def _line(self, text: str) -> None:
        print(text, file=self._outf)

    def _bind_loops(self, block: str, scope: _LoopScope) -> None:
        # Partition/rebase reconstructs the entire Schedule and invalidates even
        # ancestor RVs. Never carry a handle across that reconstruction.
        self._line(f"__loops = {self._sch_var}.get_loops({block})")
        self._line(
            f"assert len(__loops) == {len(scope.chain)}, "
            f"'unexpected physical loop chain for {block}'"
        )
        if scope.chain:
            self._line(f"{', '.join(scope.chain)}, = __loops")

    def _dump_schedule_node(
        self,
        block: str,
        node: LoopNestNode,
        path: str,
        scope: _LoopScope,
        analysis: LoopNestAnalysis,
        *,
        inside_parallel: bool = False,
    ) -> None:
        sch = self._sch_var
        if node.is_root:
            # Keep the conventional root spelling; tuple unpacking validates
            # its exact arity just as the explicit child-chain check does.
            self._line(f"{', '.join(scope.chain)}, = {sch}.get_loops({block})")
        else:
            self._bind_loops(block, scope)
        assert len(node.splits) <= 1

        if node.fuse_consumer_at:
            self._line(f"O_F0 = {sch}.get_consumers({block})[0]")
        for inp_idx in dict.fromkeys(value[0] for value in node.pack_at.values()):
            self._line(f'I_R{inp_idx} = {sch}.cache_read({block}, {inp_idx}, "global")')
        if node.buffer_at:
            self._line(f'O_W0 = {sch}.cache_write({block}, 0, "global")')
        for prod_idx in dict.fromkeys(node.fuse_producer_at.values()):
            self._line(f"I_F{prod_idx} = {sch}.get_producers({block})[{prod_idx}]")

        aliases = dict(scope.base)
        tiles = {axis: dict(chain) for axis, chain in node.tiles.items()}
        # A base loop explicitly above the partition is shared. Its descendants
        # determine which tile prefix has to exist before partitioning it away.
        for axis, prefix in analysis.shared_prefixes[path].items():
            assert prefix  # Checked analysis rejects incompatible alternatives.

            # Use the first requesting descendant's names only as readable
            # generated identifiers, never as identities across scopes.
            def names_for(current: LoopNestNode, tile_axis: str) -> list[str]:
                local = current.tiles.get(tile_axis, {})
                if local:
                    return list(local)
                for child in current.children:
                    names = names_for(child, tile_axis)
                    if names:
                        return names
                return []

            names = next(
                (
                    child_names
                    for child in node.children
                    if (child_names := names_for(child, axis))
                ),
                [],
            )
            existing = tiles.setdefault(axis, {})
            existing_sizes = list(existing.values())
            # Descendant requirements may name a suffix of pre-existing tiles
            # or new levels below them. Materialize refinements here while the
            # enclosing level is still accessible, before the partition.
            matched = 0
            if prefix[0] in existing_sizes:
                offset = existing_sizes.index(prefix[0])
                for size, existing_size in zip(prefix, existing_sizes[offset:]):
                    if size != existing_size:
                        break
                    matched += 1
            pending = prefix[matched:]
            assert (
                not pending or not existing_sizes or pending[0] <= existing_sizes[-1]
            ), (
                f"Shared tile prefix for axis {axis} at {path} disagrees with analysis: "
                f"{prefix} versus {existing_sizes}."
            )
            # Names are cosmetic: never overwrite a local declaration or let
            # two shared axes acquire the same local alias.
            used = set(node.interchange) | aliases.keys() | set(scope.chain)
            used.update(name for chain in tiles.values() for name in chain)
            for index, size in enumerate(pending, start=matched):
                name = names[index] if index < len(names) else None
                if name is None or name in used:
                    stem = f"__t_{axis}_{len(existing)}"
                    name = stem
                    suffix = 0
                    while name in used:
                        suffix += 1
                        name = f"{stem}_{suffix}"
                existing[name] = size
                used.add(name)

        for axis in dict.fromkeys([*scope.active, *tiles]):
            requested = tiles.get(axis, {})
            if not requested:
                continue
            previous = scope.prefixes.get(axis, [])
            sizes = list(requested.values())
            # A branch can repeat a shared prefix, start at one of its inner
            # levels, or request a new refinement of the innermost active tile.
            # Local names alone cannot distinguish these three cases.
            start = next(
                (index for index, (size, _) in enumerate(previous) if size == sizes[0]),
                len(previous),
            )
            matched = 0
            for (name, size), (existing_size, var) in zip(
                requested.items(), previous[start:]
            ):
                if size != existing_size:
                    break
                aliases[name] = var
                matched += 1
            residual = list(requested.items())[matched:]
            assert not previous or not residual or residual[0][1] <= previous[-1][0], (
                f"Tile refinement for axis {axis} at {path} disagrees with analysis: "
                f"{sizes} versus {[size for size, _ in previous]}."
            )
            if not residual:
                continue
            target = previous[-1][1] if previous else scope.active.get(axis)
            assert target is not None
            assert target in scope.available
            generated: list[str] = []
            for name, _ in residual:
                var = name
                suffix = 0
                while var in scope.chain or var in generated:
                    suffix += 1
                    var = f"{name}_{suffix}"
                generated.append(var)
                aliases[name] = var
            residual_sizes = [size for _, size in residual]
            if any(outer % inner for outer, inner in pairwise(residual_sizes)):
                # Non-dividing tile spans cannot be represented by factor
                # ratios. Split each level so its local bound is preserved.
                current = target
                for var, size in zip(generated, residual_sizes):
                    self._line(
                        f"{current}, {var}, = {sch}.split({current}, "
                        f"factors=[None, {size}])"
                    )
                    index = scope.chain.index(current) + 1
                    scope.chain[index:index] = [var]
                    index = scope.available.index(current) + 1
                    scope.available[index:index] = [var]
                    current = var
            else:
                factors = [
                    outer // inner for outer, inner in pairwise(residual_sizes)
                ] + [residual_sizes[-1]]
                self._line(
                    f"{', '.join([target] + generated)}, = {sch}.split({target}, "
                    f"factors=[None, {', '.join(map(str, factors))}])"
                )
                index = scope.chain.index(target) + 1
                scope.chain[index:index] = generated
                index = scope.available.index(target) + 1
                scope.available[index:index] = generated
            scope.prefixes[axis] = previous + list(zip(residual_sizes, generated))
            scope.active[axis] = generated[-1]
            scope.extents[axis] = residual_sizes[-1]

        split_axis = next(iter(node.splits), None)
        split_target: str | None = None
        if split_axis is not None:
            split_target = scope.active[split_axis]
            assert split_target in scope.available
        ordered = list(
            dict.fromkeys(
                aliases[name]
                for name in node.interchange
                # Partition branch names in interchange are declarations,
                # not physical loop aliases.
                if name in aliases and aliases[name] in scope.available
            )
        )
        if split_target is not None:
            # The partitioned level itself belongs inside the partition point,
            # even if the local incoming axis also occurs in interchange.
            ordered = [var for var in ordered if var != split_target]
            ordered.append(split_target)
        reorder = list(dict.fromkeys(ordered + scope.available))
        fixed = [var for var in scope.chain if var not in scope.available]
        scope.chain = fixed + reorder
        if reorder:
            self._line(f"{sch}.reorder({', '.join(reorder)})")

        def resolve(name: str) -> str:
            var = aliases[name]
            assert var in scope.chain
            return var

        for axis in node.buffer_at:
            self._line(f"{sch}.reverse_compute_at(O_W0, {resolve(axis)})")
        for axis, (inp_idx, _, pad) in node.pack_at.items():
            self._line(f"{sch}.compute_at(I_R{inp_idx}, {resolve(axis)})")
            dim, factor, offset = tvm_cache_read_factor_offset(self._op, inp_idx, pad)
            if factor:
                self._line(
                    f"{sch}.storage_align(I_R{inp_idx}, 0, axis={dim}, "
                    f"factor={factor}, offset={offset})"
                )
        for axis, prod_idx in node.fuse_producer_at.items():
            self._line(f"{sch}.compute_at(I_F{prod_idx}, {resolve(axis)})")
        for axis in node.fuse_consumer_at:
            self._line(f"{sch}.reverse_compute_at(O_F0, {resolve(axis)})")
        for axis in node.unroll:
            self._line(f"{sch}.unroll({resolve(axis)})")
        for axis in node.vectorize:
            self._line(f"{sch}.vectorize({resolve(axis)})")
        parallel: list[str] = []
        if not inside_parallel and node.parallelize:
            requested_parallel = {resolve(axis) for axis in node.parallelize}
            for var in reorder:
                # A partition consumes its target; fusing it would change
                # what the cuts mean. Its branches have their own loop order.
                if var == split_target:
                    break
                if var in requested_parallel:
                    parallel.append(var)
                elif parallel:
                    break
        if parallel:
            if len(parallel) > 1:
                self._line(f"{parallel[-1]} = {sch}.fuse({', '.join(parallel)})")
                start = scope.chain.index(parallel[0])
                scope.chain[start : start + len(parallel)] = [parallel[-1]]
                scope.available = [
                    var for var in scope.available if var not in parallel[:-1]
                ]
            self._line(f"{sch}.parallel({parallel[-1]})")
            inside_parallel = True
        for axis, symbol in node.external_at.items():
            self._line(
                f"{sch} = externalize_tile_below({sch}, {block}, {resolve(axis)}, {symbol!r})"
            )
        if split_axis is not None:
            assert split_target is not None
            starts = list(node.splits[split_axis].values())
            extent = scope.extents.get(split_axis)
            factors = [end - start for start, end in pairwise(starts)]
            partition_factors = ", ".join(
                [*(str(factor) for factor in factors), "None"]
            )
            self._line(
                f"{sch}, {block}_SPLITS = loop_partition_rebased("
                f"{sch}, {split_target}, [{partition_factors}])"
            )
            position = scope.chain.index(split_target)
            inner = scope.chain[position + 1 :]
            branch_base = {
                axis: var for axis, var in scope.base.items() if var in inner
            }
            branch_base[split_axis] = split_target
            for index, child in enumerate(node.children):
                child_block = f"{block}_{split_axis}_{index}"
                self._line(f"{child_block} = {sch}.get_sblock({block}_SPLITS[{index}])")
                child_active = dict(scope.active)
                child_active[split_axis] = split_target
                child_prefixes = {
                    axis: list(chain)
                    for axis, chain in scope.prefixes.items()
                    if axis != split_axis
                }
                child_extents = dict(scope.extents)
                child_extents[split_axis] = (
                    factors[index]
                    if index < len(factors)
                    else extent - starts[index]
                    if extent is not None
                    else None
                )
                self._dump_schedule_node(
                    child_block,
                    child,
                    f"{path}/{child.root}",
                    _LoopScope(
                        scope.chain[position:],
                        scope.chain[position:],
                        branch_base,
                        child_active,
                        child_prefixes,
                        child_extents,
                    ),
                    analysis,
                    inside_parallel=inside_parallel,
                )

    @override
    def emit(self, scheduler: "TVMScheduler"):
        sched = scheduler.get_loop_nest()
        # Finish both generic rewrites before analysis or output. This keeps
        # validation failures from producing a partial schedule script.
        sched = loop_nest_vectorize_as_pow2(sched)
        sched = loop_nest_unroll_as_full(sched)
        analysis = analyze_loop_nest(sched)
        if sched.root_node is None:
            return
        self._line(f'O = {self._sch_var}.get_sblock("{self._op.name}")')
        dims = sched.abstract_dims
        base = dict(zip(dims, dims))
        self._dump_schedule_node(
            "O",
            sched.root_node,
            sched.root_node.root,
            _LoopScope(list(dims), list(dims), base, dict(base)),
            analysis,
        )
        self._line(
            f"{self._sch_var} = decompose_reduction_initializers({self._sch_var})"
        )


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
