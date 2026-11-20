#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import re
from typing import Any, cast

from typing_extensions import override

import tvm
import tvm.s_tir


def loop_partition_rebased(
    sch: tvm.s_tir.Schedule,
    loop: Any,
    factors: list[int | None],
) -> tuple[tvm.s_tir.Schedule, list[str]]:
    """Partition a loop and rebase each resulting loop to start at zero.

    TVM's loop_partition preserves the original loop coordinates, while several
    schedule primitives, including split, require a zero loop minimum.  Rebuild
    the partition loops with a zero minimum and add the old minimum to every use
    of their induction variable.

    Rebuilding the IR invalidates schedule RVs, so child block names are
    returned for reacquisition on the rebuilt schedule.
    """
    working_on = sch.func_working_on
    assert working_on is not None

    partitions = sch.loop_partition(loop, cast(Any, factors))

    def computation_block(parent: Any) -> Any:
        # Inner-loop partitions introduce an extra scope block. Resolve the
        # computation, not that wrapper (which has no schedulable loop chain).
        children = sch.get_child_blocks(parent)
        if not children:
            return parent
        if len(children) != 1:
            raise ValueError("Partition scope must contain one computation block")
        return computation_block(children[0])

    children = [computation_block(partition) for partition in partitions]
    child_names = [cast(str, cast(Any, sch.get(child)).name_hint) for child in children]
    partition_vars = [
        cast(Any, sch.get(partition)).loop_var for partition in partitions
    ]

    def postorder(node: Any) -> Any:
        if not isinstance(node, tvm.tirx.For):
            return node
        if not any(node.loop_var.same_as(var) for var in partition_vars):
            return node
        if isinstance(node.min, tvm.tirx.IntImm) and node.min.value == 0:
            return node

        old_var = node.loop_var
        new_var = tvm.ir.Var(old_var.name, old_var.ty)
        body = tvm.tirx.stmt_functor.substitute(
            node.body,
            {old_var: new_var + node.min},
        )
        return tvm.tirx.For(
            new_var,
            cast(Any, 0),
            node.extent,
            node.kind,
            body,
            node.thread_binding,
            node.annotations,
            node.step,
            node.span,
        )

    old_mod = cast(Any, sch.mod)
    old_func = old_mod[working_on]
    new_body = tvm.tirx.stmt_functor.ir_transform(
        old_func.body,
        None,
        postorder,
        ["tirx.For"],
    )

    # loop_partition duplicates the partition body, including SBlock definitions.
    # The cloned blocks can consequently share the same iter-var Vars, which makes
    # the resulting TIR invalid SSA. Give every block realization fresh iter vars
    # and rewrite references local to that block.
    def uniquify_block_iter_vars(node: Any) -> Any:
        if not isinstance(node, tvm.tirx.SBlockRealize):
            return node

        block = node.block
        if not block.iter_vars:
            return node
        substitutions = {
            iter_var.var: tvm.ir.Var(iter_var.var.name, iter_var.var.ty)
            for iter_var in block.iter_vars
        }
        iter_vars = [
            tvm.tirx.IterVar(
                tvm.ir.Range(
                    tvm.tirx.stmt_functor.substitute(iter_var.dom.min, substitutions),
                    tvm.tirx.stmt_functor.substitute(
                        iter_var.dom.extent, substitutions
                    ),
                ),
                substitutions[iter_var.var],
                iter_var.iter_type,
                iter_var.thread_tag,
            )
            for iter_var in block.iter_vars
        ]

        def substitute_region(region: Any) -> Any:
            return tvm.tirx.BufferRegion(
                region.buffer,
                [
                    tvm.ir.Range(
                        tvm.tirx.stmt_functor.substitute(item.min, substitutions),
                        tvm.tirx.stmt_functor.substitute(
                            item.extent,
                            substitutions,
                        ),
                    )
                    for item in region.region
                ],
            )

        new_block = tvm.tirx.SBlock(
            iter_vars,
            [substitute_region(region) for region in block.reads],
            [substitute_region(region) for region in block.writes],
            block.name_hint,
            tvm.tirx.stmt_functor.substitute(block.body, substitutions),
            (
                tvm.tirx.stmt_functor.substitute(block.init, substitutions)
                if block.init is not None
                else None
            ),
            block.alloc_buffers,
            block.match_buffers,
            block.annotations,
            block.span,
        )
        return tvm.tirx.SBlockRealize(
            node.iter_values,
            node.predicate,
            new_block,
            node.span,
        )

    new_body = tvm.tirx.stmt_functor.ir_transform(
        new_body,
        None,
        uniquify_block_iter_vars,
        ["tirx.SBlockRealize"],
    )

    new_mod = tvm.IRModule(
        old_mod.functions,
        attrs=old_mod.attrs,
        global_infos=old_mod.global_infos,
    )
    new_mod.update_func(working_on, old_func.with_body(new_body))

    new_sch = tvm.s_tir.Schedule(new_mod)
    new_sch.work_on(working_on.name_hint)
    return new_sch, child_names


def _depends_on(expr: Any, var: Any, analyzer: Any) -> bool:
    shifted = tvm.tirx.stmt_functor.substitute(expr, {var: var + 1})
    return not analyzer.can_prove_equal(expr, shifted)


def _conjuncts(expr: Any) -> list[Any]:
    if isinstance(expr, tvm.tirx.And):
        and_expr = cast(Any, expr)
        return _conjuncts(and_expr.a) + _conjuncts(and_expr.b)
    return [expr]


def _compact_strides(buffer: Any) -> list[Any]:
    if buffer.strides:
        return list(buffer.strides)
    stride = tvm.tirx.IntImm("int32", 1)
    strides: list[Any] = []
    for extent in reversed(buffer.shape):
        strides.append(stride)
        stride = stride * extent
    return list(reversed(strides))


def _buffer_offset(
    buffer: Any,
    block_realize: Any,
    block_region: Any,
) -> Any:
    block = block_realize.block
    block_bindings = {
        iter_var.var: value
        for iter_var, value in zip(block.iter_vars, block_realize.iter_values)
    }
    indices = [region.min for region in block_region.region]
    indices = [
        tvm.tirx.stmt_functor.substitute(index, block_bindings) for index in indices
    ]
    strides = _compact_strides(buffer)
    offset = tvm.tirx.IntImm("int32", 0)
    for index, stride in zip(indices, strides):
        offset = offset + index * stride
    return offset


def _decompose_partitioned_reductions(
    sch: tvm.s_tir.Schedule,
    block_names: list[str],
) -> tvm.s_tir.Schedule:
    """Decompose reductions hidden below loop-partition scope blocks."""
    working_on = sch.func_working_on
    assert working_on is not None
    old_mod = cast(Any, sch.mod)
    old_func = old_mod[working_on]
    target_names = set(block_names)
    analyzer = tvm.arith.Analyzer()
    block_infos: dict[str, tuple[Any, list[Any], list[Any]]] = {}

    class BlockCollector(tvm.tirx.stmt_functor.StmtVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.loops: list[Any] = []

        @override
        def visit_for_(self, op: Any) -> None:
            self.loops.append(op)
            super().visit_for_(op)
            self.loops.pop()

        @override
        def visit_block_realize_(self, op: Any) -> None:
            if op.block.name_hint in target_names and op.block.init is not None:
                reduction_values = [
                    value
                    for value, iter_var in zip(
                        op.iter_values,
                        op.block.iter_vars,
                    )
                    if iter_var.iter_type == 2
                ]
                block_infos[op.block.name_hint] = (
                    op,
                    list(self.loops),
                    reduction_values,
                )
            super().visit_block_realize_(op)

    collector = BlockCollector()
    collector.visit_stmt(old_func.body)

    init_by_outer_loop: list[tuple[Any, list[Any]]] = []
    target_blocks: set[str] = set()

    def group_for(loop: Any) -> list[Any]:
        for grouped_loop, inits in init_by_outer_loop:
            if grouped_loop.same_as(loop):
                return inits
        new_inits: list[Any] = []
        init_by_outer_loop.append((loop, new_inits))
        return new_inits

    def substitute_region(region: Any, substitutions: dict[Any, Any]) -> Any:
        return tvm.tirx.BufferRegion(
            region.buffer,
            [
                tvm.ir.Range(
                    tvm.tirx.stmt_functor.substitute(item.min, substitutions),
                    tvm.tirx.stmt_functor.substitute(
                        item.min + item.extent,
                        substitutions,
                    ),
                )
                for item in region.region
            ],
        )

    for block_name, (realize, loops, reduction_values) in block_infos.items():
        reduction_loops = [
            loop
            for loop in loops
            if any(
                _depends_on(value, loop.loop_var, analyzer)
                for value in reduction_values
            )
        ]
        if not reduction_loops:
            continue
        outer_reduction = reduction_loops[0]
        outer_index = next(
            idx for idx, loop in enumerate(loops) if loop.same_as(outer_reduction)
        )
        retained_loops = [
            loop
            for loop in loops[outer_index + 1 :]
            if not any(loop.same_as(candidate) for candidate in reduction_loops)
        ]
        substitutions = {loop.loop_var: loop.min for loop in reduction_loops}
        for loop in retained_loops:
            substitutions[loop.loop_var] = tvm.ir.Var(
                f"{loop.loop_var.name}_init",
                loop.loop_var.ty,
            )

        spatial_iter_vars = []
        spatial_iter_values = []
        for value, iter_var in zip(
            realize.iter_values,
            realize.block.iter_vars,
        ):
            if iter_var.iter_type == 2:
                continue
            new_var = tvm.ir.Var(f"{iter_var.var.name}_init", iter_var.var.ty)
            substitutions[iter_var.var] = new_var
            spatial_iter_vars.append(
                tvm.tirx.IterVar(
                    iter_var.dom,
                    new_var,
                    iter_var.iter_type,
                    iter_var.thread_tag,
                )
            )
            spatial_iter_values.append(
                tvm.tirx.stmt_functor.substitute(value, substitutions)
            )

        init_block = tvm.tirx.SBlock(
            spatial_iter_vars,
            [],
            [substitute_region(write, substitutions) for write in realize.block.writes],
            f"{block_name}_init",
            tvm.tirx.stmt_functor.substitute(
                realize.block.init,
                substitutions,
            ),
            None,
            realize.block.alloc_buffers,
            realize.block.match_buffers,
            realize.block.annotations,
            realize.block.span,
        )
        init_stmt: Any = tvm.tirx.SBlockRealize(
            spatial_iter_values,
            tvm.tirx.stmt_functor.substitute(realize.predicate, substitutions),
            init_block,
            realize.span,
        )
        for loop in reversed(retained_loops):
            init_stmt = tvm.tirx.For(
                substitutions[loop.loop_var],
                tvm.tirx.stmt_functor.substitute(loop.min, substitutions),
                tvm.tirx.stmt_functor.substitute(loop.extent, substitutions),
                loop.kind,
                init_stmt,
                loop.thread_binding,
                loop.annotations,
                (
                    tvm.tirx.stmt_functor.substitute(loop.step, substitutions)
                    if loop.step is not None
                    else None
                ),
                loop.span,
            )
        group_for(outer_reduction).append(init_stmt)
        target_blocks.add(block_name)

    if not target_blocks:
        return sch

    def postorder(node: Any) -> Any:
        if isinstance(node, tvm.tirx.SBlockRealize):
            block = node.block
            if block.name_hint not in target_blocks or block.init is None:
                return node
            reads = [*block.reads, *block.writes]
            update_block = tvm.tirx.SBlock(
                list(block.iter_vars),
                reads,
                list(block.writes),
                f"{block.name_hint}_update",
                block.body,
                None,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
                block.span,
            )
            return tvm.tirx.SBlockRealize(
                list(node.iter_values),
                node.predicate,
                update_block,
                node.span,
            )
        if isinstance(node, tvm.tirx.For):
            for outer_loop, init_stmts in init_by_outer_loop:
                if node.loop_var.same_as(outer_loop.loop_var):
                    return tvm.tirx.SeqStmt([*init_stmts, node])
        return node

    new_body = tvm.tirx.stmt_functor.ir_transform(
        old_func.body,
        None,
        postorder,
        ["tirx.SBlockRealize", "tirx.For"],
    )
    new_mod = tvm.IRModule(
        old_mod.functions,
        attrs=old_mod.attrs,
        global_infos=old_mod.global_infos,
    )
    new_mod.update_func(working_on, old_func.with_body(new_body))
    new_sch = tvm.s_tir.Schedule(new_mod)
    new_sch.work_on(working_on.name_hint)
    return new_sch


def decompose_reduction_initializers(
    sch: tvm.s_tir.Schedule,
) -> tvm.s_tir.Schedule:
    """Separate every reduction initializer from its update loop nest.

    The initializer is inserted immediately before the first loop contributing
    to a reduction block variable.  Applying this after all regular schedule
    primitives preserves their annotations on both resulting loop nests and
    also handles reductions writing to a cache-write buffer. Externalized
    reductions have already had their initializer decomposed, so they are
    naturally ignored because they no longer have an init statement.
    """
    working_on = sch.func_working_on
    assert working_on is not None

    func = cast(Any, sch.mod)[working_on]
    reduction_blocks: list[str] = []

    def collect(node: Any) -> None:
        if not isinstance(node, tvm.tirx.SBlockRealize):
            return
        block = node.block
        if block.init is not None and any(
            iter_var.iter_type == 2 for iter_var in block.iter_vars
        ):
            reduction_blocks.append(cast(str, block.name_hint))

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)

    analyzer = tvm.arith.Analyzer()
    partitioned_blocks: list[str] = []

    for block_name in dict.fromkeys(reduction_blocks):
        block = sch.get_sblock(block_name)
        block_stmt = cast(Any, sch.get(block))
        block_realizes: list[Any] = []

        def collect_realize(node: Any) -> None:
            if (
                isinstance(node, tvm.tirx.SBlockRealize)
                and node.block.name_hint == block_stmt.name_hint
            ):
                block_realizes.append(node)

        current_func = cast(Any, sch.mod)[working_on]
        tvm.tirx.stmt_functor.post_order_visit(
            current_func.body,
            collect_realize,
        )
        if len(block_realizes) != 1:
            raise ValueError(
                f"Could not uniquely locate reduction block {block_name!r}"
            )

        block_realize = block_realizes[0]
        reduction_values = [
            value
            for value, iter_var in zip(
                block_realize.iter_values,
                block_stmt.iter_vars,
            )
            if iter_var.iter_type == 2
        ]
        loops = list(sch.get_loops(block))
        reduction_loop = next(
            (
                loop
                for loop in loops
                if any(
                    _depends_on(
                        value,
                        cast(Any, sch.get(loop)).loop_var,
                        analyzer,
                    )
                    for value in reduction_values
                )
            ),
            None,
        )

        # Re-read ancestry after each schedule primitive, which may rebuild
        # the IR and replace loop nodes.
        class ReductionAncestryCollector(tvm.tirx.stmt_functor.StmtVisitor):
            def __init__(self, target_name: str) -> None:
                super().__init__()
                self.target_name = target_name
                self.loops: list[Any] = []
                self.ancestry: list[Any] = []

            @override
            def visit_for_(self, op: Any) -> None:
                self.loops.append(op)
                super().visit_for_(op)
                self.loops.pop()

            @override
            def visit_block_realize_(self, op: Any) -> None:
                if op.block.name_hint == self.target_name:
                    self.ancestry.extend(self.loops)
                super().visit_block_realize_(op)

        ancestry_collector = ReductionAncestryCollector(block_name)
        ancestry_collector.visit_stmt(current_func.body)
        full_reduction_loops = [
            loop
            for loop in ancestry_collector.ancestry
            if any(
                _depends_on(value, loop.loop_var, analyzer)
                for value in reduction_values
            )
        ]
        outer_reduction_loop = full_reduction_loops[0] if full_reduction_loops else None

        # Partition scope blocks can hide an outer reduction loop from
        # get_loops(). Decomposing at the first visible reduction loop would
        # then initialize once per outer reduction iteration, so use the
        # structural fallback whenever the true outermost loop is not visible.
        if (
            reduction_loop is None
            or outer_reduction_loop is None
            or not any(
                cast(Any, sch.get(loop)).loop_var.same_as(outer_reduction_loop.loop_var)
                for loop in loops
            )
        ):
            partitioned_blocks.append(block_name)
        else:
            sch.decompose_reduction(block, reduction_loop)

    return _decompose_partitioned_reductions(sch, partitioned_blocks)


def externalize_tile_below(
    sch: tvm.s_tir.Schedule,
    block: Any,
    axis: Any,
    symbol: str,
) -> tvm.s_tir.Schedule:
    """Replace the loop tile below ``axis`` by a C-ABI external call.

    The axis itself is preserved and the function is called once per axis
    iteration. Reduction initialization is moved before the outermost loop
    needed to keep it outside the externalized subtree.

    Arguments use this deterministic order::

        int32_t symbol(output_ptr, input_ptrs...,
                       int64_t inner_extents...,
                       int64_t output_projected_strides...,
                       int64_t input_projected_strides...);

    There is currently one output. Inputs remain in PrimFunc parameter order.
    Buffer pointers denote the access position obtained by setting every loop
    below ``axis`` to zero. Each buffer receives one projected stride per inner
    loop, in loop order; a zero stride means that the buffer is invariant along
    that loop. Extents are clipped to the valid domain of the current partial
    tile. Predicate terms involving preserved loops guard the external call.
    Strides and extents are expressed in elements. The symbol-specific
    implementation defines the concrete pointer types and knows the number of
    inputs and inner dimensions. Its return value is ignored.
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) is None:
        raise ValueError(f"Invalid external C symbol: {symbol!r}")

    working_on = sch.func_working_on
    assert working_on is not None

    old_func = cast(Any, sch.mod)[working_on]
    block_stmt = cast(Any, sch.get(block))
    block_realizes: list[Any] = []

    def collect_block_realize(node: Any) -> None:
        if (
            isinstance(node, tvm.tirx.SBlockRealize)
            and node.block.name_hint == block_stmt.name_hint
        ):
            block_realizes.append(node)

    tvm.tirx.stmt_functor.post_order_visit(old_func.body, collect_block_realize)
    if len(block_realizes) != 1:
        raise ValueError("Could not uniquely locate the external block")

    block_realize = block_realizes[0]
    reduction_values = [
        value
        for value, iter_var in zip(block_realize.iter_values, block_stmt.iter_vars)
        if iter_var.iter_type == 2
    ]
    loops = list(sch.get_loops(block))
    loop_vars = [cast(Any, sch.get(loop)).loop_var for loop in loops]
    axis_var = cast(Any, sch.get(axis)).loop_var
    axis_idx = next(
        (idx for idx, var in enumerate(loop_vars) if var.same_as(axis_var)), None
    )
    if axis_idx is None:
        raise ValueError("external_at axis is not an ancestor of the block")

    analyzer = tvm.arith.Analyzer()
    reduction_loop_indices = [
        idx
        for idx, loop_var in enumerate(loop_vars)
        if any(_depends_on(value, loop_var, analyzer) for value in reduction_values)
    ]
    if reduction_loop_indices:
        # Place initialization before both the external axis and every reduction
        # loop. This also handles strip-mined and multi-dimensional reductions.
        decompose_idx = min(axis_idx, reduction_loop_indices[0])
        sch.decompose_reduction(block, loops[decompose_idx])

    old_mod = cast(Any, sch.mod)
    old_func = old_mod[working_on]
    params = list(old_func.params)
    replaced = False

    def postorder(node: Any) -> Any:
        nonlocal replaced
        if not isinstance(node, tvm.tirx.For) or not node.loop_var.same_as(axis_var):
            return node

        inner_loops: list[Any] = []
        body = node.body
        while isinstance(body, tvm.tirx.For):
            inner_loops.append(body)
            body = body.body

        if not inner_loops or not isinstance(body, tvm.tirx.SBlockRealize):
            raise ValueError(
                "external_at requires a single perfectly nested loop tile below "
                "the selected axis"
            )
        zero_substitutions = {
            inner.loop_var: tvm.tirx.IntImm("int32", 0) for inner in inner_loops
        }
        reads = list(body.block.reads)
        writes = list(body.block.writes)

        def regions_for(buffer: Any, regions: list[Any]) -> list[Any]:
            return [region for region in regions if region.buffer.name == buffer.name]

        output_params = [param for param in params if regions_for(param, writes)]
        if len(output_params) != 1:
            raise ValueError("external_at currently requires exactly one output")
        output = output_params[0]
        inputs = [
            param
            for param in params
            if regions_for(param, reads) and param.name != output.name
        ]
        buffers = [output] + inputs

        pointers: list[Any] = []
        projected_strides: list[Any] = []
        for buffer_idx, buffer in enumerate(buffers):
            regions = regions_for(buffer, writes if buffer_idx == 0 else reads)
            if len(regions) != 1:
                raise ValueError(
                    "external_at requires one access region per input and output"
                )
            access_offset = _buffer_offset(buffer, body, regions[0])
            origin = tvm.tirx.stmt_functor.substitute(access_offset, zero_substitutions)
            pointers.append(
                buffer.access_ptr(3 if buffer_idx == 0 else 1, offset=origin)
            )
            for inner in inner_loops:
                shifted_offset = tvm.tirx.stmt_functor.substitute(
                    access_offset, {inner.loop_var: inner.loop_var + 1}
                )
                projected_stride = analyzer.simplify(shifted_offset - access_offset)
                if any(
                    _depends_on(projected_stride, other.loop_var, analyzer)
                    for other in inner_loops
                ):
                    raise ValueError(
                        "external_at requires constant projected strides over "
                        "the externalized tile"
                    )
                projected_strides.append(cast(Any, projected_stride).astype("int64"))

        extents: list[Any] = []
        for inner in inner_loops:
            valid_extent = inner.extent
            if isinstance(body.predicate, tvm.tirx.IntImm) or not _depends_on(
                body.predicate, inner.loop_var, analyzer
            ):
                extents.append(cast(Any, valid_extent).astype("int64"))
                continue
            for iter_var, iter_value in zip(body.block.iter_vars, body.iter_values):
                if not _depends_on(iter_value, inner.loop_var, analyzer):
                    continue
                shifted_value = tvm.tirx.stmt_functor.substitute(
                    iter_value, {inner.loop_var: inner.loop_var + 1}
                )
                step = analyzer.simplify(shifted_value - iter_value)
                if not isinstance(step, tvm.tirx.IntImm) or step.value <= 0:
                    raise ValueError(
                        "external_at requires positive constant block bindings "
                        "for partial tiles"
                    )
                origin = tvm.tirx.stmt_functor.substitute(
                    iter_value, zero_substitutions
                )
                domain_end = cast(Any, iter_var.dom.min) + iter_var.dom.extent
                remaining = tvm.tirx.ceildiv(domain_end - origin, step)
                valid_extent = tvm.tirx.Min(
                    valid_extent,
                    tvm.tirx.Max(tvm.tirx.IntImm("int32", 0), remaining),
                )
            extents.append(cast(Any, analyzer.simplify(valid_extent)).astype("int64"))

        call = tvm.tirx.call_extern(
            "int32",
            symbol,
            *pointers,
            *extents,
            *projected_strides,
        )
        call_stmt = tvm.tirx.Evaluate(call)
        residual_predicates = [
            predicate
            for predicate in _conjuncts(body.predicate)
            if not any(
                _depends_on(predicate, inner.loop_var, analyzer)
                for inner in inner_loops
            )
        ]
        if residual_predicates:
            residual_predicate = analyzer.simplify(tvm.tirx.all(*residual_predicates))
            if not (
                isinstance(residual_predicate, tvm.tirx.IntImm)
                and residual_predicate.value != 0
            ):
                call_stmt = tvm.tirx.IfThenElse(
                    residual_predicate,
                    call_stmt,
                    None,
                )
        replaced = True
        return tvm.tirx.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            call_stmt,
            node.thread_binding,
            node.annotations,
            node.step,
            node.span,
        )

    new_body = tvm.tirx.stmt_functor.ir_transform(
        old_func.body,
        None,
        postorder,
        ["tirx.For"],
    )
    if not replaced:
        raise ValueError("Could not locate the external_at axis in scheduled TIR")

    new_mod = tvm.IRModule(
        old_mod.functions,
        attrs=old_mod.attrs,
        global_infos=old_mod.global_infos,
    )
    new_mod.update_func(working_on, old_func.with_body(new_body))
    new_sch = tvm.s_tir.Schedule(new_mod)
    new_sch.work_on(working_on.name_hint)
    return new_sch
