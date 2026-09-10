#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import re
from typing import Any, cast

import tvm
import tvm.s_tir


def loop_partition_rebased(
    sch: tvm.s_tir.Schedule,
    loop: Any,
    factors: list[int | None],
) -> tuple[tvm.s_tir.Schedule, list[Any]]:
    """Partition a loop and rebase each resulting loop to start at zero.

    TVM's loop_partition preserves the original loop coordinates, while several
    schedule primitives, including split, require a zero loop minimum.  Rebuild
    the partition loops with a zero minimum and add the old minimum to every use
    of their induction variable.

    Rebuilding the IR invalidates schedule RVs, so child blocks are reacquired
    by name and returned with the new schedule.
    """
    working_on = sch.func_working_on
    assert working_on is not None

    partitions = sch.loop_partition(loop, cast(Any, factors))
    children = [sch.get_child_blocks(partition)[0] for partition in partitions]
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
        new_var = tvm.ir.Var(f"{old_var.name}_zero", old_var.ty)
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

    new_mod = tvm.IRModule(
        old_mod.functions,
        attrs=old_mod.attrs,
        global_infos=old_mod.global_infos,
    )
    new_mod.update_func(working_on, old_func.with_body(new_body))

    new_sch = tvm.s_tir.Schedule(new_mod)
    new_sch.work_on(working_on.name_hint)
    new_children = [new_sch.get_sblock(name) for name in child_names]
    return new_sch, new_children


def _depends_on(expr: Any, var: Any, analyzer: Any) -> bool:
    shifted = tvm.tirx.stmt_functor.substitute(expr, {var: var + 1})
    return not analyzer.can_prove_equal(expr, shifted)


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
    that loop. Strides and extents are expressed in elements. The symbol-specific
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
        if not (
            isinstance(body.predicate, tvm.tirx.IntImm) and body.predicate.value != 0
        ):
            raise ValueError("external_at does not currently support partial tiles")

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

        extents = [inner.extent.astype("int64") for inner in inner_loops]
        call = tvm.tirx.call_extern(
            "int32",
            symbol,
            *pointers,
            *extents,
            *projected_strides,
        )
        replaced = True
        return tvm.tirx.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            tvm.tirx.Evaluate(call),
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
