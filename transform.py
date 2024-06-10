#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import copy

sym_count = 0
var_count = 0


def get_new_var():
    global var_count
    new_var = f"%{var_count}"
    var_count += 1
    return new_var


def get_new_seq_name():
    global sym_count
    new_seq_name = f"@seq{sym_count}"
    sym_count += 1
    return new_seq_name


def get_seq_signature(
    input_consumed=False,
    has_output=False,
    sym_name=None,
):
    sym_name = sym_name if sym_name else get_new_seq_name()
    input_var = get_new_var()

    input_attribute = ""
    if input_consumed:
        input_attribute = "{transform.consumed}"
    else:
        input_attribute = "{transform.readonly}"

    tail = " -> (!transform.any_op)" if has_output else ""

    seq_sig = (
        f"transform.named_sequence {sym_name} ({input_var}: "
        + f"!transform.any_op {input_attribute}) {tail}"
    )

    return sym_name, input_var, seq_sig


def get_terminator(namespace="transform", result=None):
    tail = f" {result} : !transform.any_op" if result else ""
    return f"{namespace}.yield{tail}"


def get_vectorize_children(op):
    vectorized = get_new_var()
    vectorize = (
        f"{vectorized} = transform.structured.vectorize_children_and_apply_patterns "
        f"{op} : (!transform.any_op) -> !transform.any_op"
    )
    return vectorized, vectorize


def get_registered_pass(op, reg):
    nvar = get_new_var()
    return nvar, (
        nvar
        + " = transform.apply_registered_pass "
        + '"'
        + reg
        + '" to '
        + op
        + ": (!transform.any_op) -> !transform.any_op"
    )


def get_vectorize(op):
    return f"transform.structured.vectorize {op} : !transform.any_op"


def get_scalarize(op):
    scalar = get_new_var()
    scalarization = (
        f"{scalar} = transform.structured.scalarize {op}"
        + ": (!transform.any_op) -> !transform.any_op"
    )
    return scalar, scalarization


def get_parent(op):
    parent = get_new_var()
    parenting = (
        f"{parent} = transform.get_parent_op {op} "
        + "{isolated_from_above} : (!transform.any_op) -> !transform.any_op"
    )
    return parent, parenting


def get_unroll(loop, factor):
    return (
        f"transform.loop.unroll {loop}"
        + "{ factor = "
        + str(factor)
        + "} : !transform.any_op"
    )


def produce_tiling_instr(current_state, dims_vector, parallel=False):
    new_state = get_new_var()
    new_loop = get_new_var()

    str_dims = "[" + ",".join([str(d) for d in dims_vector]) + "]"

    opname = "tile_using_forall" if parallel else "tile_using_for"

    attribute = "tile_sizes" if parallel else ""

    return_type = "(!transform.any_op"
    num_loops = 0
    for d in dims_vector:
        if d > 0:
            return_type += ",!transform.any_op"
            num_loops += 1
    return_type += ")"

    str_tile = (
        f"{new_state},{new_loop}:{num_loops} = transform.structured.{opname}"
        + f"{current_state} {attribute} {str_dims} :"
        + f"(!transform.any_op) -> {return_type}"
    )
    return new_state, new_loop, str_tile


def annotate(op, annotation):
    return "transform.annotate " + op + '"' + annotation + '"' + ": !transform.any_op"


def match_by_attribute(op, attr):
    nvar = get_new_var()
    return nvar, (
        nvar
        + " = transform.structured.match "
        + 'attributes{"'
        + attr
        + '"} in '
        + op
        + ": (!transform.any_op) -> !transform.any_op"
    )


def match_by_op_name(op, name):
    nvar = get_new_var()
    return nvar, (
        nvar
        + " = transform.structured.match "
        + 'ops{["'
        + name
        + '"]} in '
        + op
        + ": (!transform.any_op) -> !transform.any_op"
    )


def apply_patterns(hl_var, patterns):
    return (
        [
            f"transform.apply_patterns to {hl_var}",
            "{",
        ]
        + patterns
        + [
            "} {apply_cse} : !transform.any_op",
        ]
    )


def vector_pre_hoist_apply_patterns(hl_var):
    hl_patterns0 = apply_patterns(
        hl_var,
        [
            "transform.apply_patterns.memref.fold_memref_alias_ops",
            "transform.apply_patterns.canonicalization",
        ],
    )
    return hl_patterns0


def vector_lower_outerproduct_patterns(hl_var):
    hl_patterns0 = apply_patterns(
        hl_var,
        [
            "transform.apply_patterns.vector.lower_outerproduct",
            'transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"',
            "transform.apply_patterns.canonicalization",
        ],
    )
    return hl_patterns0


def vector_hoist(hl_var):
    nvar = get_new_var()
    hoist = (
        f"{nvar} = transform.structured.hoist_redundant_vector_transfers "
        + f"{hl_var} : (!transform.any_op) -> !transform.any_op"
    )
    return nvar, hoist


def tiling_apply_patterns(hl_var):
    hl_patterns0 = apply_patterns(
        hl_var,
        [
            "transform.apply_patterns.canonicalization",
            "transform.apply_patterns.linalg.tiling_canonicalization",
            "transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes",
        ],
    )
    return hl_patterns0
