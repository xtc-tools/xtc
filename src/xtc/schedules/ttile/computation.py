#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import math
from typing import List, Dict, Tuple
from enum import Enum

from xtc.schedules.ttile.scheme import stringify_lambda_choice


# 1) Specification of the target computation being optimized


class Computation_spec(Enum):
    CONV = 1
    MATMULT = 2

    def __str__(self):
        if self == Computation_spec.CONV:
            return "conv2d"
        elif self == Computation_spec.MATMULT:
            return "matmul"
        else:
            raise ValueError("Computation_spec::__str__ : unknown computation")

    # TODO: extend this enum into a equational language (in order to manage all possible tensor contraction)


class Computation:
    spec: Computation_spec
    elem_size: int  # in octets

    def __init__(self, spec: Computation_spec, elem_size: int):
        self.spec = spec
        self.elem_size = elem_size
        return

    def __str__(self):
        s = f"{str(self.spec)}(f{self.elem_size * 8})"
        return s


# ======================================================================================

# 2) Base properties of each computations
# TODO: find a way to infer them?


# Get the parallel dimensions of a computation
# Note that it uses the h/w r/s convention for conv2d (because of renaming)
def get_parallel_dims(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.CONV:
        parallel_dims = ["n", "h", "w", "f"]
    elif comp.spec == Computation_spec.MATMULT:
        parallel_dims = ["i", "j"]
    else:
        raise ValueError("get_parallel_dims : unknown computation.")
    return parallel_dims


# Get the reduction dimensions of a computation
# Note that it uses the h/w r/s convention for conv2d (because of renaming)
def get_reduction_dims(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.CONV:
        reduction_dims = ["c", "r", "s"]
    elif comp.spec == Computation_spec.MATMULT:
        reduction_dims = ["k"]
    else:
        raise ValueError("get_reduction_dims : unknown computation.")
    return reduction_dims


# Given a computation, return the list of dimensions of its iteration space
# WARNING: assume that no "," or "*" is present in any dimension name (cf "stringify_lambda_choice")
def get_ldims_computation(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.MATMULT:
        return ["i", "j", "k"]
    elif comp.spec == Computation_spec.CONV:
        return ["n", "h", "w", "f", "r", "s", "c"]
    else:
        raise ValueError("get_ldims_computation :: unknown computation spec.")


# List of non-dimension parameter that still appear in access functions
def get_ldims_stride_computation(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.MATMULT:
        return []
    elif comp.spec == Computation_spec.CONV:
        return ["strx", "stry"]
    else:
        raise ValueError("get_ldims_stride_computation :: unknown computation spec.")


# Given a list of dimensions, build a dictionary associating each dim to 1
def get_default_sizes(comp: Computation) -> dict[str, List[int]]:
    ldims = get_ldims_computation(comp)
    ldims_stride = get_ldims_stride_computation(comp)

    d_lsizes_default = dict()
    for dim in ldims:
        d_lsizes_default[dim] = [1]
    for dim in ldims_stride:
        d_lsizes_default[dim] = [1]

    return d_lsizes_default


# Given a computation, return a map associating each array/tensor
# 	to their list of array accesses (ordered from outer to inner)
# + number of times this access occurs in the computation.
def get_array_accesses(comp: Computation) -> dict[str, Tuple[List[str], int]]:
    if comp.spec == Computation_spec.MATMULT:
        d_arrays_accs = dict()
        d_arrays_accs["A"] = (["i", "k"], 1)
        d_arrays_accs["B"] = (["k", "j"], 1)
        d_arrays_accs["C"] = (["i", "j"], 2)
        return d_arrays_accs
    elif comp.spec == Computation_spec.CONV:
        d_arrays_accs = dict()
        d_arrays_accs["O"] = (["n", "h", "w", "f"], 2)
        d_arrays_accs["I"] = (["n", "h * strx + r", "w * stry + s", "c"], 1)
        d_arrays_accs["K"] = (["r", "s", "c", "f"], 1)
        return d_arrays_accs
    else:
        raise ValueError("get_array_accesses :: unknown computation spec.")


# Given a computation, return the list of dimensions (and the array access lvl)
# that are combined between each other in an affine expression of the array access.
# Note that this could be infered from "get_array_accesses"
def get_combi_dimensions(comp: Computation) -> dict[str, List[Tuple[List[str], int]]]:
    if comp.spec == Computation_spec.MATMULT:
        d_combi_dims: dict[str, List[Tuple[List[str], int]]] = dict()
        d_combi_dims["A"] = []
        d_combi_dims["B"] = []
        d_combi_dims["C"] = []
        return d_combi_dims
    elif comp.spec == Computation_spec.CONV:
        d_combi_dims = dict()
        d_combi_dims["O"] = []
        d_combi_dims["I"] = [(["h", "r"], 1), (["w", "s"], 2)]
        d_combi_dims["K"] = []
        return d_combi_dims
    else:
        raise ValueError("get_combi_dimensions :: unknown computation spec.")


# Array allocation - given an array and a list of dimension size, provide the
# strides of the linearized arrays
def get_array_allocation_contiguous(
    comp: Computation, dprob_sizes: dict[str, int]
) -> dict[str, List[int]]:
    if comp.spec == Computation_spec.MATMULT:
        d_arrays_strides = dict()
        size_k = dprob_sizes["k"]
        size_j = dprob_sizes["j"]
        d_arrays_strides["A"] = [size_k, 1]
        d_arrays_strides["B"] = [size_j, 1]
        d_arrays_strides["C"] = [size_j, 1]
        return d_arrays_strides
    elif comp.spec == Computation_spec.CONV:
        d_arrays_strides = dict()
        size_f = dprob_sizes["f"]
        size_c = dprob_sizes["c"]
        size_w = dprob_sizes["w"]
        size_h = dprob_sizes["h"]
        size_r = dprob_sizes["r"]
        size_s = dprob_sizes["s"]
        strx = dprob_sizes["strx"]
        stry = dprob_sizes["stry"]

        d_arrays_strides["O"] = [size_h * size_w * size_f, size_w * size_f, size_f, 1]
        d_arrays_strides["I"] = [
            ((size_h * strx) + size_r - 1) * ((size_w * stry) + size_s - 1) * size_c,
            ((size_w * stry) + size_s - 1) * size_c,
            size_c,
            1,
        ]
        d_arrays_strides["K"] = [size_s * size_c * size_f, size_c * size_f, size_f, 1]
        return d_arrays_strides
    else:
        raise ValueError("get_array_allocation_contiguous :: unknown computation spec.")


# Provide a default order of contiguous allocation of the arrays
def get_list_array_contiguous_alloc(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.MATMULT:
        return ["C", "A", "B"]
    elif comp.spec == Computation_spec.CONV:
        return ["O", "I", "K"]
    else:
        raise ValueError("get_list_array_contiguous_alloc :: unknown computation spec.")


# Assuming that the arrays used by computation "comp" are allocated in a contiguous manner
# in the order provided by "lcont_arr_order", compute the starting cache set of array "array_name"
# assuming a problem sizes "dprob_sizes" and a number of cache set "num_cache_set".
# We assume that the allocation of the very first array in "lcont_arr_order" is aligned with cache set #0
def get_shift_array(
    comp: Computation,
    lcont_arr_order: List[str],
    array_name: str,
    dprob_sizes: dict[str, int],
    num_cache_set: int,
    cache_line_size: int,
) -> int:
    ldims_stride = get_ldims_stride_computation(comp)

    # 1) Recover the list of arrays before the considered array
    larr_before = []

    assert array_name in lcont_arr_order
    for arr_name in lcont_arr_order:
        if arr_name == array_name:
            break
        larr_before.append(arr_name)

    # DEBUG
    # print(larr_before)

    # For each array before, we compute their contribution to the offset
    d_arrays_accs = get_array_accesses(comp)
    offset = 0
    for arr_before in larr_before:
        # We compute the size of the array "arr_before" from their access function
        laccess = d_arrays_accs[arr_before][0]

        # prob_sizes store the Nd such that 0<=d<Nd.
        # We build d_prob_sizes_inclu (inclusif) storing the (Nd-1)
        d_prob_sizes_inclu = dict()
        for dim, val in dprob_sizes.items():
            if dim in ldims_stride:
                d_prob_sizes_inclu[dim] = val
            else:
                d_prob_sizes_inclu[dim] = val - 1

        # print(f"\t\t{d_prob_sizes_inclu=}")

        # Get the size of output
        size_array = 1
        for expr in laccess:
            size_dim_inclu = eval(expr, d_prob_sizes_inclu)
            size_dim = size_dim_inclu + 1

            # DEBUG
            # print(f"\t\t{expr=} -> {size_dim=}")

            size_array = size_array * size_dim

        # Convert to "cache line" unit, from "word" unit
        size_array = int(size_array / cache_line_size)

        # DEBUG
        # print(f"\t{arr_before=} => {size_array=}")

        # Accumulate
        offset = (offset + size_array) % num_cache_set

    return offset


# Given a computation, return a map associating each array/tensor
# 	to their reuse dimension (in the strict sense). These must be a subset of the list of dim
# returned by "get_ldims_computation"
def get_strict_reuse_dims(comp: Computation) -> dict[str, List[str]]:
    if comp.spec == Computation_spec.MATMULT:
        d_reuse = dict()
        d_reuse["A"] = ["j"]
        d_reuse["B"] = ["i"]
        d_reuse["C"] = ["k"]
        return d_reuse
    elif comp.spec == Computation_spec.CONV:
        d_reuse = dict()
        d_reuse["O"] = ["r", "s", "c"]
        d_reuse["I"] = ["f"]
        d_reuse["K"] = ["n", "h", "w"]
        return d_reuse
    else:
        raise ValueError("get_reuse_dims :: unknown computation spec.")


# Given a computation, return a map associating each array/tensor
# 	to their reuse dimension (accounting small dims). These must be a subset of the list of dim
# returned by "get_ldims_computation"
def get_reuse_dims(comp: Computation) -> dict[str, List[str]]:
    d_reuse = get_strict_reuse_dims(comp)
    if comp.spec == Computation_spec.CONV:
        # Add r/s which are small dimensions
        d_reuse["I"] = ["r", "s", "f"]
    return d_reuse


# ======================================================================================

# 3) Advanced properties of computations

# a) Volume of computation


# Given a problem size, return the number of operations (= size of the iteration space) of a computation
def compute_number_ops(comp: Computation, dprob_sizes: dict[str, int]) -> int:
    if comp.spec == Computation_spec.MATMULT:
        num_ops = 1
        for k, v in dprob_sizes.items():
            num_ops = num_ops * v
        return num_ops
    elif comp.spec == Computation_spec.CONV:
        ldims_stride = get_ldims_stride_computation(comp)
        num_ops = 1
        for k, v in dprob_sizes.items():
            if k not in ldims_stride:
                num_ops = num_ops * v
            else:
                # Stride for convolution: divides its number of iteration since they are used as steps
                assert num_ops % v == 0
                num_ops = num_ops // v
        return num_ops
    else:
        raise ValueError("compute_number_ops :: unknown computation spec.")


# b) Footprint computation


# [Aux function] Compute the footprint of an array, given "dsizes: dim |---> size"
def compute_footprint_nolambda(
    d_arrays_accs: dict[str, Tuple[List[str], int]],
    comp: Computation,
    d_sizes: dict[str, int],
) -> dict[str, int]:
    # Prepare the substitution map (for Python's "eval")
    d_lsizes_int = get_default_sizes(comp)
    ldim_stride = get_ldims_stride_computation(comp)
    # print(d_sizes)
    # print(d_lsizes_int)

    d_sizes_int = dict()
    for dim in d_lsizes_int.keys():
        if dim in d_sizes:
            val_dim = d_sizes[dim]
        else:
            val_dim = d_lsizes_int[dim][0]

        if dim in ldim_stride:
            d_sizes_int[dim] = val_dim
        else:
            d_sizes_int[dim] = val_dim - 1  # max value with "<="

    # DEBUG
    # print(f"d_sizes_int = {d_sizes_int}")

    dfootprint = dict()

    # For all arrays...
    for a, lexpr_sizes in d_arrays_accs.items():
        # Substitution in the sizes per dimension, and global multiplication
        fp_val = 1
        for expr_sizes in lexpr_sizes[0]:
            val_size = eval(expr_sizes, d_sizes_int) + 1

            # DEBUG
            # print(f"[Array {a}] {expr_sizes} + 1 -> {val_size}")

            fp_val *= val_size

        # DEBUG
        # print(f"Array {a} - {fp_val=}")

        # Rectification if comp=CONV:
        #  For array "a=I", when size_r < stride_x , we need to count "size_h * (size_r/stride_x)" on dim h
        # Same for size_s and stride_y
        if (
            (comp.spec == Computation_spec.CONV)
            and (a == "I")
            and ((d_sizes_int["strx"] > 1) or (d_sizes_int["stry"] > 1))
        ):
            # Assume that we are in "hwrs" notation
            assert "x" not in d_sizes_int.keys()

            # If we have holes in the dimension x for array access of I
            strx = d_sizes_int["strx"]
            if ((d_sizes_int["r"] + 1) < strx) and (d_sizes_int["h"] > 1):
                fp_val = int(math.ceil(fp_val / strx))

            # If we have holes in the dimension y for array access of I
            stry = d_sizes_int["stry"]
            if ((d_sizes_int["s"] + 1) < stry) and (d_sizes_int["w"] > 1):
                fp_val = int(math.ceil(fp_val / stry))

        # Tadam!
        dfootprint[a] = fp_val

    return dfootprint


# Compute the footprint of an array, given a tile size
# Input:
# - d_arrays_accs : informations about the array accesses (check "get_array_accesses")
# - d_lsizes : sizes of the considered tile [dim] |---> [list of sizes, depending on the lambda branch]
# Output:
# - ddfootprint: [stringified lambda choice] |--> ( [name_array] |--> footprint value )
#  Note that if there are no lambda, then the first dict will always be something like
# [ "" |--> ( [name_array] |--> footprint value ) ]
#  Use the "stringify_lambda_choice" aux function with the same order of dims (ldims) to get the
# right entry of these maps.
#
# The footprint value is in number of elements (ex: float), so it is not cache line aware
def compute_footprint(
    comp: Computation,
    d_arrays_accs: dict[str, Tuple[List[str], int]],
    d_lsizes: dict[str, List[int]],
):
    # d_lsizes keys are dims, that must match the dims used in the expressions on the right of dict_arrays
    ddfootprint = dict()

    # We detect the lambda encountered at that level, from d_lsizes
    # total_num_branch : unfolded iteration
    d_lambda_num_branch = dict()
    total_num_branch = 1
    for d in d_lsizes:
        if len(d_lsizes[d]) > 1:
            # We have a lambda on d
            num_branch_dim_d = len(d_lsizes[d])
            d_lambda_num_branch[d] = num_branch_dim_d
            total_num_branch = total_num_branch * num_branch_dim_d

    # Reference list of dimensions for this computation
    ldims = get_ldims_computation(comp)

    # For each combination of choice of these lambda:
    for nbranch_unfolded in range(total_num_branch):
        # Recover the corresponding choice in lambda
        temp_nbranch_unfolded = nbranch_unfolded
        d_dim_i_lambda_branch = dict()

        # Assumed to be the same order for all iteration of nbranch_unfolded
        for dim in d_lambda_num_branch:
            num_branch_dim = d_lambda_num_branch[dim]

            # Branch picked: modulo with "num_branch_dim"
            i_lambda_branch = temp_nbranch_unfolded % num_branch_dim

            # Remains (for the future dims)
            temp_nbranch_unfolded = temp_nbranch_unfolded // num_branch_dim

            d_dim_i_lambda_branch[dim] = i_lambda_branch

        # Sanity check
        assert temp_nbranch_unfolded == 0

        # DEBUG (new iterator)
        # print(d_dim_i_lambda_branch)

        # For this choice of branch, recover the values specialized to this branch
        d_sizes_spec_lambdabr = dict()  # [Dim] --> [Value]
        for dim in d_lsizes:
            if dim not in d_dim_i_lambda_branch:
                assert len(d_lsizes[dim]) == 1  # Sanity check
                d_sizes_spec_lambdabr[dim] = d_lsizes[dim][0]
            else:
                d_sizes_spec_lambdabr[dim] = d_lsizes[dim][d_dim_i_lambda_branch[dim]]

        # Now, we use "d_sizes_spec_lambdabr" to compute the footprint
        # using the algorithm that do not consider any lambda
        d_footprint_val = compute_footprint_nolambda(
            d_arrays_accs, comp, d_sizes_spec_lambdabr
        )

        # DEBUG
        # print(f"{d_dim_i_lambda_branch} |--> {d_footprint_val}")

        # Commit
        str_lambda_loc = stringify_lambda_choice(ldims, d_dim_i_lambda_branch)
        ddfootprint[str_lambda_loc] = d_footprint_val

    return ddfootprint
