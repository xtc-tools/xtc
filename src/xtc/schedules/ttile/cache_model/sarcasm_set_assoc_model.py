#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from typing import List, Tuple, Optional, Dict
import math

from xtc.schedules.ttile.scheme import Atom, AtomType
from xtc.schedules.ttile.scheme import get_sizes_scheme, get_unlambda_sizes_scheme
from xtc.schedules.ttile.scheme import (
    stringify_lambda_choice,
    get_list_dims_str_lambda_loc,
)
from xtc.schedules.ttile.scheme import (
    recover_all_branchid_from_stringified,
    recover_branchid_from_stringified,
)
from xtc.schedules.ttile.scheme import (
    remove_dim_from_stringified,
    get_max_num_lambda_branch,
)

from xtc.schedules.ttile.computation import (
    Computation,
    get_array_accesses,
    get_array_allocation_contiguous,
    get_shift_array,
)
from xtc.schedules.ttile.computation import (
    get_ldims_computation,
    get_ldims_stride_computation,
    get_combi_dimensions,
    get_default_sizes,
)
from xtc.schedules.ttile.computation import compute_footprint_nolambda

from xtc.schedules.ttile.cache_model.full_assoc_model import (
    total_fieldname_fp,
    ReuseLoopStrat,
    full_assoc_model_with_fp,
)


# This Python file contains the SARCASM set-associativity model,
# Set-Associative Rotating Cache Analytical/Simulating Model


# ====================================================================

# 1) Preprocessing - preparing the information about the problem

StrideInfos = List[Tuple[int, str]]
IterTileInfos = List[dict[str, int]]


# [Aux function] Build the set of information about the access function,
#    in particular what are their stride (in number of cache lines)
#    If we have an access function such as "x+w", we will have two entries of same stride in the output.
#   WARNING: all indices must happen only once on the access function of the considered computation.
#   WARNING: the problem sizes must be such that the cache lines divides the stride of the array allocation.
#
# Inputs:
#  - comp: the computation being considered
#  - prob_sizes : size of the problem. Mapping [dim_name] |--> [int]
#  - cache_line_size : size of a cache line (int)
# Output:
#  - maccess_func_dim_name_coeff : [arr_name] |--> list of ( stride [granularity in cache_line], dim_name )
#    The list is inner to outer, and indicates how many cache line do we shift if I increment
#      the index of a dimension by 1? (except for the last dimension)
def build_maccess_func_coeff(
    comp: Computation, prob_sizes: dict[str, int], cache_line_size: int
) -> dict[str, StrideInfos]:
    # STEP 1: Get the stride for each array (in word)
    d_lstride_elem = get_array_allocation_contiguous(comp, prob_sizes)

    # DEBUG
    # print(f"{d_lstride_elem=}  (in word)")

    # STEP 2: Convert these stride in cache lines + check that alignment with cache line is ok
    d_lstride = dict()
    for arr_name, lstride in d_lstride_elem.items():
        for k in range(0, len(lstride) - 1):
            stride_converted = lstride[k] / cache_line_size
            if not float(stride_converted).is_integer():
                print(f"{stride_converted=}")
                print(
                    "Broken assumption on the inputs - stride after innermost dim are not multiple of cache line size"
                )
                exit(1)
            lstride[k] = int(stride_converted)
        d_lstride[arr_name] = lstride

    # DEBUG
    # print(f"{d_lstride=}  (in Cache Lines)")

    # STEP 3: access function management: we need to extract the affine function of each array accesses
    # + to compose them with "d_lstride"
    d_arrays_accs = get_array_accesses(comp)
    ldim_stride = get_ldims_stride_computation(comp)
    d_ll_dim_coeff_access = dict()
    for arr_name, laccess_info in d_arrays_accs.items():
        laccess = laccess_info[0]

        # DEBUG
        # print(laccess)

        # For each dim of the access function
        ll_dim_coeff_access = []
        for k in range(len(laccess)):
            str_access = laccess[k]

            # DEBUG
            # print(str_access)

            # Manage the non-dimension parameters
            if len(str_access) > 1:  # To save computation
                for d in ldim_stride:
                    str_access = str_access.replace(d, str(prob_sizes[d]))

            # DEBUG
            # print(str_access)

            # Parse "str_access" as an affine function (with "+" and "*" / example: "h * 2 + r")
            l_dim_coeff_access = []
            lterms = str_access.split("+")
            for term in lterms:
                term = term.strip()

                if "*" in term:
                    latom = term.split("*")
                    assert len(latom) == 2

                    latom_striped = [atom.strip() for atom in latom]

                    # One of them are a integer, the other a dim
                    if latom_striped[0].isdecimal():
                        coeff = int(latom_striped[0])
                        str_dim = latom_striped[1]
                    else:
                        assert latom_striped[1].isdecimal()
                        coeff = int(latom_striped[1])
                        str_dim = latom_striped[0]
                else:
                    coeff = 1
                    str_dim = term

                assert str_dim in prob_sizes

                # Commit this part of the expression
                l_dim_coeff_access.append((str_dim, coeff))

            # Commit
            ll_dim_coeff_access.append(l_dim_coeff_access)

        # DEBUG
        # print(f"{laccess} --> {ll_dim_coeff_access=}")

        d_ll_dim_coeff_access[arr_name] = ll_dim_coeff_access

    # DEBUG
    # print(f"{d_ll_dim_coeff_access=}")

    # Step 4: Compose d_lstride with d_ll_dim_coeff_access
    maccess_func_dim_name_coeff = dict()

    for arr_name in d_lstride:
        # Recover the informations
        lstride = d_lstride[arr_name]
        ll_dim_coeff_access = d_ll_dim_coeff_access[arr_name]

        # The access function output must match the array allocation dimensionality
        assert len(lstride) == len(ll_dim_coeff_access)

        llstride_dimname = []
        for k in range(len(lstride)):
            stride_k = lstride[k]
            l_dim_coeff_access_k = ll_dim_coeff_access[k]

            for dim_name, coeff in l_dim_coeff_access_k:
                full_stride = coeff * stride_k
                llstride_dimname.append((full_stride, dim_name))

        # Commit
        maccess_func_dim_name_coeff[arr_name] = llstride_dimname

    return maccess_func_dim_name_coeff


# [Aux function] Build "d_bound_tiles_cacheline", which stores number of times a stride will be moved per dims.
#
# Inputs:
#  - llstride_dimname : Storing the stride/dim name for each array accesses
#    Comes from "build_maccess_func_coeff" + instance it to an "arr_name"
#  - d_ltilesizes : Dict [dim] |--> [ tilesize on different lambda branches of this dimension ]
#    Comes from "scheme::get_sizes_scheme"
#      WARNING: do not build from a "dprob_size" (ex: coming from "prob_sizes.py"), due to them being sizes
#        and not number of iterations (this is important for CONV when we have strides)
#  - cache_line_size : Due to last element treatment (which is 1 instead of "1/cache_line_size" )
# Output:
#  - ld_bound_tiles_cacheline : List [for each dim of the access]
# ( [Lambda branch] |--> Number of times we iterate on this stride)
# Same order of dim than "llstride_dimname"
#
# Note: trying to simplify the "llstride_dimname" and "ld_bound_tiles_cacheline"
#   is NOT a good idea.
#   Typically, applying a modulo "num_cache_set" to the stride could be viewed as a way to simplify the
#   later computation, except that it breaks an assumption in "arrange_ld_btcl_combi_dims"
# Likewise, having all the entries in llstride_dimname/ld_bound_tiles_cacheline is actually useful.
def build_bound_tiles_cacheline(
    comp: Computation,
    llstride_dimname: StrideInfos,
    d_ltilesizes: dict[str, List[int]],
    cache_line_size: int,
) -> IterTileInfos:
    # Example of llstride_dimname: [[204304, 'n'], [1808, 'h'], [904, 'r'], [8, 'w'], [4, 's'], [1, 'c']]

    # Recover the dimension name from the current array access
    ldim_names = [lelem[1] for lelem in llstride_dimname]
    ldim_all = get_ldims_computation(comp)
    ldims_stride = get_ldims_stride_computation(comp)

    # a) Lambda branch management: we need to list all possible lambda branches
    # Assumption: max 1 lambda per dimension

    # We list the number of lambda branch per dim
    d_lambda_num_branch = dict()
    n_global_lambda_br = 1
    for dim_name in d_ltilesizes:
        d_lambda_num_branch[dim_name] = len(d_ltilesizes[dim_name])
        n_global_lambda_br = n_global_lambda_br * len(d_ltilesizes[dim_name])

    # We iterate over all possible lambda branches and store their id in "ld_dim_i_lambda_branch"
    ld_dim_i_lambda_branch = []
    for i_br in range(n_global_lambda_br):
        # We recover the id of the lambda branch, ie the list of choices
        # Note: same code than "computation :: compute_footprint"
        temp_nbranch_unfolded = i_br
        d_dim_i_lambda_branch = dict()

        # Assumed to be the same order for all iteration of nbranch_unfolded
        for dim in d_lambda_num_branch:
            # Skip the stride infos
            if dim in ldims_stride:
                continue

            num_branch_dim = d_lambda_num_branch[dim]

            # Branch picked: modulo with "num_branch_dim"
            if num_branch_dim > 1:
                i_lambda_branch = temp_nbranch_unfolded % num_branch_dim

                # Remains (for the future dims)
                temp_nbranch_unfolded = temp_nbranch_unfolded // num_branch_dim

                d_dim_i_lambda_branch[dim] = i_lambda_branch

        # Sanity check
        assert temp_nbranch_unfolded == 0

        # Commit
        ld_dim_i_lambda_branch.append(d_dim_i_lambda_branch)

    # DEBUG
    # print(f"{ld_dim_i_lambda_branch=}")

    # b) For each dim, do the work
    ld_bound_tiles_cacheline = []
    for dim_name in ldim_names:
        d_bound_tiles_cacheline = dict()
        lsizes = d_ltilesizes[dim_name]

        # We need to use the global lambda branch id, not only the lambda choice restricted
        # to a single dim
        for d_dim_i_lambda_branch in ld_dim_i_lambda_branch:
            if dim_name in d_dim_i_lambda_branch:
                i_lambda_branch = d_dim_i_lambda_branch[dim_name]
            else:
                i_lambda_branch = 0
            str_lambda_loc = stringify_lambda_choice(ldim_all, d_dim_i_lambda_branch)
            d_bound_tiles_cacheline[str_lambda_loc] = lsizes[i_lambda_branch]

        # Commit
        ld_bound_tiles_cacheline.append(d_bound_tiles_cacheline)

    # c) Manage the last dimension
    # => Correct it to deal with cache_line_size (due to last item of maccess_func_coeff)
    d_innermost_dim = ld_bound_tiles_cacheline[-1]
    for str_lambda_loc in d_innermost_dim.keys():
        num_steps = d_innermost_dim[str_lambda_loc]
        n_num_steps = int(math.ceil(num_steps / cache_line_size))
        d_innermost_dim[str_lambda_loc] = n_num_steps

    return ld_bound_tiles_cacheline


# ====================================================================

# 2) Detailed FootPrint (DFP) computation

DFP_LvlArray = dict[str, List[int]]  # [lambda_loc] [cache set]


# Deep copy of dl_fp (detailed footprint, which is a dict of list)
def _deep_copy_dl_fp(dl_fp: DFP_LvlArray) -> DFP_LvlArray:
    n_dl_fp = dict()
    for k in dl_fp:
        n_dl_fp[k] = dl_fp[k].copy()
    return n_dl_fp


# [Aux function] Repeat while rotating a detailed footprint. This is the core operation
#   that computes the detailed footprint of the level above it.
#
# Inputs:
#  - dl_fp : detailed footprint (dict [lambda_loc] |--> list of size Nset, element = number of cache line in a cache set)
#  - d_ratio : number of rotations dict [lambda_loc] |--> number of time we need to rotate
#  - last_dl_fp : starting detailed footprint (from the previous loop level)
#  - d_shift_dim : shift factor when rotating on the given dim. dict: [lambda_loc] |--> integer
#  - dim_atom : dimension being rotated
#  - Nset : number of cache sets
#
# Note about the lambda_loc - if we have a new UL/TL: last_dl_fp have a different set of lambda_loc
#  compared to dl_fp/d_ratio/d_shift_dim
# All branches of dl_fp are initialized as the first iteration of their parent lambda_loc branch
#
# Output:
#  - dl_fp : freshly repeated & rotated
def repeat_and_rotate_dfp(
    dl_fp: DFP_LvlArray,
    d_ratio: dict[str, int],
    last_dl_fp: DFP_LvlArray,
    d_shift_dim: dict[str, int],
    dim_atom: str,
    Nset: int,
) -> DFP_LvlArray:
    for lambda_loc in d_ratio.keys():
        # "lambda_loc" should be present everywhere
        l_fp = dl_fp[lambda_loc]
        ratio = d_ratio[lambda_loc]
        shift_dim = d_shift_dim[lambda_loc]
        assert ratio > 0

        # last_l_fp: if we have a new ULambda/TLambda, the lambda_loc might differs
        if lambda_loc in last_dl_fp.keys():
            # No problem
            last_l_fp = last_dl_fp[lambda_loc]
        else:
            # We have a new ULambda/TLambda situation.
            # We need to recover the corresponding lambda loc below the UL/TL
            ldims_ref = get_list_dims_str_lambda_loc(lambda_loc)
            lambda_loc_below = remove_dim_from_stringified(
                lambda_loc, dim_atom, ldims_ref
            )
            last_l_fp = last_dl_fp[lambda_loc_below]

        assert len(l_fp) == Nset

        # Note: first iteration (i_iter = 0) already done in l_fp
        for i_iter in range(1, ratio):
            # We sum the pattern of last_l_fp on l_fp, while shifting by (i_iter * shift_dim)
            # 		and modulo Nset
            shift_pattern = i_iter * shift_dim

            # Repeat & rotate
            for k in range(Nset):
                nk = (k + shift_pattern) % Nset
                l_fp[nk] = l_fp[nk] + last_l_fp[k]

    return dl_fp


# [Aux function] Sum the contributions of different lambda branches. Useful for the "Seq" case
#
# Inputs:
#  - dl_fp : detailed footprint (dict [lambda_loc] |--> list of size Nset, element = number of cache line in a cache set)
#     We assume that dl_fp keys (the lambda_loc) are the lambda_loc from "after Seq"
#       (ie. the dim of the Seq does not appear in the lambda_loc anymore since it is merged)
#       We do not care about the values of dl_fp (since we can easily recompute it from last_dl_fp /
#         no need to assume it got the first branch here)
#  - last_dl_fp : starting detailed footprint (from the previous loop level)
#  - d_lshift : shift factor of each branches of the lambda. dict: [lambda_loc] |--> List[integer] (one per branch)
#  - dim_seq : dimension of the Seq atom currently being considered
#  - Nset : number of cache sets
# Output:
#  - dl_fp, freshly repeated & rotated (after the merge of the branches of the current "Seq")
def sum_with_shift(
    dl_fp: DFP_LvlArray,
    last_dl_fp: DFP_LvlArray,
    d_lshift: dict[str, List[int]],
    dim_seq: str,
    Nset: int,
) -> DFP_LvlArray:
    # [Preprocessing] Match the lambda_loc of last_dl_fp to a lambda_loc from dl_fp

    # Recover the list of dims used in dl_fp (above the Seq atom)
    for str_lambda_loc in dl_fp.keys():
        ldim_above_seq = get_list_dims_str_lambda_loc(str_lambda_loc)
        break

    # dl_lambda_above_to_lambda_below : [str_lambda_loc_above] |---> [ list of all corresponding str_lambda_loc below seq]
    dl_lambda_above_to_lambda_below: dict[str, List[str]] = dict()
    for str_lambda_loc_belseq in last_dl_fp.keys():
        d_dim_brid_belseq = recover_all_branchid_from_stringified(str_lambda_loc_belseq)

        # Corresponding lambda from above seq
        del d_dim_brid_belseq[dim_seq]
        str_lambda_loc_aboveseq = stringify_lambda_choice(
            ldim_above_seq, d_dim_brid_belseq
        )

        # Commit
        if str_lambda_loc_aboveseq in dl_lambda_above_to_lambda_below:
            dl_lambda_above_to_lambda_below[str_lambda_loc_aboveseq].append(
                str_lambda_loc_belseq
            )
        else:
            dl_lambda_above_to_lambda_below[str_lambda_loc_aboveseq] = [
                str_lambda_loc_belseq
            ]

    # DEBUG
    # print(f"{dl_lambda_above_to_lambda_below=}")

    # [Start of the algo] Iterate on the lambda_loc above Seq
    for lambda_loc_above_seq in dl_fp.keys():
        # We recover the list of lambda_loc below seq
        l_lambda_loc_below_seq = dl_lambda_above_to_lambda_below[lambda_loc_above_seq]

        # Sum over all the branches that arrive here
        l_fp = [0] * Nset
        for lambda_loc_below_seq in l_lambda_loc_below_seq:
            # Recover which branch are we looking at & their info
            i_branch = recover_branchid_from_stringified(lambda_loc_below_seq, dim_seq)

            last_l_fp = last_dl_fp[lambda_loc_below_seq]
            shift_pattern = d_lshift[lambda_loc_above_seq][i_branch]
            for k in range(Nset):
                nk = (k + shift_pattern) % Nset
                l_fp[nk] = l_fp[nk] + last_l_fp[k]

        # Commit
        dl_fp[lambda_loc_above_seq] = l_fp

    return dl_fp


# Alternative to level-by-level computation, when it does not rotate nicely from previous loop lvl
# In that case, we perform a direct computation, without relying on the dfp from the loop level below
#
# Inputs:
#  - ll_stride_dimname : Storing the stride/dim name for each array accesses
#     List of ( stride [granularity in cache_line], dim_name )
#  - ld_bound_tiles_cacheline : Number of iterations over a dimension
#     List [order of ll_stride_dimname elements]  ( [Lambda branch] |--> Number of times we iterate on this stride)
#
# [Inputs to recover the initial shift of the array]
#  - comp : current computation
#  - lcont_arr_order : order of contiguous allocation of the arrays used by comp
#  - array_name : name of the array currently considered
#  - dprob_sizes : problem sizes from the complete scheme
#     (ex: obtained as result of "scheme :: get_unlambda_sizes_scheme")
#  - num_cache_set : number of cache set in the currently considered cache
#  - cache_line_size : number of elements in a cache line
#
# [Optional argument]
#  - ldim_ignore ["[]" by default]: list of dimension that we skip
#     (typically because they are part of a combination in an array access, cf array "I" in CONV
#      since they might be later computed from scratch)
#  - starting_dl_fp ["None" by default]: starting "dl_fp", if we do not start from scratch
# WARNING: we assume that the dimension we consider are starting from scratch (it impacts the shift computation)
# => This "starting_dl_fp" should have content only for dimension that are ignored
#
# Output:
# - dl_fp, for this loop level and array
# dict [lambda_loc] |---> [ list of number of cache lines accessed per cache sets]
def dl_fp_direct_computation(
    ll_stride_dimname: StrideInfos,
    ld_bound_tiles_cacheline: IterTileInfos,
    comp: Computation,
    lcont_arr_order: List[str],
    arr_name: str,
    dprob_sizes: dict[str, int],
    num_cache_set: int,
    cache_line_size: int,
    ldim_ignore: List[str] = [],
    starting_dl_fp: Optional[DFP_LvlArray] = None,
) -> DFP_LvlArray:
    # 1) Starting footprint
    # For each lambda branches...
    dl_fp = dict()
    for str_lambda_loc in ld_bound_tiles_cacheline[0].keys():
        l_fp = []
        if starting_dl_fp != None:
            l_fp = starting_dl_fp[str_lambda_loc]
            assert len(l_fp) == num_cache_set
        else:
            # Build it from scratch
            l_fp = [0] * num_cache_set
            i_shift_init = get_shift_array(
                comp,
                lcont_arr_order,
                arr_name,
                dprob_sizes,
                num_cache_set,
                cache_line_size,
            )
            l_fp[i_shift_init] = 1

        # Commit
        dl_fp[str_lambda_loc] = l_fp

    # 2) For each dimension of the stride (which is not ignored),
    # we repeat & rotate l_fp
    assert len(ll_stride_dimname) == len(ld_bound_tiles_cacheline)
    for i_dim in range(len(ll_stride_dimname)):
        # Recover the infos
        l_stride_dimname = ll_stride_dimname[i_dim]
        stride = l_stride_dimname[0]
        dim_name = l_stride_dimname[1]
        d_bound_tiles_cacheline = ld_bound_tiles_cacheline[i_dim]

        # If we are on a dim to be ignored, skip it
        if dim_name in ldim_ignore:
            continue

        # We prepare for a "repeat and rotate"
        # - d_ratio = d_bound_tiles_cacheline
        # - d_shift_dim needs to be computed from the size below?
        #  However, by construction (and input hypothesis), it is always 1
        size_tile_below = 1  # By construction
        shift_dim = stride * size_tile_below

        d_shift_dim = dict()
        for str_lambda_loc in d_bound_tiles_cacheline.keys():
            d_shift_dim[str_lambda_loc] = shift_dim

        last_dl_fp = _deep_copy_dl_fp(dl_fp)

        # Go!
        dl_fp = repeat_and_rotate_dfp(
            dl_fp,
            d_bound_tiles_cacheline,
            last_dl_fp,
            d_shift_dim,
            dim_name,
            num_cache_set,
        )

        # DEBUG
        # print(f"Dim {dim_name} - dl_fp={dl_fp}")

    return dl_fp


# [Aux function] Level-by-level dfp computation - case where we have a ratio from previous loop lvl
#   (i.e., currently T, U, UL, TL, Tparal)
#
# Input:
#  - last_dl_fp : "dl_fp" from the loop level right below it
#  - llstride_dimname : Storing the stride/dim name for each array accesses
#     List of ( stride [granularity in cache_line], dim_name )
#  - d_sizes_below : basically "ld_bound_tiles_cacheline_below"
#        (Number of iterations over a dimension, loop lvl below the atom)
#     But with the lambda_loc of d_ratio
#     ([Lambda branch] |--> Number of times we iterate on this stride)
#
#  - comp : current computation
#  - arr_name : name of the current array
#
#  - dim_atom : dimension on which the shift is happening
#  - d_ratio : dict of ratio ([lambda_loc] |--> ratio)
# This might need to be build from the information of the atom
#     Note: if we have a new UL/TL that modify the set of lambda_loc,
#       the ones from d_ratio are taken as reference to build the new "dl_fp".
#  - num_cache_set : number of cache sets
#
# Output:
#  - dl_fp : where the repeat/rotation of the current atom is now done
def _dl_fp_ratio_atom(
    last_dl_fp: DFP_LvlArray,
    llstride_dimname: StrideInfos,
    d_sizes_below: dict[
        str, int
    ],  # ld_bound_tiles_cacheline_above, ld_bound_tiles_cacheline_below,
    comp: Computation,
    arr_name: str,
    dim_atom: str,
    d_ratio: dict[str, int],
    num_cache_set: int,
) -> DFP_LvlArray:
    # 1) Do we have reuse along dimension "dim_atom" for this array
    #  This is checked by seeing if "dim_atom" is inside "llstride_dimname"
    #  If yes, then no computation required: dl_fp is exactly the one from the previous iteration
    kdim_atom = -1
    for kstride_dim in range(len(llstride_dimname)):
        (stride, dimname) = llstride_dimname[kstride_dim]
        if dimname == dim_atom:
            kdim_atom = kstride_dim

    if kdim_atom == (-1):
        dl_fp = _deep_copy_dl_fp(last_dl_fp)
        return dl_fp

    # Recover the pertinent infos
    dstride = dict()
    for lambda_loc in d_ratio.keys():
        base_stride_dim = llstride_dimname[kdim_atom][0]
        current_dim_size = d_sizes_below[lambda_loc]
        dstride[lambda_loc] = base_stride_dim * current_dim_size

    # 3) We are done being paranoid, so we simply call repeat_and_rotate_dfp
    #  on the right arguments
    d_shift_dim = dict()
    for lambda_loc in d_ratio.keys():
        d_shift_dim[lambda_loc] = dstride[lambda_loc]

    # dl_fp : we need to manage its potential new set of lambda_loc.
    # Knowing if we have a new lambda is doable by comparing the lambda_loc of "d_ratio" with "last_dl_fp"
    llambda_loc_temp = list(d_ratio.keys())
    lambda_loc_dratio_0 = llambda_loc_temp[0]

    # DEBUG
    # print()
    # print(f"{arr_name=} {dim_atom=}")
    # print(f"   {last_dl_fp=}")
    # print(f"   {d_ratio=} {d_sizes_below=}")
    # print(f"   {lambda_loc_dratio_0=} last_dl_fp_keys={list(last_dl_fp.keys())}")

    if lambda_loc_dratio_0 in last_dl_fp.keys():
        # Simple case, a deep copy is enough
        dl_fp = _deep_copy_dl_fp(last_dl_fp)
    else:
        # More complicated case: we need to duplicate the values of last_dl_fp
        #  in respect to the new lambda_loc
        ldims_ref_lambda_loc_above = get_list_dims_str_lambda_loc(lambda_loc_dratio_0)

        dl_fp = dict()
        for lambda_loc_above in d_ratio.keys():
            # Recover the corresponding lambda_loc from below
            lambda_loc_below = remove_dim_from_stringified(
                lambda_loc_above, dim_atom, ldims_ref_lambda_loc_above
            )
            l_fp_below = last_dl_fp[lambda_loc_below]

            # Commit
            dl_fp[lambda_loc_above] = l_fp_below.copy()

    dl_fp = repeat_and_rotate_dfp(
        dl_fp, d_ratio, last_dl_fp, d_shift_dim, dim_atom, num_cache_set
    )

    return dl_fp


# [Aux function] Level-by-level dfp computation - case of the Seq atom
#
# Input:
#  - last_dl_fp : "dl_fp" from the loop level right below it
#  - llstride_dimname : Storing the stride/dim name for each array accesses
#     List of ( stride [granularity in cache_line], dim_name )
#  - ld_bound_tiles_cacheline_below : Number of iterations over a dimension, loop lvl below the atom
#     List [order of llstride_dimname elements]  ( [Lambda branch] |--> Number of times we iterate on this stride)
#
#  - dim_atom : dimension on which the shift is happening
#  - num_cache_set : number of cache sets
#
# Output:
#  - dl_fp : where the repeat/rotation of the current atom is now done
def _dl_fp_seq_atom(
    last_dl_fp: DFP_LvlArray,
    llstride_dimname: StrideInfos,
    ld_bound_tiles_cacheline_below: IterTileInfos,
    dim_atom: str,
    num_cache_set: int,
) -> DFP_LvlArray:
    # 0) The lambda branches are getting reduced here: we need to prepare the new dl_fp
    dl_fp = dict()

    # Recover the list of dims used in dl_fp (above the Seq atom)
    ldims_lambdaloc_aboveseq = []
    ldims_lambdaloc_belowseq = []
    for str_lambda_loc in last_dl_fp.keys():
        ldims_lambdaloc_belowseq = get_list_dims_str_lambda_loc(str_lambda_loc)

        ldims_lambdaloc_aboveseq = ldims_lambdaloc_belowseq.copy()
        ldims_lambdaloc_aboveseq.remove(dim_atom)
        break

    num_branch_seq = get_max_num_lambda_branch(list(last_dl_fp.keys()), dim_atom)

    # DEBUG
    # print(f"{ldims_lambdaloc_belowseq=}  | {ldims_lambdaloc_aboveseq=} | {num_branch_seq=}")

    # Build a dummy dl_fp. Its keys are correct, its value is the branch of (id=0) matching this key.
    for str_lambda_loc in last_dl_fp.keys():
        d_dim_brid = recover_all_branchid_from_stringified(str_lambda_loc)

        # Not the first branch => skip it
        if d_dim_brid[dim_atom] != 0:
            continue

        # Get the corresponding key above the Seq atom
        del d_dim_brid[dim_atom]
        str_lambda_loc_aboveseq = stringify_lambda_choice(
            ldims_lambdaloc_aboveseq, d_dim_brid
        )

        # Deep copy, adapted to the reduced str_lambda_loc
        dl_fp[str_lambda_loc_aboveseq] = last_dl_fp[str_lambda_loc].copy()

    # 1) Do we have reuse along dimension "dim_atom" for this array
    #  This is checked by seeing if "dim_atom" is inside "llstride_dimname"
    #  If yes, then no computation required: dl_fp is exactly the one from the previous iteration
    kdim_atom = -1
    for kstride_dim in range(len(llstride_dimname)):
        (stride, dimname) = llstride_dimname[kstride_dim]
        if dimname == dim_atom:
            kdim_atom = kstride_dim

    if kdim_atom == (-1):
        # Lambda loc management already done at step 0
        # + because the array does not interact with this dim, the first branch is enough.
        #  (technically any branch would be enough)
        return dl_fp

    # Recover the pertinent infos
    stride_dim = llstride_dimname[kdim_atom][0]
    d_sizes_below = ld_bound_tiles_cacheline_below[kdim_atom]  # d = lambda_branches

    # DEBUG
    # print(f"{stride=} | {d_sizes_below=}")

    # 2) Prepare the call to "sum_with_shift"

    # We build d_lshift, containing the shift of each lambda_branch between each other
    # Its lambda_loc are the ones "above Seq"
    d_lshift = dict()
    for lambda_loc_above in dl_fp.keys():
        l_shift = [0]  # First branch (id=0) already counted

        acc = 0
        for i in range(1, num_branch_seq):
            # We check the lambda branches before the start of the i-th branch
            #  => At iteration "i", we add the contribution of the "i-1"-th lambdabranch

            # Build the id of the (i-1)-th branch below seq that ends up in "lambda_loc_above"
            d_dim_brid = recover_all_branchid_from_stringified(lambda_loc_above)
            d_dim_brid[dim_atom] = i - 1
            lambda_loc_below_id_i = stringify_lambda_choice(
                ldims_lambdaloc_belowseq, d_dim_brid
            )

            # Now that we have the id of the branch, we can recover its size
            size_prev_lambda_br = d_sizes_below[lambda_loc_below_id_i]
            shift_prev_lambda_br = size_prev_lambda_br * stride_dim

            # DEBUG
            # print(f"{size_prev_lambda_br=} | {stride_dim=}")
            # print(f"{shift_prev_lambda_br=}")

            # Accumulate
            acc = (acc + shift_prev_lambda_br) % num_cache_set
            l_shift.append(acc)

        # Commit
        d_lshift[lambda_loc_above] = l_shift

    # Let's do it
    dl_fp = sum_with_shift(dl_fp, last_dl_fp, d_lshift, dim_atom, num_cache_set)

    return dl_fp


# [Very Aux function] Auxilliary function that arrange ld_bound_tiles_cacheline to prepare for a direct computation
#  Happens only in the case of dimensions that are combined together in the same array access expression (ex: "h*strx+w")
#
# Input:
#  - ll_stride_dimname: list of ( stride [granularity in cache_line], dim_name )
#   (used for the "dim_name" order)
#  - ld_bound_tiles_cacheline: list of ( [Lambda branch] |--> Number of times we iterate on this stride)
#   (order shared by ll_stride_dimname)
#  - ldims_combi: the dimension of the affine array access expression being considered.
# Output:
#  - ld_bound_tiles_cacheline_combi: arranged ld_bound_tiles_cacheline_combi
#    Note: the correspondance with the dims of ll_stride_dimname must be kept.
def arrange_ld_btcl_combi_dims(
    ll_stride_dimname: StrideInfos,
    ld_bound_tiles_cacheline: IterTileInfos,
    ldims_combi: List[str],
) -> IterTileInfos:
    # Deep copy of ld_bound_tiles_cacheline
    nld_bound_tiles_cacheline = []
    for d_lambdaloc_iter in ld_bound_tiles_cacheline:
        nld_bound_tiles_cacheline.append(d_lambdaloc_iter.copy())

    # 1) We identify the range of index in ll_stride_dimname that match ldims_combi
    l_dimname_order = [elem[1] for elem in ll_stride_dimname]
    l_index_dimname = [l_dimname_order.index(dim_name) for dim_name in ldims_combi]

    # Assumption: must be a single interval (and not scattered-out)
    min_index = min(l_index_dimname)
    max_index = max(l_index_dimname)
    assert max_index - min_index + 1 == len(ldims_combi)

    # Note: in the case of an access like "h*strx + r", the trick of capping the number of iterations
    # of "r" by "strx" (and) does not work due to issues on the final size of this access.
    # We need to separate that into 2 cases:
    #  - if R<strx : we have holes, and we need to have 2 entries in ld_bound_tiles_cachelines
    #  - if R>=strx : Merge the entries from h and r: H is set to 1 iter and R to combined range of iters
    # in ld_bound_tiles_cachelines
    #
    # When generalizing this condition, we obtain the following algorithm:
    #
    # We go from inner to outer stride dimension
    #  (note: stride can be equal between dimensions, ex: if we have "h+r")
    #  If we have a inner stride dim with more iteration than its ratio with the outer stride dim, merge them
    #  Else, keep them separate

    # Other hidden assumption: the strides must be divisible (or equal) between 2 consecutive dim combi
    # and in decreasing order in ll_stride_dimname
    for ind in range(min_index, max_index):
        stride_out = ll_stride_dimname[ind][0]
        stride_in = ll_stride_dimname[ind + 1][0]

        if stride_out < stride_in:
            raise ValueError(
                f"arrange_ld_btcl_combi_dims : strides of combi dims are not in the right order"
            )
        if stride_out % stride_in != 0:
            dimname_out = ll_stride_dimname[ind][1]
            dimname_in = ll_stride_dimname[ind + 1][1]
            raise ValueError(
                f"arrange_ld_btcl_combi_dims : strides of dims {dimname_out} and {dimname_in} are not divisible"
            )

    # 2) For each dimension from outer to inner
    for lambda_loc in ld_bound_tiles_cacheline[0].keys():
        # For a given branch, check what should we give as ld_bound_tiles_cacheline_combi

        # Information to keep if we detect a interval of continous cache line access across dims
        start_interv_ind = -1
        current_interv_size = 0  # Accumulation of the size of the intervals

        for ind in range(max_index, min_index, -1):  # [max_index, ... , min_index+1]
            # DEBUG
            # print(f"Dim {ll_stride_dimname[ind][1]} - {start_interv_ind=} , {current_interv_size=}")

            num_iter_below = nld_bound_tiles_cacheline[ind][lambda_loc]
            if start_interv_ind == (-1):
                stride_start = ll_stride_dimname[ind][0]
            else:
                stride_start = ll_stride_dimname[start_interv_ind][0]
            stride_above = ll_stride_dimname[ind][0]
            stride_below = ll_stride_dimname[ind - 1][0]

            # Difference of stride between the stride of the start of the interval and the current stride
            stridediff_below = int(stride_below / stride_start)
            stridediff_above = int(stride_above / stride_start)

            # Updating the current continuous interval size
            # When commit to start_interv_ind (intervals ends)
            # 	interval_size = 1 + \sum_{dim} [stridediff_dim] * ( [num_iter_dim] - 1)
            current_interv_size = current_interv_size + stridediff_above * (
                num_iter_below - 1
            )

            # DEBUG
            # print(f"\t{stride_below=} , {stride_start=}")
            # print(f"\t{num_iter_below=} , {stridediff_below=}")

            if num_iter_below >= stridediff_below:
                # We have a continous part (starting or following)

                # Are we at the start of an interval ?
                if start_interv_ind == -1:
                    start_interv_ind = ind
                else:
                    # Continue interval (there are 3 or more dims that are continous)
                    # current_interv_size already updated
                    pass
            else:
                # No interval in progress: no modification required
                if start_interv_ind == (-1):
                    current_interv_size = 0
                    continue

                # We have a break of the continous interval: update ld_bound_tiles_cacheline

                # Finishing the computation of current_interv_size
                current_interv_size = current_interv_size + 1
                nld_bound_tiles_cacheline[start_interv_ind][lambda_loc] = (
                    current_interv_size
                )

                for ind_mid in range(ind, start_interv_ind):
                    nld_bound_tiles_cacheline[ind_mid][lambda_loc] = 1

                # Reset the interval size and the start of the interval
                current_interv_size = 0
                start_interv_ind = -1

        # DEBUG
        # print(f"End loop (dim {ll_stride_dimname[min_index][1]}) - {start_interv_ind=} , {current_interv_size=}")

        # We check if we have an overflow, ie if "start_interv_ind != -1" at the end
        if start_interv_ind != (-1):
            # Update current_interv_size and update with iteration "min_index"
            stride_above = ll_stride_dimname[min_index][0]
            stride_start = ll_stride_dimname[start_interv_ind][0]
            stridediff_above = int(stride_above / stride_start)

            num_iter_below = nld_bound_tiles_cacheline[min_index][lambda_loc]
            current_interv_size = current_interv_size + stridediff_above * (
                num_iter_below - 1
            )

            nld_bound_tiles_cacheline[start_interv_ind][lambda_loc] = (
                current_interv_size + 1
            )

            for ind_mid in range(min_index, start_interv_ind):
                nld_bound_tiles_cacheline[ind_mid][lambda_loc] = 1
    return nld_bound_tiles_cacheline


# [Main component function] Compute the detailed footprint across all loop lvl for all arrays
#     by reusing the dfp from the previous level whenever it is possible.
#   This is the big one.
#
# Input:
#  - scheme : the (complete) scheme being considered
#  - comp : the considered computation
#  - prob_sizes : the problem sizes (including the potential stride dimensions)
#  - lcont_arr_order : order of contiguous allocation across arrays
#  - num_cache_set : number of cache sets in the cache
#  - cache_line_size : number of element in a cache line
#
# Output:
#  - lddl_fp : [loop lvl] [array_name] [lambda_branch] [cache set]
#    Contains all the data footprints across all loop lvls and arrays of the computation
#    Note: the lambda_loc in the lddl MUST be the same across of the array_name (needed for later permutation)
def periodic_extra_cacheset_estimation_lvl(
    scheme: List[Atom],
    comp: Computation,
    prob_sizes: dict[str, int],
    lcont_arr_order: List[str],
    num_cache_set: int,
    cache_line_size: int,
    b_sanity_check: bool = False,
) -> List[dict[str, DFP_LvlArray]]:
    # 1) Preprocessing step - gather the information about the number of iterations + shift of the arrays
    l_dims_comp = get_ldims_computation(comp)  # For lambda_loc reconstruction
    l_dims_stride = get_ldims_stride_computation(comp)

    maccess_func_dim_name_coeff = build_maccess_func_coeff(
        comp, prob_sizes, cache_line_size
    )

    dlld_bound_tiles_cacheline = (
        dict()
    )  # [Array] [loop_lvl] [dim_access_array] [lambda_loc] => [num_iter]
    for array_name in maccess_func_dim_name_coeff.keys():
        llstride_dimname = maccess_func_dim_name_coeff[array_name]

        # Update llstride_dimname
        # llstride_dimname = simplify_periodic_cs_estim(llstride_dimname, num_cache_set)
        # ===> Note: provoke issues with "arrange_ld_btcl_combi_dims" hypothesis
        maccess_func_dim_name_coeff[array_name] = llstride_dimname

        lld_bound_tiles_cacheline = []
        for loop_lvl in range(len(scheme)):
            sub_scheme = scheme[: loop_lvl + 1]

            d_ltilesizes = get_sizes_scheme(sub_scheme)
            for dim_stride in l_dims_stride:  # Stride dimensions: same than prob_sizes
                d_ltilesizes[dim_stride] = [prob_sizes[dim_stride]]
            d_def_sizes = get_default_sizes(comp)
            d_ltilesizes = d_def_sizes | d_ltilesizes

            # DEBUG
            # print(d_ltilesizes)

            ld_bound_tiles_cacheline = build_bound_tiles_cacheline(
                comp, llstride_dimname, d_ltilesizes, cache_line_size
            )

            # DEBUG
            # print(f"{array_name=} {loop_lvl=} -> {ld_bound_tiles_cacheline=}")

            # Commit!
            lld_bound_tiles_cacheline.append(ld_bound_tiles_cacheline)

        # Commit!
        dlld_bound_tiles_cacheline[array_name] = lld_bound_tiles_cacheline

    # DEBUG
    # print(f"{maccess_func_dim_name_coeff=}")
    # print(f"{dlld_bound_tiles_cacheline['O'][3]=}")

    # 2) Init - first loop lvl
    lddl_fp = []  # [loop lvl] [array_name] [lambda_branch] [cache set]

    # The atom at the first loop lvl should always be Vectorisation
    assert scheme[0].type == AtomType.VECT

    ddl_fp_first = dict()
    for array_name in maccess_func_dim_name_coeff.keys():
        dl_fp_first = dict()

        # At that level, there are no lambda loc
        str_lambda_loc = ""

        # We init l_fp_first with "nothing in all the cache sets"
        l_fp_first = []
        for k in range(num_cache_set):
            l_fp_first.append(0)

        # Due to the size of vectorization compared to the cache line size,
        # there is a single cache set accessed at this loop lvl
        assert scheme[0].ratio <= cache_line_size

        # We compute the starting cache set of the considered array
        # It is supposed to be at least aligned with the cache lines
        start_shift = get_shift_array(
            comp,
            lcont_arr_order,
            array_name,
            prob_sizes,
            num_cache_set,
            cache_line_size,
        )
        l_fp_first[start_shift] = 1

        # DEBUG
        # print(f"{array_name=} => {start_shift=}")

        # Commit
        dl_fp_first[str_lambda_loc] = l_fp_first
        ddl_fp_first[array_name] = dl_fp_first

    # Commit
    lddl_fp.append(ddl_fp_first)

    # DEBUG
    # print("\t - End Init pass:")
    # print_lddl_fp(lddl_fp)

    # 3) First pass - dimensions that are alone in their expression of the considered array access
    #  (else, we have a issue with combined dimensions, such as h+r, where a "T(h,34) T(r,3)"
    # cannot be obtained through a repeat/rotation)
    # These remaining dimensions (from d_combi_dims) will be managed during the second pass,
    #   by using a direct computation to complete the dfp calculation.
    d_arrays_accs = get_array_accesses(comp)
    d_combi_dims = get_combi_dimensions(comp)

    # Reminder for the lowly programmer/debugger that I am:
    #   lddl_fp : [loop lvl] [array_name] [lambda_branch] [cache set]
    for loop_lvl in range(2, len(scheme) + 1):
        atom_below = scheme[loop_lvl - 1]
        sub_scheme = scheme[:loop_lvl]

        # Recover the dfp from the previous level
        last_ddl_fp = lddl_fp[loop_lvl - 2]

        # Let's start
        ddl_fp = dict()
        for array_name in maccess_func_dim_name_coeff.keys():
            # Need to compute the dl_fp [lambda_loc] [cache set] for this loop lvl and array name

            # First, we recover ldims_combi (the dimensions to be banned)
            lldims_combi = d_combi_dims[array_name]
            ldims_combi = []
            for lelem in lldims_combi:
                lnew_elem = lelem[0]
                for ndim in lnew_elem:
                    ldims_combi.append(ndim)

            # DEBUG
            # print(f"{ldims_combi=}")

            # Note: do we try to get the case "h+r" where the size of "r" is 1 ?

            # We check if the dimension of the atom is inside ldims_combi of this array
            # If yes, we have to ignore it.
            dim_atom = atom_below.dim
            last_dl_fp = last_ddl_fp[array_name]

            # Note: due to lambda_loc management, this is factorized with the code of
            #    the case where the current dimension does not interact with the array
            b_skip_level = dim_atom in ldims_combi

            # We can now safely repeat & rotate on this dim.
            # We recover the pertinent informations to call either "_dl_fp_ratio_atom" or "_dl_fp_seq_atom"
            llstride_dimname = maccess_func_dim_name_coeff[array_name]
            ld_bound_tiles_cacheline = dlld_bound_tiles_cacheline[array_name][
                loop_lvl - 1
            ]
            ld_bound_tiles_cacheline_below = dlld_bound_tiles_cacheline[array_name][
                loop_lvl - 2
            ]

            # We find the index of dim_atom inside llstride_dimname (shared with ld_bound_tiles_cacheline)
            kdim_llstride_dimname = -1
            for kdim in range(len(llstride_dimname)):
                (stride, dim) = llstride_dimname[kdim]
                if dim == dim_atom:
                    kdim_llstride_dimname = kdim

            # If dim_atom is not in the currently examined array access => no change
            #   (also the code if the dimension is part of the combination dimension => skip this level for this pass)
            if b_skip_level or (kdim_llstride_dimname == (-1)):
                if atom_below.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
                    # Is it the first lambda of this dimension?
                    l_lambda_loc_ld_btc = list(ld_bound_tiles_cacheline_below[0].keys())
                    d_dim_bid_random_lambda_loc = recover_all_branchid_from_stringified(
                        l_lambda_loc_ld_btc[0]
                    )
                    if dim_atom in d_dim_bid_random_lambda_loc.keys():
                        # Not the first lambda in this dim => Deep copy is enough
                        dl_fp = _deep_copy_dl_fp(last_dl_fp)
                        ddl_fp[array_name] = dl_fp
                        continue
                    else:
                        # We are encountering the first lambda on this dimension
                        #   We have work to do here to correct the lambda loc used
                        num_br_lambda = len(atom_below.lratios)
                        dl_fp = dict()
                        for lambda_loc in last_dl_fp.keys():
                            for br_id in range(num_br_lambda):
                                # Build the new lambda_loc by adding the entry for dim_atom
                                d_dim_bid = recover_all_branchid_from_stringified(
                                    lambda_loc
                                )
                                d_dim_bid[dim_atom] = br_id
                                new_lambda_loc = stringify_lambda_choice(
                                    l_dims_comp, d_dim_bid
                                )

                                # Copy
                                dl_fp[new_lambda_loc] = last_dl_fp[lambda_loc].copy()

                        ddl_fp[array_name] = dl_fp
                        continue
                elif atom_below.type in [AtomType.SEQ]:
                    # We need to reduce the lambda_loc of dl_fp
                    # Because dim_atom does not interact with the current array, all branches
                    #    across "dim_atom" have the same contents in last_dl_fp.
                    dl_fp = dict()
                    for lambda_loc in last_dl_fp.keys():
                        d_dim_bid = recover_all_branchid_from_stringified(lambda_loc)
                        del d_dim_bid[dim_atom]
                        new_lambda_loc = stringify_lambda_choice(l_dims_comp, d_dim_bid)

                        # Copy only if not already there
                        if new_lambda_loc not in dl_fp.keys():
                            dl_fp[new_lambda_loc] = last_dl_fp[lambda_loc].copy()

                    ddl_fp[array_name] = dl_fp
                    continue
                else:
                    assert atom_below.type in [
                        AtomType.UNROLL,
                        AtomType.TILE,
                        AtomType.TILE_PARAL,
                    ]
                    # We create dl_fp as a copy of last_dl_fp
                    dl_fp = _deep_copy_dl_fp(last_dl_fp)
                    ddl_fp[array_name] = dl_fp
                    continue
            # End of the case where the dimension does not interact with the array access

            # Reminder:
            # - llstride_dimname : ( stride [granularity in cache_line], dim_name )
            # - ld_bound_tiles_cacheline : List [for each dim of the access]
            #   ( [Lambda branch] |--> Number of times we iterate on this stride)
            if atom_below.type in [
                AtomType.UNROLL,
                AtomType.TILE,
                AtomType.TILE_PARAL,
                AtomType.ULAMBDA,
                AtomType.TLAMBDA,
            ]:
                # The only thing we need to build is "d_ratio"
                # If we have a ULAMBDA/TLAMBDA, the keys of d_ratio much be the new set of lambda_loc.
                d_ratio = dict()
                d_sizes_below = dict()

                if atom_below.type in [
                    AtomType.UNROLL,
                    AtomType.TILE,
                    AtomType.TILE_PARAL,
                ]:
                    # str_lambda_loc are the same than below
                    for lambda_loc in ld_bound_tiles_cacheline[0].keys():
                        size_above = ld_bound_tiles_cacheline[kdim_llstride_dimname][
                            lambda_loc
                        ]
                        size_below = ld_bound_tiles_cacheline_below[
                            kdim_llstride_dimname
                        ][lambda_loc]
                        assert size_above % size_below == 0
                        ratio = int(size_above / size_below)

                        d_ratio[lambda_loc] = ratio
                        d_sizes_below[lambda_loc] = size_below
                elif atom_below.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
                    # DEBUG
                    # print(f"{ld_bound_tiles_cacheline[kdim_llstride_dimname]=}")
                    # print(f"{ld_bound_tiles_cacheline_below[kdim_llstride_dimname]=}")

                    # We check if "dim_atom" is already in the "lambda_loc"
                    # If yes: the lambda_loc of ld_bound_tiles_cacheline[0] are ok, we just need to match them
                    # If no: we need to build d_ratio, since "dim_atom" just had its first lambda encountered
                    l_lambda_loc_ld_btc = list(ld_bound_tiles_cacheline_below[0].keys())
                    d_dim_bid_random_lambda_loc = recover_all_branchid_from_stringified(
                        l_lambda_loc_ld_btc[0]
                    )

                    # If the lambda_loc already showed that there is a lambda on this dim,
                    #   this is the continuation of the same lambda branch (first case). Else, this is a new lambda.
                    if dim_atom in d_dim_bid_random_lambda_loc.keys():
                        # Not a new lambda in this dimension
                        for lambda_loc in ld_bound_tiles_cacheline[0].keys():
                            # Do not use the ratio but ld_bound_tiles_cacheline (due to last dim management)
                            # Check on which br_id we are for dim_atom
                            # br_id = recover_branchid_from_stringified(lambda_loc, dim_atom)
                            # d_ratio[lambda_loc] = atom_below.lratios[br_id]
                            size_above = ld_bound_tiles_cacheline[
                                kdim_llstride_dimname
                            ][lambda_loc]
                            size_below = ld_bound_tiles_cacheline_below[
                                kdim_llstride_dimname
                            ][lambda_loc]
                            assert size_above % size_below == 0
                            ratio = int(size_above / size_below)

                            d_ratio[lambda_loc] = ratio
                            d_sizes_below[lambda_loc] = size_below
                    else:
                        # New lambda in this dimension
                        for lambda_loc in ld_bound_tiles_cacheline_below[0].keys():
                            for br_id in range(len(atom_below.lratios)):
                                # Build the new lambda_loc by adding the entry for dim_atom
                                d_dim_bid = recover_all_branchid_from_stringified(
                                    lambda_loc
                                )
                                d_dim_bid[dim_atom] = br_id
                                new_lambda_loc = stringify_lambda_choice(
                                    l_dims_comp, d_dim_bid
                                )

                                # DEBUG
                                # print(f"{lambda_loc=} + ({dim_atom}, {br_id=}) => {new_lambda_loc=}")

                                size_above = ld_bound_tiles_cacheline[
                                    kdim_llstride_dimname
                                ][new_lambda_loc]
                                size_below = ld_bound_tiles_cacheline_below[
                                    kdim_llstride_dimname
                                ][lambda_loc]
                                assert size_above % size_below == 0
                                ratio = int(size_above / size_below)

                                d_ratio[new_lambda_loc] = ratio
                                d_sizes_below[new_lambda_loc] = size_below
                else:
                    raise ValueError(
                        f"sarcasm_set_assoc_model : issue with type of atom_below : {atom_below.type}"
                    )

                # DEBUG
                # print(f"{loop_lvl=} ({dim_atom=}) {array_name=} => {d_ratio=}")

                # We need to update llstride_dimname so that the previous iteration on this dim are taken into account?
                # Ex: [T(x,2) T(x,2)] : the outer loop have a stride twice the inner loop
                # => This is done through "d_sizes_below" that counts the stride coming from the inner loops
                #  (that needs to be multiplied to the base stride inside llstride_dimname)

                # Let's go
                dl_fp = _dl_fp_ratio_atom(
                    last_dl_fp,
                    llstride_dimname,
                    d_sizes_below,
                    comp,
                    array_name,
                    dim_atom,
                    d_ratio,
                    num_cache_set,
                )

            elif atom_below.type in [AtomType.SEQ]:
                dl_fp = _dl_fp_seq_atom(
                    last_dl_fp,
                    llstride_dimname,
                    ld_bound_tiles_cacheline_below,
                    dim_atom,
                    num_cache_set,
                )
            else:
                raise ValueError(
                    f"sarcasm_set_assoc_model::periodic_extra_cacheset_estimation_lvl : Unvalid atom {atom_below}"
                )

            # Commit !
            ddl_fp[array_name] = dl_fp

        # Commit !
        lddl_fp.append(ddl_fp)

    # DEBUG
    # print("\t - End First pass:")
    # print_lddl_fp(lddl_fp)

    # 4) Second pass - dimensions that are inside "l_dims_comp" and were not managed by the first pass
    # We need to use direct computation on them
    # NOTE/TODO for later: refine the detection criterion (ex: h+r, if r=0 at that point, a rotation works)
    # => This same criterion should be used in the first and second pass in a complementary way
    for loop_lvl in range(2, len(scheme) + 1):
        atom_below = scheme[loop_lvl - 1]

        # Recover the dfp from the current level
        ddl_fp = lddl_fp[loop_lvl - 1]

        for array_name in maccess_func_dim_name_coeff.keys():
            # dl_fp : [lambda_branch] [cache set]
            dl_fp = ddl_fp[array_name]
            ll_stride_dimname = maccess_func_dim_name_coeff[array_name]
            ld_bound_tiles_cacheline = dlld_bound_tiles_cacheline[array_name][
                loop_lvl - 1
            ]

            lldims_combi = d_combi_dims[array_name]

            if lldims_combi == []:
                # This array does not have a complicated array access
                # => We can skip it, since its dl_fp is already fully computed
                continue

            # We are in an array with (at least one) complicated access.
            # We need to enrich dl_fp at this level to take them into account

            # For each complicated accesses...
            for info_dim_combi in lldims_combi:
                ldims_combi = info_dim_combi[0]
                pos_access = info_dim_combi[1]

                # We recover the size of these access at that loop lvl by checking "dlld_bound_tiles_cacheline"
                dd_dim_size_combi = dict()
                for dim_combi in ldims_combi:
                    # We search "maccess_func_dim_name_coeff" for its index in dlld_btc
                    # maccess_func_dim_name_coeff : [arr_name] |--> list of ( stride [granularity in cache_line], dim_name )
                    kdim_combi = -1
                    for kdim_access in range(len(ll_stride_dimname)):
                        (stride, dim) = ll_stride_dimname[kdim_access]
                        if dim == dim_combi:
                            stride_dim_combi = stride
                            kdim_combi = kdim_access
                    # Check that the dim was correctly found
                    assert kdim_combi != (-1)

                    # dlld_bound_tiles_cacheline : [Array] [loop_lvl] [dim_access_array] [lambda_loc] => [num_iter]

                    # ... and using kdim_combi, we recover the sizes (along lambda_loc) of dim_combi for this loop lvl
                    d_sizes_kdimcombi = dlld_bound_tiles_cacheline[array_name][
                        loop_lvl - 1
                    ][kdim_combi]

                    # Commit
                    dd_dim_size_combi[dim_combi] = d_sizes_kdimcombi

                # DEBUG
                # print(f"{dd_dim_size_combi=}")
                # print(f"[Before shortcut detect - {array_name}/{pos_access}] {ld_bound_tiles_cacheline=}")

                # Shortcut: if all the sizes in dd_dim_size_combi are 1, we can skip
                b_all_sizes_at_1 = True
                for dim_combi in dd_dim_size_combi.keys():
                    d_dim_size_combi = dd_dim_size_combi[dim_combi]
                    for lambda_loc in d_dim_size_combi.keys():
                        size = d_dim_size_combi[lambda_loc]
                        if size > 1:
                            b_all_sizes_at_1 = False
                            break
                    if not b_all_sizes_at_1:
                        break
                if b_all_sizes_at_1:
                    continue

                # So, we do have some work to do here - we need to prepare for a rotation here

                # We prepare a call to dl_fp_direct_computation
                # So, we need to build a specific "ld_bound_tiles_cacheline" here ([dim] [lambda_loc])

                # Note: in the case of an access like "h*strx + r", the trick of capping the number of iterations
                # of "r" by "strx" (and) does not work due to issues on the final size of this access.
                # We need to separate that into 2 cases:
                #  - if R<strx : we have holes, and we need to have 2 entries in ld_bound_tiles_cachelines
                #  - if R>=strx : Merge the entries from h and r: H is set to 1 iter and R to combined range of iters
                # in ld_bound_tiles_cachelines
                #
                # When generalizing this condition, we obtain the following algorithm:
                #
                # We go from inner to outer stride dimension
                #  (note: stride can be equal between dimensions, ex: if we have "h+r")
                #  If we have a inner stride dim with more iteration than its ratio with the outer stride dim, merge them
                #  Else, keep them separate

                # Assertion: ldims_combi are consecutive dimensions in ld_bound_tiles_cacheline
                # ld_bound_tiles_cacheline_combi must be of the same size than ll_stride_dimname
                ld_bound_tiles_cacheline_combi = arrange_ld_btcl_combi_dims(
                    ll_stride_dimname, ld_bound_tiles_cacheline, ldims_combi
                )

                # ldim_ignore: all the dims which are not in "ldims_combi"
                ldim_ignore = []
                for l_stride_dimname in ll_stride_dimname:
                    dim_name = l_stride_dimname[1]
                    if dim_name not in ldims_combi:
                        ldim_ignore.append(dim_name)

                # DEBUG
                # print(f"{ll_stride_dimname=}")
                # print(f"{ld_bound_tiles_cacheline=}")
                # print(f"{ld_bound_tiles_cacheline_combi=}")

                # print(f"{ldim_ignore=}")

                # Let's go!
                dl_fp = dl_fp_direct_computation(
                    ll_stride_dimname,
                    ld_bound_tiles_cacheline_combi,
                    comp,
                    lcont_arr_order,
                    array_name,
                    prob_sizes,
                    num_cache_set,
                    cache_line_size,
                    ldim_ignore=ldim_ignore,
                    starting_dl_fp=dl_fp,
                )

            # Update ddl_fp with the new dl_fp
            ddl_fp[array_name] = dl_fp

    # DEBUG
    # print("\t - End Second pass:")
    # print_lddl_fp(lddl_fp)

    # 5) Optional global sanity check
    #   When checking the outerlevel, is the sum of all cache line coherent with the global footprint for each array?
    if b_sanity_check:
        dfootprint_elem = compute_footprint_nolambda(d_arrays_accs, comp, prob_sizes)
        dfootprint = dict()
        for array_name in dfootprint_elem:
            dfootprint[array_name] = int(
                math.ceil(dfootprint_elem[array_name] / cache_line_size)
            )

        # DEBUG
        # print(f"{dfootprint=}")

        for array_name in d_arrays_accs.keys():
            # Sum all the cache line at the last loop level, over all the cache sets
            sum_cacheline_arr = sum(lddl_fp[-1][array_name][""])

            # Compare it with the fp of the array (divided by the cache line size)
            fp_cacheline_arr = dfootprint[array_name]

            if sum_cacheline_arr != fp_cacheline_arr:
                print(
                    f"[array {array_name}] sum_cacheline_arr = {sum_cacheline_arr} | should be {fp_cacheline_arr}"
                )
                assert False  # If test fails, we have an internal bug !

    return lddl_fp


# Pretty-printer function - For debugging
def print_lddl_fp(lddl_fp: List[dict[str, DFP_LvlArray]]):
    # lddl_fp : [loop lvl] [array_name] [lambda_branch] [cache set]
    print("lddl_fp = [[[")
    for loop_lvl in range(len(lddl_fp)):
        print(f"* Loop Level {loop_lvl}:")
        for arr in lddl_fp[loop_lvl].keys():
            dl_val = lddl_fp[loop_lvl][arr]
            print(f"  - Array {arr} :")

            str_arr = ""
            for str_lambda_loc in dl_val.keys():
                str_arr = str_arr + f'    - "{str_lambda_loc}" : ['
                is_first = True
                for cache_set_id in range(len(dl_val[str_lambda_loc])):
                    if not is_first:
                        str_arr = str_arr + ", "
                    else:
                        is_first = False
                    str_arr = str_arr + str(dl_val[str_lambda_loc][cache_set_id])
                str_arr = str_arr + "]\n"
            print(str_arr, end="")
    print("]]]")
    return


# ====================================================================

# 3) Combining dfp across arrays and core (approximated) set-associative cache model algorithm


# [Main component function] Combine the detailled footprint of each array by directly summing them
#  WARNING: This is the part where the model is approximated, since the shift of each dfp between each other
#    moves in function of the surrounding loop indexes.
#  This can be viewed as computing the combination for the very first iteration of each surrounding loops
#
#  Input:
#   - lddl_fp : Detailled data footprint, freshly computed from "periodic_extra_cacheset_estimation_lvl"
#     [loop lvl] [array_name] [lambda_branch] [cache set]
#   - num_cache_set : number of cache sets in the cache  (can technically be deduced from lddl_fp, but save computation)
#
#  Output:
#   - lddl_fp : with an extra entry on "full_assoc_model::total_fieldname_fp"
def combine_dfp_exact_combi(
    lddl_fp: List[dict[str, DFP_LvlArray]], num_cache_set: int
) -> List[dict[str, DFP_LvlArray]]:
    for ddl_fp in lddl_fp:  # [Loop lvl]
        dl_fp_summed = dict()

        b_first_array = True
        for array_name in ddl_fp.keys():  # [Array name]
            dl_fp_arr = ddl_fp[array_name]

            # Init (on the first array_name)
            if b_first_array:
                for lambda_loc in dl_fp_arr.keys():
                    dl_fp_summed[lambda_loc] = [0] * num_cache_set
                b_first_array = False

            # Summation across all arrays
            for lambda_loc in dl_fp_arr.keys():  # [lambda_branch]
                l_fp_arr = dl_fp_arr[lambda_loc]

                for i in range(num_cache_set):  # [Cache set]
                    dl_fp_summed[lambda_loc][i] += l_fp_arr[i]

        # Commit
        ddl_fp[total_fieldname_fp] = dl_fp_summed

    return lddl_fp


DFP_Slice_CacheSet = List[
    dict[str, dict[str, int]]
]  # [loop lvl] [lambda_branch] [array_name]


# [Aux function] Reorder the elements of a ldl_fp in order to expose the cache sets
#    at the outer lvl of the data structure
#  Input:
#   - lddl_fp : [loop lvl] [array_name] [lambda_branch] [cache set]
#     (with "_Total" entry / from "combine_dfp_exact_combi")
#   - num_cache_set : number of cache sets in the cache
#     (can technically be deduced from ldl_fp, but save computation)
#  Output:
#   - lldd_fp : [cache set] [loop lvl] [lambda_branch] [array_name]
#   Note: need to be aligned with the "ldd_footprint" from the "full_assoc_model.py"
def _convert_lddl_to_lldd_fp(
    lddl_fp: List[dict[str, DFP_LvlArray]], num_cache_set: int
) -> List[DFP_Slice_CacheSet]:
    num_loop = len(lddl_fp)
    arr_keys = list(lddl_fp[0].keys())
    random_arr = arr_keys[0]

    # Prepare lldd_fp (fill with 0s)
    lldd_fp = []
    for i_cs in range(num_cache_set):
        ldd_fp = []
        for k_loop in range(num_loop):
            dd_fp = dict()
            for lambda_loc in lddl_fp[k_loop][random_arr].keys():
                d_fp = dict()
                for arr_name in arr_keys:
                    d_fp[arr_name] = 0
                # Commit
                dd_fp[lambda_loc] = d_fp
            # Commit
            ldd_fp.append(dd_fp)
        # Commit
        lldd_fp.append(ldd_fp)

    # Transfer the values from one data structure to the other
    for k_loop in range(num_loop):
        for arr_name in arr_keys:
            for lambda_loc in lddl_fp[k_loop][arr_name].keys():
                for i_cs in range(num_cache_set):
                    lldd_fp[i_cs][k_loop][lambda_loc][arr_name] = lddl_fp[k_loop][
                        arr_name
                    ][lambda_loc][i_cs]

    return lldd_fp


# Pretty-printer function - For debugging
def print_lldd_fp(lldd_fp: List[DFP_Slice_CacheSet]):
    print("lldd_fp = [[[")
    for i_cs in range(len(lldd_fp)):
        print(f"* Cache Set #{i_cs}:")
        for loop_lvl in range(len(lldd_fp[i_cs])):
            print(f"  - Loop Level {loop_lvl + 1}: {lldd_fp[i_cs][loop_lvl]}")
    print("]]]")

    return


# [Main function] Estimate the number of cache misses, using the sarcasm set-associative cache model
#
# Input:
# - scheme : the considered scheme
# - comp : the considered computation
# - prob_sizes : the problem sizes (including the stride dimensions)
# - lcont_arr_order : the order of contiguous allocation of the arrays
#
# [Cache properties]
#  - lcachesizes : list of cache sizes (in number of elements)
#  - lassoc_cache : cache associativity
#  - lnum_cache_set : list of number of cache sets
#  - cache_line_size : size of a cache line (in number of elements)
# [Note: all of these list must be of same size, and each element is a cache]
# [We also assume that the cache are in increasing size order (the number of cache set must be equal/increasing)]
# [  and that the number of cache sets are divisible between each other (small optim to reuse computation of "lddl_fp")]
#
# [Optional arguments]
#  - b_sanity_check : perform some extra sanity check to make bug tracking easier
#  - reuse_strat_full_assoc : control the reuse strategy of the underlying fully associative model
#
# Output:
#  - lcacheline_misses : For each cache [list], gives the estimated number of cache misses for this level of cache
def compute_cacheset_aware_comm_vol(
    scheme: List[Atom],
    comp: Computation,
    prob_sizes: dict[str, int],
    lcont_arr_order: List[str],
    lcachesizes: List[int],
    lassoc_cache: List[int],
    lnum_cache_set: List[int],
    cache_line_size: int,
    b_sanity_check: bool = False,
    reuse_strat_full_assoc: ReuseLoopStrat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE,
) -> List[int]:
    """
    Estimate the number of cache misses, using the sarcasm set-associative cache model

    Input:
     - scheme : the considered scheme
     - comp : the considered computation
     - prob_sizes : the problem sizes (including the stride dimensions)
     - lcont_arr_order : the order of contiguous allocation of the arrays

    [Cache properties]
     - lcachesizes : list of cache sizes (in number of elements)
     - lassoc_cache : cache associativity
     - lnum_cache_set : list of number of cache sets
     - cache_line_size : size of a cache line (in number of elements)
    [Note: all of these list must be of same size, and each element is a cache]
    [We also assume that the cache are in increasing size order (the number of cache set must be equal/increasing)]
    [  and that the number of cache sets are divisible between each other (small optim to reuse computation of "lddl_fp")]

    [Optional arguments]
     - b_sanity_check : perform some extra sanity check to make bug tracking easier
     - reuse_strat_full_assoc : control the reuse strategy of the underlying fully associative model

    Output:
     - lcacheline_misses : For each cache [list], gives the estimated number of cache misses for this level of cache
    """

    # 0) Check that the cache properties are coherent
    if b_sanity_check:
        assert len(lcachesizes) == len(lassoc_cache)
        assert len(lcachesizes) == len(lnum_cache_set)
        for i in range(len(lcachesizes)):
            assert (
                lcachesizes[i] == cache_line_size * lassoc_cache[i] * lnum_cache_set[i]
            )

        for i in range(len(lcachesizes) - 1):
            assert lnum_cache_set[i] <= lnum_cache_set[i + 1]
            assert lnum_cache_set[i + 1] % lnum_cache_set[i] == 0

    # 1) Computing the detailled footprint (each array + combine)
    # Optimization: we compute the lddl_fp of the biggest cache, and we recover the lddl_fp of the
    # smaller cache by folding them along the cache sets.
    #  If you don't want this optimization, call this function several times.
    ind_last_level_cache = len(lcachesizes) - 1
    num_cache_set_llc = lnum_cache_set[ind_last_level_cache]

    lddl_fp_llc = periodic_extra_cacheset_estimation_lvl(
        scheme,
        comp,
        prob_sizes,
        lcont_arr_order,
        num_cache_set_llc,
        cache_line_size,
        b_sanity_check=b_sanity_check,
    )
    lddl_fp_llc = combine_dfp_exact_combi(lddl_fp_llc, num_cache_set_llc)
    lldd_fp_lcc = _convert_lddl_to_lldd_fp(lddl_fp_llc, num_cache_set_llc)
    # lldd_fp_lcc : [cache set] [loop lvl] [lambda_branch] [array_name]

    # Quantities that will be common whatever the loop level or cache level considered
    num_loop = len(lldd_fp_lcc[0])
    arr_keys = list(
        lldd_fp_lcc[0][-1][""].keys()
    )  # Note: last lvl should no longer have lambda branches

    # 2) Apply the fully associative cache model on each cache sets
    l_estimated_cachemisses = []
    for cache_lvl in range(len(lcachesizes)):
        # a) Folding of "lldd_fp" to fit the cache level
        # + we need to multiply by cache_line_size to fit exactly the input of full_assoc_model

        # Check the factor we need to use to fold lldd_fp_lcc
        num_cache_set = lnum_cache_set[cache_lvl]
        ratio_n_cache_set = int(num_cache_set_llc / num_cache_set)

        # Note: if ratio_n_cache_set==1, then it might be useful to optimize full_assoc_model
        # to prevent them to redo a division by cache_line_size at the middle

        # We fold "lldd_fp_lcc" by a factor "ratio_n_cache_set"
        lldd_fp = []
        # First "num_cache_set" elements: deep copy of lldd_fp_lcc
        for i_cs in range(num_cache_set):
            ldd_fp = []
            for k_loop in range(num_loop):
                dd_fp = dict()
                for lambda_loc in lldd_fp_lcc[i_cs][k_loop].keys():
                    d_fp = dict()
                    for arr_name in arr_keys:
                        # Copy
                        d_fp[arr_name] = (
                            lldd_fp_lcc[i_cs][k_loop][lambda_loc][arr_name]
                            * cache_line_size
                        )
                    # Commit
                    dd_fp[lambda_loc] = d_fp
                # Commit
                ldd_fp.append(dd_fp)
            # Commit
            lldd_fp.append(ldd_fp)

        # Rest of the elements: we fold on the "num_cache_set" elements
        for i_cs in range(num_cache_set, num_cache_set_llc):
            for k_loop in range(num_loop):
                for lambda_loc in lldd_fp_lcc[i_cs][k_loop].keys():
                    for arr_name in arr_keys:
                        folded_i_cs = i_cs % num_cache_set
                        val = lldd_fp_lcc[i_cs][k_loop][lambda_loc][arr_name]
                        lldd_fp[folded_i_cs][k_loop][lambda_loc][arr_name] += (
                            val * cache_line_size
                        )

        # At that point "lldd_fp" is the correct detailled footprint for the considered cache level

        # DEBUG
        # print_lldd_fp(lldd_fp)

        # b) Using the fully associative cache model on each cache sets
        l_num_comm_across_cache_set = []  # Detail of the number of cache misses per cache sets
        for i_cs in range(num_cache_set):
            # We have spent so many time and functions to prepare lldd_fp for this exact line
            ldd_footprint = lldd_fp[i_cs]

            # DEBUG
            # if (i_cs==0):
            # print(f"{ldd_footprint=}")

            lcachesizes_fa_model = [lassoc_cache[cache_lvl] * cache_line_size]
            lnum_comm = full_assoc_model_with_fp(
                scheme,
                ldd_footprint,
                lcachesizes_fa_model,
                cache_line_size,
                comp,
                reuse_strat_full_assoc,
            )
            assert len(lnum_comm) == 1

            # Commit
            l_num_comm_across_cache_set.append(lnum_comm[0])

        # DEBUG
        # print(f"{l_num_comm_across_cache_set=}")

        # Sum all contributions to get the full estimation
        est_cachemisses = sum(l_num_comm_across_cache_set)

        # Commit
        l_estimated_cachemisses.append(est_cachemisses)

    return l_estimated_cachemisses
