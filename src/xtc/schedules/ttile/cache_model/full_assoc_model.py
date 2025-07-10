#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from typing import List, Dict, Tuple, Optional
from enum import Enum

from xtc.schedules.ttile.scheme import (
    Atom,
    AtomType,
    build_scheme_from_str,
    get_sizes_scheme,
)
from xtc.schedules.ttile.scheme import (
    stringify_lambda_choice,
    recover_all_branchid_from_stringified,
    recover_branchid_from_stringified,
    get_max_num_lambda_branch,
)
from xtc.schedules.ttile.scheme import (
    get_list_dims_str_lambda_loc,
    is_stringify_subset,
    remove_dim_from_stringified,
    identify_subset_in_list_stringify,
)
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.computation import (
    get_array_accesses,
    get_ldims_computation,
    get_default_sizes,
    get_reuse_dims,
    get_ldims_stride_computation,
)
from xtc.schedules.ttile.computation import compute_footprint


# This Python file contains the full-associativity model, to estimate a volume of comms

# ====================================================================

# 1) Footprint computation per level of the scheme

LevelFootprint = dict[str, dict[str, int]]

# Used in the footprint map to store the sum of the footprint of all arrays
total_fieldname_fp = "_Total"


# [Aux function] Rectify the footprint at one level to take into account
# the cache line (in respect to the inner dimension of arrays)
def rectify_fp_cache_line(
    ddfootprint: LevelFootprint,
    d_arrays_accs: dict[str, Tuple[List[str], int]],
    d_lsizes: dict[str, List[int]],
    cache_line_size: int,
) -> LevelFootprint:
    # print(ddfootprint)
    # print(d_arrays_accs)
    # print(d_lsizes)

    for str_lambda_branch in ddfootprint.keys():
        dfootprint = ddfootprint[str_lambda_branch]

        for arr in dfootprint.keys():
            fp_val = dfootprint[arr]

            # Recover the innermost array access
            inner_dim_arr = d_arrays_accs[arr][0][-1]
            assert (
                "+" not in inner_dim_arr
            )  # Else, we have an issue with cache line alignment

            # Check the stringified string to get the lambda branch id on that dim
            # If no lambda dim, then 0
            numbr_inner_dim = recover_branchid_from_stringified(
                str_lambda_branch, inner_dim_arr
            )

            # Recover the size of the tile along this inner dim
            size_inner_dim_arr = d_lsizes[inner_dim_arr][numbr_inner_dim]

            # Check if it is aligned with cache lines
            if size_inner_dim_arr % cache_line_size != 0:
                # We need to rectify the footprint of this array: extra cells are used
                # (cache lines not completly exploited)
                n_size = (
                    int(size_inner_dim_arr / cache_line_size) + 1
                ) * cache_line_size

                # In the footprint computation, replace size_inner_dim_arr par n_size
                # Since the access is simple, we just need to divide/multiply them
                nfp_val = int(fp_val / size_inner_dim_arr * n_size)

                # Commit
                dfootprint[arr] = nfp_val

    return ddfootprint


# Compute the footprint for each level
# Inputs (similar to Ioopt):
# d_full_sizes : full size of the problem (dict [dim_name] |--> int). Used only to complete the "stride" dims values.
#   d_arrays_accs: dictionnary [name_array] |--> [list of 2 elements, elem0 = list of accesses, elem1 = number of accesses]
# prog_dims: list of dimensions to the iteration space
# cache_line_size : in number of elements (ex: float32)
# Output:
#   ldd_footprint: list (loop level) of dictionnary
# For a loop lvl: [stringified lambda choice] |--> ( [name_array] |--> footprint value )
#   The innermost dict have an additional entry:
# [total_fieldname_fp] |--> Sum of the previous list of footprint values
def compute_footprint_for_each_level(
    scheme: List[Atom],
    comp: Computation,
    d_full_sizes: dict[str, int],
    d_arrays_accs: dict[str, Tuple[List[str], int]],
    prog_dims: List[str],
    cache_line_size: int,
) -> List[LevelFootprint]:
    # Check that the total_fieldname_fp is not taken
    assert total_fieldname_fp not in d_arrays_accs.keys()

    ldims_stride = get_ldims_stride_computation(comp)

    ldd_footprint = []
    # For each level of the scheme
    for loop_lvl in range(1, len(scheme) + 1):
        # Part of the scheme corresponding to the current level
        pref_scheme = scheme[:loop_lvl]

        # Get the sizes and footprint of the level
        # d_lsizes : size of the current tile of the scheme
        # "name_dim" |--> [lsizes]
        d_lsizes = get_sizes_scheme(pref_scheme)

        # Add the stride (so that we perform the right computation)
        for dim_stride in ldims_stride:
            if dim_stride in d_full_sizes:
                d_lsizes[dim_stride] = [d_full_sizes[dim_stride]]

        d_lsizes_default = get_default_sizes(comp)
        d_lsizes = d_lsizes_default | d_lsizes

        # DEBUG
        # print(f"Loop lvl {loop_lvl} - {d_lsizes=}")

        # ddfootprint: [stringified lambda choice] |--> ( [name_array] |--> footprint value )
        ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

        # DEBUG
        # print(ddfootprint)

        # Take into account the cache line alignment issues
        # Augment ddfootprint with elements on the same cache lines even if they are not used
        ddfootprint = rectify_fp_cache_line(
            ddfootprint, d_arrays_accs, d_lsizes, cache_line_size
        )

        # Enrich dfootprint with the sum of all arrays
        for strbr in ddfootprint.keys():
            dfootprint = ddfootprint[strbr]

            # Sum across all arrays
            total_fp = 0
            for arr in dfootprint.keys():
                fp_val = dfootprint[arr]
                total_fp += fp_val

            # Add total_fp to dfootprint
            dfootprint[total_fieldname_fp] = total_fp

        # Commit
        ldd_footprint.append(ddfootprint)

    return ldd_footprint


# Pretty printer function for ldd_footprint (output of compute_footprint_for_each_level)
def print_ldd_footprint(
    scheme: List[Atom],
    ldd_footprint: List[LevelFootprint],
    o_cache_line_size: Optional[int] = None,
):
    for i in range(len(ldd_footprint)):
        print(f"- Loop lvl {i + 1} - Last scheme atom: {scheme[i]}:")

        dd_footprint = ldd_footprint[i]
        for k, d_fp in dd_footprint.items():
            if o_cache_line_size != None:
                d_fp_cl = dict()
                for arr, velem in d_fp.items():
                    d_fp_cl[arr] = int(velem / o_cache_line_size)
                print(f'    LBranch "{k}" -> {d_fp_cl} [in CL]')
            else:
                print(f'    LBranch "{k}" -> {d_fp}')

    return


# ====================================================================
# 2) Find specific loop levels (saturation + reuse/num_repet for each array)


# [Aux function] Add a lambda_id (branch # "kbr") for dimension "dim", that is
# not already present in str_lambda_loc
def extend_str_lambda_loc(str_lambda_loc: str, dim: str, kbr: int, comp: Computation):
    # Recover all info from str_lambda_loc
    d_dim_brid = recover_all_branchid_from_stringified(str_lambda_loc)
    assert dim not in d_dim_brid

    # Add the new info
    d_dim_brid[dim] = kbr

    # Put things together
    ldims_ref = get_ldims_computation(comp)
    str_ext_lambda_loc = stringify_lambda_choice(ldims_ref, d_dim_brid)

    return str_ext_lambda_loc


# [Aux function] Find the saturation loop level, i.e. the loop level where the global footprint
#    across all arrays goes over the cachesize.
#
#  Due to the presence of lambdas, this value might change for each lambda branches, thus
#    we return a dictionnary "d_sat_loop_lvl: [str_lambda_branch] |---> loop_lvl"
#
#  This "loop_lvl" corresponds to the level directly ABOVE saturation.
#  If no saturation happens on any branches, then "loop_lvl" is set to "None" (don't forget to catch that case).
#    Note that because the scheme is complete, if saturation happens at any branch,
#      the whole scheme is saturated at top loop lvl.
#  If a saturation happens in the middle of a lambda branch, we need to keep track of different
#      saturation levels
#
# Input:
#   ldd_footprint (from "compute_footprint_for_each_level")
#   cachesize : size of the cache in number of element
# Output:
#   d_sat_loop_lvl: [str_lambda_branch] |---> loop_lvl (or "None")
def find_saturation_level(
    ldd_footprint: List[LevelFootprint], cachesize: int, comp: Computation
) -> dict[str, Optional[int]]:
    num_loop_level = len(ldd_footprint)

    # Quick sanity check on ldd_footprint: no lambda branches remaining at the last level?
    # assert( len( ldd_footprint[num_loop_level-1][... str_branch ...][total_fieldname_fp] ) == 1 )

    # Initialization of our output: first level does not have saturation
    d_sat_loop_lvl: dict[str, Optional[int]] = dict()
    d_sat_loop_lvl[""] = None

    # IMPORTANT: Note that the key of d_sat_loop_lvl will not always correspond to
    # the key of dd_footprint, due to the presence of new TL/UL and Seq, that might be interleaved.
    # => The key of d_sat_loop_lvl will not diminish in size when encountering a "Seq",
    # in order to corresponds to each branches of the "concrete" C program generated at the end.
    # This also used the fact that there is only a single "Seq" per dimension.
    #
    # Exemples:
    #   [ TL(X, ..) TL(Y, ..) Seq(X) Seq(Y) ] => d_sat_loop_lvl will have 4 keys at the end (ex: "X1,Y2")
    #   [ TL(X, ..) Seq(X) TL(Y, ..) Seq(Y) ] => d_sat_loop_lvl will have the same 4 keys at the end,
    # even if the dd_footprint only see "X1"/"X2" and "Y1"/"Y2".

    # For each loop level (in between the atoms of a scheme)
    for loop_lvl in range(1, num_loop_level + 1):
        # DEBUG
        # print(f"[Saturation] Start loop level {loop_lvl} => {d_sat_loop_lvl=}")

        # Retrieving the total footprint across all arrays for this level
        dd_footprint = ldd_footprint[loop_lvl - 1]

        # We check the status of saturation on each keys of dd_footprint
        d_sat_local_lvl = dict()  # [lambda_branch id of dd_footprint] |---> Boolean
        for str_lambda_loc, dfp_val in dd_footprint.items():
            if dfp_val[total_fieldname_fp] > cachesize:
                d_sat_local_lvl[str_lambda_loc] = True

        # DEBUG
        # print(f"[Saturation] Saturation info at loop lvl {loop_lvl} => {d_sat_local_lvl}")

        # Now, we integrate these new infos with "d_sat_loop_lvl"

        # 1) We need to check if we need to duplicate the entries of "d_sat_loop_lvl"
        # (happens when we encounter a TL/UL on a new dimension)
        first_str_lambda_loc_dfp = list(dd_footprint.keys())[0]
        ldims_dd_footprint = get_list_dims_str_lambda_loc(first_str_lambda_loc_dfp)

        first_str_lambda_loc_dsat = list(d_sat_loop_lvl.keys())[0]
        ldims_d_sat_loop_lvl = get_list_dims_str_lambda_loc(first_str_lambda_loc_dsat)

        for dim in ldims_dd_footprint:
            if dim not in ldims_d_sat_loop_lvl:
                # We need to extend "d_sat_loop_lvl" along "dim"

                # a) Recover the number of branches needed
                num_lambda_br_dim = get_max_num_lambda_branch(
                    list(dd_footprint.keys()), dim
                )

                nd_sat_loop_lvl = dict()
                for str_lambda_loc, opt_loop_lvl in d_sat_loop_lvl.items():
                    # Duplicate the info according to the new branches
                    for kbr in range(num_lambda_br_dim):
                        n_str_lambda_loc = extend_str_lambda_loc(
                            str_lambda_loc, dim, kbr, comp
                        )
                        nd_sat_loop_lvl[n_str_lambda_loc] = opt_loop_lvl

                # Commit
                d_sat_loop_lvl = nd_sat_loop_lvl

                # Can only trigger on a single dimension (UL/TL only on 1 dim)
                break

        # Note: if all dims of dd_footprint where encountered in d_sat_loop_lvl, then no change needed

        # DEBUG
        # print(f"	- Adapt d_sat_loop_lvl done => {d_sat_loop_lvl=}")

        # 2) We integrate the info of "d_sat_local_lvl" to see if there are branches
        # of "d_sat_loop_lvl" that just got saturated (i.e., currently at null, but got sat)
        for str_lambda_loc_dsat, opt_loop_lvl in d_sat_loop_lvl.items():
            # This branch is already saturated: nothing to add
            if opt_loop_lvl != None:
                continue

            # Is this branch saturated now?
            # => We check if there is a "str_lambda" of "d_sat_local_lvl" that is a subset of "str_lambda_loc_dsat"
            for str_lambda_loc_local_lvl, b_is_sat in d_sat_local_lvl.items():
                if not b_is_sat:
                    continue

                if is_stringify_subset(str_lambda_loc_local_lvl, str_lambda_loc_dsat):
                    # We have a new saturation here (since "b_is_sat" is True)
                    # => We update "d_sat_loop_lvl"
                    d_sat_loop_lvl[str_lambda_loc_dsat] = loop_lvl

        # d_sat_loop_lvl is not up-to-date with the local level information

        # DEBUG
        # print(f"[Saturation] Merged sat info (updated after loop lvl {loop_lvl}) => {d_sat_loop_lvl}")

    return d_sat_loop_lvl


# For debugging
_debug_reuse_full_assoc = False


# Class to list the different loop reuse strategies in "compute_num_repet_reuse"
class ReuseLoopStrat(Enum):
    NO_LOOP_REUSE = 1
    MAX1_LOOP_REUSE = 2
    UNLIMITED_LOOP_REUSE = 3

    def __str__(self):
        match self:
            case ReuseLoopStrat.NO_LOOP_REUSE:
                return "No loop lvl reuse"
            case ReuseLoopStrat.MAX1_LOOP_REUSE:
                return "Max 1 loop lvl reuse"
            case ReuseLoopStrat.UNLIMITED_LOOP_REUSE:
                return "Unlimited loop lvl reuse"
            case _:
                raise ValueError("Unknown ReuseLoopStrat in ReuseLoopStrat::__str__")


# [Aux function] Find the number of repetition of a given branch of a loop
#  Inputs:
#    - scheme = the scheme we examine
#    - arr = the name of the array
#    - sat_loop_lvl = the saturation loop lvl (division between loop lvl, "i" is between scheme[i] and scheme[i+1])
#    - str_lambda_loc = the lambda branch we are currently examining
#    - comp = the considered computation
#    - reuse_strat = a "ReuseLoopStrat" (cf enum above) that control how we manage the reuse coming
# from the loops above saturation level.
#  Output:
#    - sat_loop_lvl = the reuse loop lvl (should be equal or above sat_loop_lvl)
#    - num_repet = number of time the accesses of "acc" of the tile at lvl "sat_loop_lvl" in "scheme" is repeated.
def compute_num_repet_reuse(
    scheme: List[Atom],
    arr: str,
    sat_loop_lvl: int,
    str_lambda_loc: str,
    comp: Computation,
    reuse_strat: ReuseLoopStrat,
) -> Tuple[int, int]:
    # 1) From where do we start counting? (reuse strat)
    if sat_loop_lvl == len(scheme):
        # We are already at top level: no loop above to consider reuse on
        reuse_loop_lvl = sat_loop_lvl
    else:
        # What reuse strat should we consider?
        d_reuse = get_reuse_dims(comp)
        ldim_reuse_arr = d_reuse[arr]

        if reuse_strat == ReuseLoopStrat.NO_LOOP_REUSE:
            reuse_loop_lvl = sat_loop_lvl
        elif reuse_strat == ReuseLoopStrat.MAX1_LOOP_REUSE:
            # Check the atom right below saturation level
            # Note: if no reuse during saturation level then assume data is evicted before end of iterations
            #  If reuse during saturation level, then same block of data reuse again and again so stay in cache
            # ===> We really need to check starting from right below the saturation level.
            current_atom = scheme[sat_loop_lvl - 1]

            # These should be removed (preprocess of scheme) before going in here
            assert current_atom.type != AtomType.HOIST
            dim = current_atom.dim

            if dim in ldim_reuse_arr:
                reuse_loop_lvl = sat_loop_lvl + 1
            else:
                reuse_loop_lvl = sat_loop_lvl
        elif reuse_strat == ReuseLoopStrat.UNLIMITED_LOOP_REUSE:
            reuse_loop_lvl = sat_loop_lvl
            while reuse_loop_lvl <= len(scheme):
                # Check the atom right above the current reuse level
                current_atom = scheme[reuse_loop_lvl - 1]

                # These should be removed (preprocess of scheme) before going in here
                assert current_atom.type != AtomType.HOIST
                dim = current_atom.dim

                if dim in ldim_reuse_arr:
                    reuse_loop_lvl = reuse_loop_lvl + 1
                else:
                    break
        else:
            raise ValueError(f"Unknown loop reuse strat {reuse_strat}")

    # DEBUG
    if _debug_reuse_full_assoc:
        print(
            f"        Array {arr} (sat lvl = {sat_loop_lvl}) => Reuse lvl = {reuse_loop_lvl}"
        )

    # 1B) fp_reuse_loop_lvl = reuse_loop_lvl, except that we stop at a "Seq"
    # (else, the value taken for the footprint is wrong, since we are taking the fp from several branches)
    fp_reuse_loop_lvl = None
    for ilvl in range(sat_loop_lvl, reuse_loop_lvl + 1):
        current_atom = scheme[ilvl - 1]

        if current_atom.type == AtomType.SEQ:
            fp_reuse_loop_lvl = ilvl - 1

    if fp_reuse_loop_lvl == None:
        fp_reuse_loop_lvl = reuse_loop_lvl

    # 2) Counting in the scheme, starting from reuse_loop_lvl
    num_repet = 1
    for ilvl in range(reuse_loop_lvl, len(scheme)):
        # Recover the multiplier here
        current_atom = scheme[ilvl]

        # This case is not managed
        # Note that it could be through lambda ?
        assert current_atom.type != AtomType.TILE_PARTIAL

        if current_atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
            # We recover the right ratio in the list of ratio
            dim_atom = current_atom.dim
            i_br = recover_branchid_from_stringified(str_lambda_loc, dim_atom)
            ratio = current_atom.lratios[i_br]
        elif current_atom.type in [AtomType.SEQ]:
            # No extra repetition
            ratio = 1
        elif current_atom.type in [
            AtomType.VECT,
            AtomType.UNROLL,
            AtomType.TILE,
            AtomType.TILE_PARAL,
        ]:
            ratio = current_atom.ratio
        else:
            raise ValueError(
                f"full_assoc_model::compute_num_repet_reuse : unrecognized atom type {current_atom.type}"
            )

        # Accumulate
        num_repet = num_repet * ratio
    return (fp_reuse_loop_lvl, num_repet)


# ====================================================================
# 3) Main function

# For debugging
_debug_full_assoc = False


# MAIN FUNCTION - Compute the number of cache misses, given a scheme, a computation and a list of cache sizes
# The output will have the same number of elements of "lcachesizes".
# Inputs:
#   - "scheme" : the scheme whose number of cache misses we want to estimate
# - "ldd_footprint" : the footprint of the scheme at each loop levels
#   - "lcachesizes" : the sizes of the cache we consider.
# - "cache_line_size" : the size of a cache line (in elements)
# - "comp" : the considered computation
# Output:
# - "lnum_comms" : list of number of cache miss, each element corresponds to an entry of lcachesizes
def full_assoc_model_with_fp(
    scheme: List[Atom],
    ldd_footprint: List[LevelFootprint],
    lcachesizes: List[int],
    cache_line_size: int,
    comp: Computation,
    reuse_strat: ReuseLoopStrat,
) -> List[int]:
    d_arrays_accs = get_array_accesses(comp)
    prog_dims = get_ldims_computation(comp)

    # For each level of cache...
    lnum_comms = []
    for cachesize in lcachesizes:
        # DEBUG
        if _debug_full_assoc:
            print(f"Entering cachesize = {cachesize}")

        # Compute the saturation levels
        d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

        # DEBUG
        if _debug_full_assoc:
            print(f"  Saturation levels = {d_sat_loop_lvl}")

        # Check if we have saturation:
        # - No saturation = All values of "d_sat_loop_lvl" are "None"
        # - Saturation = All values of "d_sat_loop_lvl" are not "None"
        # - Because the scheme is complete, there are no middle ===> We check this assertion
        b_has_saturated = False
        b_has_not_saturated = False
        for sat_lvl in d_sat_loop_lvl.values():
            if sat_lvl == None:
                b_has_not_saturated = True
            else:
                b_has_saturated = True
        assert b_has_saturated == (not b_has_not_saturated)

        #  Here is the algo to compute the number of cache misses, with the choice made in term of reuse between iterations
        #    of the same loop level.
        #
        #  A) If we do not have lambda (easy case - to have an intuition of what happens before going into the full generality)
        #    No saturation: \sum_{arr} footprint[ last_lvl ][arr]   (cf Case 1)
        #    Saturation:  \sum_{arr}  num_repet(arr, sat_lvl) * footprint[ sat_lvl][arr]
        # => APPROXIMATION HERE: We assume that the reuse between iterations at saturation level
        # happens only between consecutive iterations.
        #
        #  B) Case with lambda/Seq (this is what has to be implemented here):
        #    The trick is to consider "Seq" as a scalar dimension, with non-uniform tiles below it (its branches)
        #      (i) Saturation at "Seq" or above it:
        #        No special case required, since this part is covered by the footprint computation/
        #      (ii) Saturation below "Seq":
        #        We just have to sum the contribution of all the sub-branches, by computing their "num_repet"
        #          (which are different for each branches) and sum all of them across all branches
        #        ==> APPROXIMATION HERE: we assume no reuse between the different branches of a "Seq"
        #      (iii) Mix of saturated/non-saturated branches below "Seq":
        #        There are no reuse possible on the "non-saturated" branches, due to the "saturated" ones
        #          => By setting the saturation level of the non-seq branches to the level below "Seq"
        #            we fall back to case B.(ii) and have the correct expression
        #
        # About reuse, this is taken care in the computation of "num_repet" (much simpler that the alternative of
        #   raising the saturation level).
        #
        # ===> General formula is (with adaptation) basically a sum over all arrays and lambda branches of
        # "num_repet(arr, sat_lvl) * footprint(sat_lvl)"
        # with some adaptation on "num_repet" (reuse management) + sat_lvl (case B(iii)) to make it work

        # Case 1: no saturation happens (i.e., scheme is cache resident)
        # => There are only cold misses, and no eviction
        if b_has_not_saturated:
            # DEBUG
            if _debug_full_assoc:
                print("  The scheme has not saturated - only cold misses here.")

            # Number of cache miss is the footprint
            n_cache_misses = ldd_footprint[-1][""][total_fieldname_fp]

            assert n_cache_misses % cache_line_size == 0
            n_cache_misses = int(n_cache_misses / cache_line_size)

            # Commit
            lnum_comms.append(n_cache_misses)
            continue

        # DEBUG
        if _debug_full_assoc:
            print(
                "  Saturation happened - updating saturation levels with special cases"
            )

        # For typing, d_sat_loop_lvl to be "de-Optional-ed"
        d_sat_loop_lvl_unsat: dict[str, int] = dict()
        for str_lambda_loc, sat_loop_lvl in d_sat_loop_lvl.items():
            assert sat_loop_lvl != None
            d_sat_loop_lvl_unsat[str_lambda_loc] = sat_loop_lvl

        # Summation is a bit more complex than advertised, since we do not want to sum a branch twice
        # => Ex: If there is a "Seq" below saturation level, that part will appear twice in d_sat_loop_lvl
        # => We need to adapt/filter them to have a "straight" dict on which we can perform our summation
        #
        # Criteria:
        # - [case B(iii)] If we have a saturation level of a branch right above a Seq,
        # and there are other branches (along this dimension) with lower saturation level
        # THEN? we need to lower this saturation level by 1.
        # => Cannot trigger a cascade of lowering (if there is also a Seq below), since that branch was already a Nsat
        # - [Seq below saturation] If we have a saturation level above a Seq, only consider one branch of this Seq.
        # => Avoid counting these iterations multiple times.
        n_d_sat_loop_lvl: dict[str, int] = dict()
        for str_lambda_loc, sat_loop_lvl in d_sat_loop_lvl_unsat.items():
            # Atoms that are below the saturation loop lvl
            pref_scheme = scheme[:sat_loop_lvl]

            # Check for Seq in these atoms
            b_keep_branch = True
            for k in range(len(pref_scheme)):
                atom = pref_scheme[k]
                if atom.type == AtomType.SEQ:
                    dim_seq = atom.dim
                    if k == len(pref_scheme) - 1:
                        # Case B(iii)
                        # The Seq is at saturation: we just need to update sat_loop_lvl to push it right below "Seq"
                        sat_loop_lvl = sat_loop_lvl - 1
                    else:
                        # The Seq is well below saturation: we check the branchid along its dimension
                        # and if it is not "0", we do not keep it
                        # => By construction, the only branch that should be kept is the "0" one
                        brid = recover_branchid_from_stringified(
                            str_lambda_loc, dim_seq
                        )
                        if brid != 0:
                            b_keep_branch = False
                            break
                        else:
                            # Keep the entry, but to correspond to the key of "ldd_footprint",
                            #  we need to remove dim_seq from str_lambda_loc
                            str_lambda_loc = remove_dim_from_stringified(
                                str_lambda_loc, dim_seq, prog_dims
                            )

            # Commit
            if b_keep_branch:
                n_d_sat_loop_lvl[str_lambda_loc] = sat_loop_lvl

        # New version replace the previous one
        d_sat_loop_lvl_unsat = n_d_sat_loop_lvl

        # DEBUG
        if _debug_full_assoc:
            print(f"  Saturation levels updated = {d_sat_loop_lvl_unsat}")

        # Case 2: saturation does happen at some point
        n_cache_misses = 0
        for arr in d_arrays_accs.keys():
            # DEBUG
            if _debug_full_assoc:
                print(f"  For array {arr}:")

            n_cache_misses_arr = 0

            # The branches of d_sat_loop_lvl_unsat are now the ones we want to sum over
            for str_lambda_loc in d_sat_loop_lvl_unsat.keys():
                # DEBUG
                if _debug_full_assoc:
                    print(f'    - For lambda location "{str_lambda_loc}":')

                # Getting the num_repet
                sat_loop_lvl_br = d_sat_loop_lvl_unsat[str_lambda_loc]
                (reuse_loop_lvl_br, num_repet_br) = compute_num_repet_reuse(
                    scheme, arr, sat_loop_lvl_br, str_lambda_loc, comp, reuse_strat
                )

                # Note we prefer to take the footprint of reuse dimension at the reuse lvl,
                #   rather than the saturation level, even if in 99% of the cases they should be
                #   identical. The exception is due to the presence of "small dimensions"
                #   (cf the reuse dimensions of conv2d for example), which still grows the footprint
                # during reuse, but in a small manner (small enough to consider it a "quasi-reuse" dimension)

                dd_footprint = ldd_footprint[reuse_loop_lvl_br - 1]

                # DEBUG
                # print(f"sat_loop_lvl_br-1 = {sat_loop_lvl_br-1}")
                # print(f"  => Available keys = { list(ldd_footprint[sat_loop_lvl_br-1].keys())}")
                # print(f"str_lambda_loc = {str_lambda_loc}")

                # Quick adaptation of the stringified lambda branch location
                lkeys_ddfp = list(dd_footprint.keys())
                if str_lambda_loc not in lkeys_ddfp:
                    lstr_lambda_loc_fp = identify_subset_in_list_stringify(
                        lkeys_ddfp, str_lambda_loc
                    )
                    assert (
                        len(lstr_lambda_loc_fp) == 1
                    )  # Should be satisfied by construction
                    str_lambda_loc_fp = lstr_lambda_loc_fp[0]
                else:
                    str_lambda_loc_fp = str_lambda_loc

                # Getting the footprint at saturation lvl
                fp_val_br = dd_footprint[str_lambda_loc_fp][arr]

                # Convert into number of cache lines for the cache misses
                assert fp_val_br % cache_line_size == 0
                ncm_val_br = int(fp_val_br / cache_line_size)

                # Contribution of this branch of the program to the number of cache misses
                n_cache_misses_br = ncm_val_br * num_repet_br
                n_cache_misses_arr += n_cache_misses_br

                # DEBUG
                if _debug_full_assoc:
                    print(
                        f"      {ncm_val_br} [FP_CL] * {num_repet_br} [REP] = {n_cache_misses_br} [CM]"
                    )

            # DEBUG
            if _debug_full_assoc:
                print(f"  => Total for array {arr} = {n_cache_misses_arr}")

            # Commit the cache misses of this array
            n_cache_misses += n_cache_misses_arr

        # DEBUG
        if _debug_full_assoc:
            print(f"  => Total cache misses = {n_cache_misses}")

        # And commit for the cache level
        lnum_comms.append(n_cache_misses)

    return lnum_comms


# Wrapper function - since we need to have the ldd_footprint as the input
#  Call this function to use the model on a fresh scheme.
#  If you are currently at the middle of the set-assoc model, call "full_asssoc_model_with_fp" instead
def compute_full_assoc_cache_misses(
    scheme: List[Atom],
    d_full_sizes: dict[str, int],
    lcachesizes: List[int],
    cache_line_size: int,
    comp: Computation,
    reuse_strat: ReuseLoopStrat,
) -> List[int]:
    # We factorize the footprint computation for all cache levels
    d_arrays_accs = get_array_accesses(comp)
    prog_dims = get_ldims_computation(comp)
    ldd_footprint = compute_footprint_for_each_level(
        scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size
    )

    # DEBUG
    if _debug_full_assoc:
        print("[Full assoc cache model] ldd_footprint =")
        print_ldd_footprint(scheme, ldd_footprint, o_cache_line_size=cache_line_size)

    # Call the main function
    lnum_comms = full_assoc_model_with_fp(
        scheme, ldd_footprint, lcachesizes, cache_line_size, comp, reuse_strat
    )
    return lnum_comms
