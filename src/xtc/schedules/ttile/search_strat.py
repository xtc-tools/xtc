#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import math
import random
from typing import List, Tuple, Optional, Any
from typing_extensions import override

from xtc.schedules.ttile.scheme import Atom
from xtc.schedules.ttile.computation import Computation
from xtc.schedules.ttile.archi import Archi

import xtc.schedules.ttile.scheme as scheme
from xtc.schedules.ttile.microkernel import (
    is_compatible_microkernel,
    convert_microkernel_info_to_microkernel,
)
from xtc.schedules.ttile.microkernel import (
    load_microkernel_info,
    sort_microkernels_by_pperf,
    is_faster_microkernel_than,
)

# Ttile microkernel-based search strategy


# Class to store the information about a strategy using a single microkernel
class Microkernel_strat_single:
    d_mickern: dict[str, int]
    perf: float

    def __init__(self, d_mickern_info: dict[str, Any]):
        self.d_mickern = dict()
        for dim_name in d_mickern_info:
            if dim_name != "peak_perf":
                self.d_mickern[dim_name] = d_mickern_info[dim_name]
        self.perf = d_mickern_info["peak_perf"]

    @override
    def __str__(self):
        return f"Single({self.d_mickern}|perf = {self.perf})"


# Class to store the information about a strategy using a lambda of microkernel
class Microkernel_strat_lambda:
    d_mickern_1: dict[str, int]
    num_repet_1: int
    d_mickern_2: dict[str, int]
    num_repet_2: int

    lambda_dim: str

    perf: float

    def __init__(
        self,
        num_repet_1: int,
        d_mickern_info_1: dict[str, int],
        num_repet_2: int,
        d_mickern_info_2: dict[str, int],
        lambda_dim: str,
    ):
        self.num_repet_1 = num_repet_1
        self.num_repet_2 = num_repet_2
        self.lambda_dim = lambda_dim

        self.d_mickern_1 = dict()
        for dim_name in d_mickern_info_1:
            if dim_name != "peak_perf":
                self.d_mickern_1[dim_name] = d_mickern_info_1[dim_name]
        perf1 = d_mickern_info_1["peak_perf"]

        self.d_mickern_2 = dict()
        for dim_name in d_mickern_info_2:
            if dim_name != "peak_perf":
                self.d_mickern_2[dim_name] = d_mickern_info_2[dim_name]
        perf2 = d_mickern_info_2["peak_perf"]

        # perf: weighted average between both microkernels
        self.perf = round(
            (perf1 * num_repet_1 + perf2 * num_repet_2) / (num_repet_1 + num_repet_2), 3
        )

    @override
    def __str__(self):
        return f"Lambda({self.num_repet_1}*{self.d_mickern_1} + {self.num_repet_2}*{self.d_mickern_2}|perf={self.perf})"


# Sum of both classes
Microkernel_strat = Microkernel_strat_single | Microkernel_strat_lambda


# Check if a microkernel strategy is a lambda or not
def _is_microkernel_strat_lambda(mickern_strat: Microkernel_strat) -> bool:
    if isinstance(mickern_strat, Microkernel_strat_single):
        return False
    elif isinstance(mickern_strat, Microkernel_strat_lambda):
        return True
    else:
        raise ValueError(
            f"is_microkernel_strat_lambda : {mickern_strat} is neither a single or lambda strat."
        )


# ====================================================================

# 1) Microkernel selection, in respect to the problem size


# [Aux aux function] Group a list of microkernel according to their size along "dim".
def _regroup_microkernels_split_bucket(
    ld_mickern_info_bucket: List[dict[str, int]], dim: str
) -> List[List[dict[str, int]]]:
    # Use a dictionnary to group microkernels according to their size along "dim"
    dsplited: dict[int, List[dict[str, int]]] = dict()
    for d_mickern_info in ld_mickern_info_bucket:
        mickern_dim_size = d_mickern_info[dim]

        if mickern_dim_size in dsplited:
            dsplited[mickern_dim_size].append(d_mickern_info)
        else:
            dsplited[mickern_dim_size] = [d_mickern_info]

    # Convert the dictionary into a list of list.
    l_ld_mickern_info = []
    for k, lv in dsplited.items():
        l_ld_mickern_info.append(lv)

    return l_ld_mickern_info


# [Aux function] Given a list of microkernel, regroup them by dimension, except for lambda_dim
# Assume that the list of microkernels in "ld_mickern" have all the same keys (dim name).
def _regroup_microkernels_by_size_not_lambda_dim(
    ld_mickern: List[dict[str, int]], lambda_dim: str
) -> List[List[dict[str, int]]]:
    if ld_mickern == []:
        return []

    ldims = list(ld_mickern[0].keys())
    ldims.remove(lambda_dim)
    ldims.remove("peak_perf")

    # DEBUG
    # print(f"ldims = {ldims}")

    # Work recursively on each dimensions of ldims
    l_ld_mickern_info = [ld_mickern]
    for dim in ldims:
        nl_ld_mickern_info: List[List[dict[str, int]]] = []

        # For each bucket...
        for ld_mickern_info_bucket in l_ld_mickern_info:
            l_ld_mickern_info_bucket = _regroup_microkernels_split_bucket(
                ld_mickern_info_bucket, dim
            )

            # Accumulate
            nl_ld_mickern_info = nl_ld_mickern_info + l_ld_mickern_info_bucket

        # Update the working one
        l_ld_mickern_info = nl_ld_mickern_info

    return l_ld_mickern_info


# [Aux aux function] Perform Euclid algorithm on 2 numbers that are prime between each other
# to figure out coefficients a,b such that a*x1p+b*x2p = 1
# Invariant: arg1 = lcoeff1 . [x1p, x2p]^T  | arg2 = lcoeff2 . [x1p, x2p]^T
def _euclid_algorithm_prime(
    arg1: int, arg2: int, lcoeff1: List[int], lcoeff2: List[int]
) -> Tuple[int, int]:
    # Order the input
    if arg1 < arg2:
        return _euclid_algorithm_prime(arg2, arg1, lcoeff2, lcoeff1)

    # One step
    mod_arg12 = arg1 % arg2
    int_div_arg12 = int((arg1 - mod_arg12) / arg2)

    assert mod_arg12 != 0  # arg1 and arg2 are supposed to be prime between each other

    nlcoeff = [
        lcoeff1[0] - int_div_arg12 * lcoeff2[0],
        lcoeff1[1] - int_div_arg12 * lcoeff2[1],
    ]

    if mod_arg12 == 1:
        # Algo done, return [a,b] = nlcoeff
        return (nlcoeff[0], nlcoeff[1])
    else:
        # Recursion
        return _euclid_algorithm_prime(arg2, mod_arg12, lcoeff2, nlcoeff)


# [Aux aux function] List all the divisors of "n", including 1/itself
def _list_all_divisors(n: int) -> List[int]:
    if n == 1:
        return [1]
    ldiv = [1]

    half_way = int(math.floor(math.sqrt(n)))
    for div in range(2, half_way + 1):
        if n % div == 0:
            ldiv.append(div)
            if n != div * div:
                ldiv.append(int(n / div))

    ldiv.append(n)

    return ldiv


# [Aux function] Find all combination of a,b,k>0 such that (a*x1 + b*x2) * k = size_x
def find_affine_combination_dividing(x1: int, x2: int, size_x: int) -> List[List[int]]:
    # DEBUG
    # print(f"x1 = {x1} | x2 = {x2} | size_x = {size_x}")

    # Algorithm:
    # 1) Makes x1/x2 prime between each other => reduce size_x in function (if not divisible, fails)
    # 2) Find one a/b (not always positive) such that a * x1 + b * x2 = size_x
    # (note: x1/x2 is prime => we can use Euclid here to get a*x1+b*x2=1 then scale back)
    # 3) Make a/b both positive ===> If not possible, fails.
    # At that point, if succeed, then we have at least 1 combination here.
    # 4) If a/b have common multiple, counts the divisors (===> corresponds to divisors of the full tile size, cf "k")

    # 1) Compute gcd of x1 and x2 to simplify the problem
    gcd_x12 = math.gcd(x1, x2)
    if size_x % gcd_x12 != 0:
        return []

    x1p = int(x1 / gcd_x12)
    x2p = int(x2 / gcd_x12)
    size_x_p = int(size_x / gcd_x12)

    # DEBUG
    # print(f"x1p = {x1p}, x2p = {x2p}")

    # 2) Euclid algorithm to get a,b (not always positive) such that "a*x1p + b*x2p = 1"
    if x1p == 1:
        (a, b) = (1, 0)
    elif x2p == 1:
        (a, b) = (0, 1)
    else:
        (a, b) = _euclid_algorithm_prime(x1p, x2p, [1, 0], [0, 1])

    # We multiply a and b by size_x_p, such that "a*x1p + b*x2p = size_x_p"
    a = int(a * size_x_p)
    b = int(b * size_x_p)

    # DEBUG
    # print(f"{a=} {b=}")

    # 3) Vary a and b to list all combination that makes them both positive
    # "a*x1p + b*x2p = size_x_p" => "(a+x2p)*x1p + (b-x2p)*x2p = size_x_p"
    # Note: this list might be empty

    # We start with the lower positive a possible, then we go by increment of "x2p"
    # 0< small_a < x2p
    if a < 0:
        small_a = a + (x2p * size_x_p)  # Unsubtle, to be sure than small_a is positive
    else:
        small_a = a
    small_a = small_a % x2p
    if small_a == 0:
        small_a = x2p

    num_increm_small_a = int(
        (a - small_a) / x2p
    )  # small_a = a - num_increm_small_a * x2p
    small_b = b + num_increm_small_a * x1p  # small_b = b + num_increm_small_a * x1p

    assert small_a * x1p + small_b * x2p == size_x_p

    # While small_b >0, we increase small_a
    l_ab = []
    while small_b > 0:
        # Record the combination
        l_ab.append([small_a, small_b])

        # Update small_a/small_b
        small_a = small_a + x2p
        small_b = small_b - x1p

    # DEBUG
    # print(f"{l_ab=}")

    # 4) If a/b have common multiple, counts the divisors (===> corresponds to divisors of the full tile size, cf "k")
    l_abk = []
    for ab_found in l_ab:
        a = ab_found[0]
        b = ab_found[1]
        gcd_ab = math.gcd(a, b)

        # List all divisors of gcd_ab, including 1
        ldiv = _list_all_divisors(gcd_ab)

        # print(f"{gcd_ab=}")
        # print(f"{ldiv=}")

        for k in ldiv:
            l_abk.append([int(a / k), int(b / k), k])

    return l_abk


# From a list "ld_mickern_info" of information about the microkernel of a problem, potentially filtered
# and "dprob_sizes", the problem sizes
# Isolate compatible single microkernel, or lambda of microkernel (if blambda is activated)
# lambda_dim is the dim along which we will try to vary the size of the microkernel to make it fit
def select_microkernel_ttile(
    ld_mickern_info: list[dict[str, Any]],
    vector_dim: str,
    num_elem_vector: int,
    dprob_sizes: dict[str, int],
    blambda: bool = False,
    lambda_dim: Optional[str] = None,
) -> List[Microkernel_strat]:
    # No lambda
    l_mickern_strat: List[Microkernel_strat] = []
    for d_mickern_info in ld_mickern_info:
        # Don't forget to take vectorization into account when comparing the sizes
        d_mickern_info_sizes = d_mickern_info.copy()
        d_mickern_info_sizes[vector_dim] = (
            d_mickern_info_sizes[vector_dim] * num_elem_vector
        )

        if is_compatible_microkernel(d_mickern_info_sizes, dprob_sizes):
            mickern_strat = Microkernel_strat_single(d_mickern_info)
            l_mickern_strat.append(mickern_strat)

    if not blambda:
        return l_mickern_strat

    # Now, we have to deal with lambda (in extra)

    # Check that lambda_dim is the right one
    assert lambda_dim != None
    assert lambda_dim in dprob_sizes

    # Filter the microkernel in respect to the rest of the problem sizes
    dprob_sizes_no_lambda_dim = dprob_sizes.copy()
    del dprob_sizes_no_lambda_dim[lambda_dim]

    l_mickern_base_lambda = []
    for d_mickern_info in ld_mickern_info:
        # Don't forget to take vectorization into account when comparing the sizes
        # Remove lambda_dim, since it is variable (we check if there are possible combination afterwards)
        d_mickern_info_copy = d_mickern_info.copy()
        del d_mickern_info_copy[lambda_dim]
        d_mickern_info_copy[vector_dim] = (
            d_mickern_info_copy[vector_dim] * num_elem_vector
        )

        if is_compatible_microkernel(d_mickern_info_copy, dprob_sizes_no_lambda_dim):
            l_mickern_base_lambda.append(d_mickern_info)

    # DEBUG
    # print(f"{l_mickern_base_lambda=}")

    # From the "l_mickern_base_lambda", we look at their value on "lambda_dim"
    # and check how to compose them.

    # 1) Regroup microkernels depending on the other dims than lambda_dim
    l_ld_mickern_info = _regroup_microkernels_by_size_not_lambda_dim(
        l_mickern_base_lambda, lambda_dim
    )

    # DEBUG
    # print("l_ld_mickern_info =")
    # for ld_mickern_info in l_ld_mickern_info:
    # print(f"- {ld_mickern_info}")

    # Note: the sizes in each buckets should be consecutive.

    # 2) For each bucket, go over each couple of different elements (x1,x2),
    # then find (a * x1 + b * x2) * k = size_x where a,b>0 and k>0
    size_x = dprob_sizes[lambda_dim]

    # For each bucket of combinable microkernels ...
    for ld_mickern_info in l_ld_mickern_info:
        len_bucket = len(ld_mickern_info)
        if len_bucket < 2:
            continue

        # For each pairs of different elements
        for i in range(len_bucket):
            for j in range(i + 1, len_bucket):
                d_mickern1 = ld_mickern_info[i]
                d_mickern2 = ld_mickern_info[j]

                x1 = d_mickern1[lambda_dim]
                x2 = d_mickern2[lambda_dim]

                # Performing a bit of mathematical black magic...
                l_abk = find_affine_combination_dividing(x1, x2, size_x)

                for abk in l_abk:
                    num_repet_1 = abk[0]
                    num_repet_2 = abk[1]

                    # Build the corresponding microkernel strat
                    lambda_strat = Microkernel_strat_lambda(
                        num_repet_1, d_mickern1, num_repet_2, d_mickern2, lambda_dim
                    )
                    l_mickern_strat.append(lambda_strat)

    # All done
    return l_mickern_strat


# Convert a microkernel strategy information into a concrete microkernel scheme
def convert_microkernel_strat_to_scheme(
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    unroll_order: List[str],
    mickern_strat: Microkernel_strat,
) -> List[Atom]:
    match mickern_strat:
        case Microkernel_strat_single():
            # Reuse the microkernel structure and its code generator
            mickern = convert_microkernel_info_to_microkernel(
                machine, comp, vector_dim, unroll_order, mickern_strat.d_mickern
            )
            scheme_mickern = mickern.to_ttile_scheme()
            return scheme_mickern

        case Microkernel_strat_lambda():
            lambda_dim = mickern_strat.lambda_dim
            d_mickern_1 = mickern_strat.d_mickern_1
            d_mickern_2 = mickern_strat.d_mickern_2

            # We start from the scheme of the first microkernel
            mickern1 = convert_microkernel_info_to_microkernel(
                machine, comp, vector_dim, unroll_order, d_mickern_1
            )
            scheme_mickern = mickern1.to_ttile_scheme()

            # We edit this scheme to replace the unroll on lamba_dim into a UL
            lratio_lambda_dim = [d_mickern_1[lambda_dim], d_mickern_2[lambda_dim]]
            atom_UL = scheme.new_unrollLambda_atom(lambda_dim, lratio_lambda_dim)

            for i in range(len(scheme_mickern)):
                if (scheme_mickern[i].type == scheme.AtomType.UNROLL) and (
                    scheme_mickern[i].dim == lambda_dim
                ):
                    # Replace this atom by atom_UL
                    scheme_mickern[i] = atom_UL
                    break

            return scheme_mickern


# ====================================================================

# 2) Ttile selection algorithm - microkernel-based random scheme selection algorithm

# Description of the Ttile strat ( "tree_search.ml :: random_candidates_from_uk" + "tile_spec.ml"):
# [Microkernel side]
# - From a list of "good" microkernels, compatible with the problem size
# (selected by using a threshold or keeping the N-best)
# (note that they can be lambda)
# - Have a tile loop on the reduction dimension to repeat the microkernel
# to exploit their register-locality.
# This repetition should have at least "cmin" iteration (32 is a typical number)
# Note that if the remaining problem dimension is below "cmin", we have to adapt "cmin" to take all of the problem size.
# - Above this reduction tile loop, we place a "hoist" atom (should be useless, but in the case of)
#
# [Parallelism side]
# - If parallelism: we start reserving iterations from the parallel dimension.
# Given a number of thread/core, try to get the exact number of iteration on parallel dimension (to book them)
# If we don't manage to get an exact count, try to book at least "4 * nthread" parallel iterations
# If this is not feasible, book as much parallel iterations as possible
#
# [Rest of the tiles] (this is where most of the randomly draw algorithm might vary)
# - Compute the rest of the factor/problem sizes that needs to be distributed
# - Draw randomly a dimension which is not completed,
# then check the remaining size and draw randomly one of its divisor
# - Continue until every dimension is completed.


# [Aux function] Divides some sizes by other sizes
# Note: d_size_div must only have list of 1 element on its values (aka, no lambda)
def _dprob_sizes_remaining(
    dprob_sizes: dict[str, int], d_size_div: dict[str, List[int]]
) -> dict[str, int]:
    d_res_sizes = dprob_sizes.copy()

    for dim in d_size_div.keys():
        size_1 = d_res_sizes[dim]
        size_div = d_size_div[dim]

        assert len(size_div) == 1

        # Check size_div divides the first set of sizes
        assert size_1 % size_div[0] == 0

        # Perform the division
        d_res_sizes[dim] = size_1 // size_div[0]

    return d_res_sizes


# [Aux aux function] Get the list of prime divisors of a number
# 	Recursive function: "n" is the rest of the number,
# and "k" is the current coefficient being checked
def _list_all_prime_divisors(n: int, k: int = 2) -> List[int]:
    # Termination cases
    if n == 1:
        return []
    if k > n:
        return [n]

    # Recursion
    if n % k == 0:
        # k is a new prime divisor: add it
        ldiv = _list_all_prime_divisors(int(n / k), k)
        ldiv.append(k)
        return ldiv
    else:
        # Carry on
        return _list_all_prime_divisors(n, k + 1)


# [Aux function] Reservation of the parallel iterations
# Reference function is "tree_search.ml :: get_parallelization_reservation"
# lparallel_dim = list of the parallel dimension, on which we can have parallelism
# dsize_rem = dict: key = dimension / value = remaining ratio to distribute in a scheme for that dim
# nthread = number of thread/degree of parallelism targetted
# threashold_ndiv = if perfect divisibility is not attainable, try to get (nthread*threashold_ndiv) parallel iterations
def _book_parallel_dimensions(
    lparallel_dim: List[str],
    d_size_rem: dict[str, int],
    nthread: int,
    threashold_ndiv: int = 4,
) -> List[Atom]:
    assert nthread != None

    # 1) First algorithm - Perfect load balance
    # We try to pick randomly factors from the sizes of the parallel dimension,
    # to reach exactly "nthread"

    # We get all the prime divisors of the parallel dimensions
    d_div_para_dim = dict()
    for dim in lparallel_dim:
        d_div_para_dim[dim] = _list_all_prime_divisors(d_size_rem[dim])

    # We get the prime divisors of nthread
    ldiv_nthread = _list_all_prime_divisors(nthread)

    # DEBUG
    # print(f"{d_div_para_dim=}")
    # print(f"{ldiv_nthread=}")

    # For each element of ldiv_nthread, we check if it is possible to get it
    # 	If yes, select randomly the dimension from which we get it
    bworked = True
    l_parallel_scheme = []

    for div_nthr in ldiv_nthread:
        # List of possible dimensions, that have this prime factor
        lposs_dim = []
        for dim in lparallel_dim:
            if div_nthr in d_div_para_dim[dim]:
                lposs_dim.append(dim)

        # If lposs_dim is empty: the method fails (div_nthr cannot be found)
        if lposs_dim == []:
            bworked = False
            break

        # Else, draw randomly inside lposs_dim
        dim_selected = random.choice(lposs_dim)

        # Now that the decision is made, create the atom
        ntilepar_atom = scheme.new_paralleltile_atom(dim_selected, div_nthr)
        l_parallel_scheme.append(ntilepar_atom)

        # Update d_div_para_dim
        d_div_para_dim[dim_selected].remove(div_nthr)

    # If success
    if bworked:
        # DEBUG
        # print("First parallel algorithm worked")

        return l_parallel_scheme

    # Else: too bad, let's switch to the backup plan.
    l_parallel_scheme = []

    # 2) Second algorithm - try to book at least "4 * nthread" parallel iterations
    # If this is not feasible, book as much parallel iterations as possible
    threashold = threashold_ndiv * nthread

    # While we do not cross the threashold
    num_cur_thr = 1

    l_remain_parallel_dim = []
    for dim in lparallel_dim:
        if d_size_rem[dim] > 1:
            l_remain_parallel_dim.append(dim)

    # We keep track of the divisors already taken
    d_ratio_taken = dict()
    for dim in lparallel_dim:
        d_ratio_taken[dim] = 1

    while (num_cur_thr < threashold) and (l_remain_parallel_dim != []):
        # Select randomly a dimension from "l_remain_parallel_dim"
        #  then one of its prime factor
        dim_selected = random.choice(l_remain_parallel_dim)
        size_dim_selected = d_size_rem[dim_selected] // d_ratio_taken[dim_selected]
        l_prime_factors = _list_all_prime_divisors(
            size_dim_selected
        )  # Note we could avoid recomputation
        ratio_selected = random.choice(l_prime_factors)

        # Build the new atom
        ntilepar_atom = scheme.new_paralleltile_atom(dim_selected, ratio_selected)
        l_parallel_scheme.append(ntilepar_atom)

        # Update d_ratio_taken
        d_ratio_taken[dim_selected] = d_ratio_taken[dim_selected] * ratio_selected

        # Update l_remain_parallel_dim if we just emptied a dimension
        if len(l_prime_factors) == 1:
            l_remain_parallel_dim.remove(dim_selected)

    return l_parallel_scheme


b_debug_complete_scheme_ttile_div = False


# Given a microkernel and a problem size, randomly draw a configuration for the rest of the tiles
# This is Ttile algorithm (cf "ml_utils/search/search.ml :: NonDivRandomSelection" for the implementation)
# Alternatively: tree_search :: random_candidates_from_uk
def complete_scheme_from_mickern_ttile_div(
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    unroll_order: List[str],
    mickern_strat: Microkernel_strat,
    dprob_sizes: dict[str, int],
    reuse_dim: str,
    reuse_loop_min: int,
    loutput_array_name: List[str],
    lparallel_dim: List[str],
    nthread: Optional[int] = None,
) -> List[Atom]:
    # 1) Microkernel side
    l_mickern_scheme = convert_microkernel_strat_to_scheme(
        machine, comp, vector_dim, unroll_order, mickern_strat
    )

    # Remaining atoms to complete a lambda (to be inserted above)
    if _is_microkernel_strat_lambda(mickern_strat):
        assert isinstance(mickern_strat, Microkernel_strat_lambda)
        lambda_dim = mickern_strat.lambda_dim
        lratios = [mickern_strat.num_repet_1, mickern_strat.num_repet_2]
        num_repet_tile_atom = scheme.new_tileLambda_atom(lambda_dim, lratios)

        seq_atom = scheme.new_seq_atom(lambda_dim)

        l_lambda_scheme = [num_repet_tile_atom, seq_atom]
    else:
        lambda_dim = None
        l_lambda_scheme = []

    # Compute the remaining size that needs to be managed by the rest of the scheme
    d_sizemickern = scheme.get_sizes_scheme(l_mickern_scheme)
    if lambda_dim == None:
        # No lambda in the microkernel
        d_size_rem = _dprob_sizes_remaining(dprob_sizes, d_sizemickern)
    else:
        # Lambda in the microkernel
        assert isinstance(mickern_strat, Microkernel_strat_lambda)
        dsub_sizes = dict()
        for dim in d_sizemickern:
            if dim != lambda_dim:
                dsub_sizes[dim] = d_sizemickern[dim]
            else:
                assert len(d_sizemickern[lambda_dim]) == 2
                num_repet_1 = mickern_strat.num_repet_1
                num_repet_2 = mickern_strat.num_repet_2
                dsub_sizes[lambda_dim] = [
                    num_repet_1 * d_sizemickern[dim][0]
                    + num_repet_2 * d_sizemickern[dim][1]
                ]
        # print(f"{dprob_sizes=}")
        # print(f"{dsub_sizes=}")
        d_size_rem = _dprob_sizes_remaining(dprob_sizes, dsub_sizes)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_div:
        print(f"{lambda_dim=}")
        print(f"l_mickern_scheme={scheme.convert_scheme_to_str(l_mickern_scheme)}")
        print(f"l_lambda_scheme={scheme.convert_scheme_to_str(l_lambda_scheme)}")
        print(f"d_size_rem (after mickern) = {d_size_rem}")
        print()

    # 2) Parallel side
    if nthread == None:
        l_parallel_scheme = []
    else:
        # Book parallel dimensions
        l_parallel_scheme = _book_parallel_dimensions(
            lparallel_dim, d_size_rem, nthread
        )

        # Update the remaining sizes
        d_size_parallel = scheme.get_sizes_scheme(l_parallel_scheme)
        d_size_rem = _dprob_sizes_remaining(d_size_rem, d_size_parallel)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_div:
        print(f"l_parallel_scheme={scheme.convert_scheme_to_str(l_parallel_scheme)}")
        print(f"d_size_rem (after //) = {d_size_rem}")
        print()

    # 3) Rest of the scheme

    # a) First, set up a reuse_dim loop above the microkernel
    reuse_loop_min_adapted = min(reuse_loop_min, d_size_rem[reuse_dim])
    assert (
        dprob_sizes[reuse_dim] % reuse_loop_min_adapted == 0
    )  # TODO: improve this? :/
    atom_reuse_loop = scheme.new_tile_atom(reuse_dim, reuse_loop_min_adapted)
    atom_hoist = scheme.new_hoist_atom(loutput_array_name)

    lreuse_loop_scheme = [atom_reuse_loop, atom_hoist]

    # Update the remaining sizes
    d_size_reuse = scheme.get_sizes_scheme(lreuse_loop_scheme)
    d_size_rem = _dprob_sizes_remaining(d_size_rem, d_size_reuse)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_div:
        print(f"lreuse_loop_scheme={scheme.convert_scheme_to_str(lreuse_loop_scheme)}")
        print(f"d_size_rem (after reuse loop) = {d_size_rem}")

    # b) Selection of the rest of the divisors

    # Create a bag of scheme atoms:
    # For each dimension, draw one of its divisor, and continue until there is nothing left
    #   Don't forget to add the lambda, if we have one from the microkernel
    l_bag_lscheme = []

    if _is_microkernel_strat_lambda(mickern_strat):
        l_bag_lscheme.append(l_lambda_scheme)

    for dim in d_size_rem.keys():
        rem_ratio = d_size_rem[dim]

        # Decide how to decompose the remaining ratio of this dim
        while rem_ratio > 1:
            ldiv_rem_ratio = _list_all_divisors(rem_ratio)
            ldiv_rem_ratio.remove(1)

            n_ratio = random.choice(ldiv_rem_ratio)

            # Create the new atom and add it to the bad of scheme atoms
            n_atom = scheme.new_tile_atom(dim, n_ratio)
            l_bag_lscheme.append([n_atom])

            # Update rem_ratio
            rem_ratio = rem_ratio // n_ratio

    # DEBUG
    # print("l_bag_lscheme =")
    # for lscheme in l_bag_lscheme:
    # print(f"- {scheme.convert_scheme_to_str(lscheme)}")

    # Now empty the bag in a random order to get the full scheme
    l_middle_scheme: List[Atom] = []

    while l_bag_lscheme != []:
        latom = random.choice(l_bag_lscheme)
        l_middle_scheme = l_middle_scheme + latom

        l_bag_lscheme.remove(latom)

    # DEBUG
    if b_debug_complete_scheme_ttile_div:
        print(f"l_middle_scheme={scheme.convert_scheme_to_str(l_middle_scheme)}")

    # 4) Full assembly of the scheme
    full_scheme = (
        l_mickern_scheme + lreuse_loop_scheme + l_middle_scheme + l_parallel_scheme
    )
    return full_scheme


# Main function for the Ttile scheme selection algorithm
# [Problem specification]
# - ld_mickern_info = List of microkernel information we wish to use
# - dprob_sizes = problem size
# - machine = target architecture (include available vectorization info and cache sizes)
# - comp = computation (include nature of the computation + size of the element)
# - nthread = None / the number of threads if we want to trigger the parallelism
#
# [Dimension specification]
# - vector_dim = vectorization dimension
# - lambda_dim = dimension along which we will combine microkernels
# - reuse_dim = dimension along which we will stream the microkernel
# - lparallel_dim = dimensions that can be used to parallelize (on the outermost loops)
# - loutput_array_name = list of arrays reused during the streaming of the microkernel (for Hoist)
#
# [Parameters of the search strategy]
# - threshold_mickern_perf_ratio = ratio to the best mickern perf you want for your microkernels
# - unroll_order = unroll order of the microkernel to be used
# - reuse_loop_min = minimal value of the reuse loop above the microkernel. By default 32
def full_ttile_algorithm(
    filename_mickern_info: str,
    dprob_sizes: dict[str, int],
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    lambda_dim: Optional[str],
    reuse_dim: str,
    lparallel_dim: List[str],
    loutput_array_name: List[str],
    threshold_mickern_perf_ratio: float,
    unroll_order: List[str],
    reuse_loop_min: int,
    nthread: Optional[int] = None,
) -> Optional[List[Atom]]:
    """
    Main function for the Ttile scheme selection algorithm

    Inputs:
    [Problem specification]
    - ld_mickern_info = List of microkernel information we wish to use
    - dprob_sizes = problem size
    - machine = target architecture (include available vectorization info and cache sizes)
    - comp = computation (include nature of the computation + size of the element)
    - nthread = None / the number of threads if we want to trigger the parallelism

    [Dimension specification]
    - vector_dim = vectorization dimension
    - lambda_dim = dimension along which we will combine microkernels
    - reuse_dim = dimension along which we will stream the microkernel
    - lparallel_dim = dimensions that can be used to parallelize (on the outermost loops)
    - loutput_array_name = list of arrays reused during the streaming of the microkernel (for Hoist)

    [Parameters of the search strategy]
    - threshold_mickern_perf_ratio = ratio to the best mickern perf you want for your microkernels
    - unroll_order = unroll order of the microkernel to be used
    - reuse_loop_min = minimal value of the reuse loop above the microkernel. By default 32

    Output:
    - If possible (depending on available microkernels and parameters to the algo), output a scheme
      from the Ttile search space, with only divisible tile sizes ("T/U/Tparal" atoms).
    """

    # Load the microkernel infos and select the good ones
    (machine_name, computation_name, ld_mickern_info) = load_microkernel_info(
        filename_mickern_info
    )
    assert machine_name == machine.name
    assert str(comp) == computation_name

    # Filter the microkernel infos according to performance
    ld_mickern_info = sort_microkernels_by_pperf(ld_mickern_info)
    best_perf_mickern = ld_mickern_info[-1]["peak_perf"]
    threashold_perf_mickern = best_perf_mickern * 0.85  # Arbitrary perf threshold

    ld_mickern_info_filtered = []
    for d_mickern_info in ld_mickern_info:
        if is_faster_microkernel_than(d_mickern_info, threashold_perf_mickern):
            ld_mickern_info_filtered.append(d_mickern_info)

    # Microkernel selection
    if lambda_dim == None:
        blambda = False
    else:
        blambda = True

    num_elem_vector = int(machine.vector_size / comp.elem_size)
    l_mickern_strat = select_microkernel_ttile(
        ld_mickern_info_filtered,
        vector_dim,
        num_elem_vector,
        dprob_sizes,
        blambda=blambda,
        lambda_dim=lambda_dim,
    )

    if len(l_mickern_strat) == 0:
        print(
            "No compatible microkernel strategy was found, please relax the microkernel space !"
        )
        return None

    mickern_strat = random.choice(l_mickern_strat)

    # DEBUG
    # print(f"Selected microkernel strategy: {mickern_strat}")
    # print(f"{dprob_sizes=}")

    # Get the rest
    full_scheme = complete_scheme_from_mickern_ttile_div(
        machine,
        comp,
        vector_dim,
        unroll_order,
        mickern_strat,
        dprob_sizes,
        reuse_dim,
        reuse_loop_min,
        loutput_array_name,
        lparallel_dim,
        nthread=nthread,
    )
    return full_scheme


# ====================================================================

# 3) Selection algorithms with partial tiles

b_debug_complete_scheme_ttile_nondiv = False


# Reference implementation in "search.ml" (random algorithm selection - non divisible case)
def complete_scheme_from_mickern_ttile_nondiv(
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    unroll_order: List[str],
    mickern_strat: Microkernel_strat,
    dprob_sizes: dict[str, int],
    reuse_dim: str,
    reuse_loop_min: int,
    loutput_array_name: List[str],
    lparallel_dim: List[str],
    nthread: Optional[int] = None,
) -> List[Atom]:
    # 1) Microkernel side
    l_mickern_scheme = convert_microkernel_strat_to_scheme(
        machine, comp, vector_dim, unroll_order, mickern_strat
    )

    # Remaining atoms to complete a lambda (to be inserted above)
    assert not _is_microkernel_strat_lambda(mickern_strat)

    # Compute the remaining size that needs to be managed by the rest of the scheme
    d_sizemickern = scheme.get_sizes_scheme(l_mickern_scheme)
    # No lambda in the microkernel
    d_size_rem = _dprob_sizes_remaining(dprob_sizes, d_sizemickern)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_nondiv:
        print(f"l_mickern_scheme={scheme.convert_scheme_to_str(l_mickern_scheme)}")
        print(f"d_size_rem (after mickern) = {d_size_rem}\n")

    # 2) Parallel side
    # Note: we should be able to manage parallelism as long at it is on the outer tiles.
    # Then, we really have a ratio.
    if nthread == None:
        l_parallel_scheme = []
    else:
        # Book parallel dimensions
        l_parallel_scheme = _book_parallel_dimensions(
            lparallel_dim, d_size_rem, nthread
        )

        # Update the remaining sizes
        d_size_parallel = scheme.get_sizes_scheme(l_parallel_scheme)
        d_size_rem = _dprob_sizes_remaining(d_size_rem, d_size_parallel)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_nondiv:
        print(f"l_parallel_scheme={scheme.convert_scheme_to_str(l_parallel_scheme)}")
        print(f"d_size_rem (after //) = {d_size_rem}\n")

    # 3) Rest of the scheme

    # a) First, set up a reuse_dim loop above the microkernel
    reuse_loop_min_adapted = min(reuse_loop_min, d_size_rem[reuse_dim])
    assert (
        dprob_sizes[reuse_dim] % reuse_loop_min_adapted == 0
    )  # TODO: improve this? :/
    atom_reuse_loop = scheme.new_tile_atom(reuse_dim, reuse_loop_min_adapted)
    atom_hoist = scheme.new_hoist_atom(loutput_array_name)

    lreuse_loop_scheme = [atom_reuse_loop, atom_hoist]

    # Update the remaining sizes
    d_size_reuse = scheme.get_sizes_scheme(lreuse_loop_scheme)
    d_size_rem = _dprob_sizes_remaining(d_size_rem, d_size_reuse)

    # DEBUG/Interesting variables for the rest of this function
    if b_debug_complete_scheme_ttile_nondiv:
        print(f"lreuse_loop_scheme={scheme.convert_scheme_to_str(lreuse_loop_scheme)}")
        print(f"d_size_rem (after reuse loop) = {d_size_rem}")

    # b) Selection of the rest of the tiles
    l_middle_scheme: List[Atom] = []

    # Algorithm:
    # - For each dimension d, select a number of level l_d between 1 and 4 (4 = max number of memories on CPU)
    # - For each dimension/level (d,l) (except the last level),
    #    pick a number between [2, problem size] and sort them
    #    note: even if the ratio is exact, we do not "T" to improve perf
    #        (it would have an effect only if we have 2 consecutives T)
    #  => Build a "bag" of possibilities (dim, random_size) for every dimension
    # - For the permutation: draw the possibility uniformly over all the remaining dimensions
    #  => When selected, take the new size, and iterate until the bag is empty

    l_scheme_below = l_mickern_scheme + lreuse_loop_scheme
    dsize_mickern = scheme.get_sizes_scheme(l_scheme_below)

    # Get the tile sizes for each dimension and build the corresponding scheme
    # Note that these scheme needs to be selected in order to be valid
    d_latom = dict()
    for dim in d_size_rem.keys():
        # Remaining sizes on this dim
        remaining_ratio = d_size_rem[dim]
        if remaining_ratio == 1:
            continue

        # Select the number of tiling level on that dimension
        num_tiling_lvl = random.randint(1, 4)

        # Draw the factor sizes = ( Real_tilesize / microkernel_size )
        lsizes_factor = [remaining_ratio]
        for l in range(num_tiling_lvl - 1):
            tilesize_factor = random.randint(2, remaining_ratio)
            lsizes_factor.append(tilesize_factor)

        # Sort the draws in increasing order
        lsizes_factor.sort()

        # Remove repetitions in lsizes_factor
        last_elem = None
        lsizes_factor_norepet = []
        for i in range(len(lsizes_factor)):
            if last_elem != lsizes_factor[i]:
                lsizes_factor_norepet.append(lsizes_factor[i])
                last_elem = lsizes_factor[i]

        # DEBUG
        # print(f"Dim {dim} (num_tiling_lvl = {num_tiling_lvl}) -> {lsizes_factor_norepet=}")

        # We now create the atoms for this dimension
        latom_dim = []
        for size_factor in lsizes_factor_norepet:
            lsize_mickernel = dsize_mickern[dim]
            assert len(lsize_mickernel) == 1
            size_mickernel = lsize_mickernel[0]

            size_tile = int(size_factor * size_mickernel)
            natom = scheme.new_partialtile_atom(dim, size_tile)
            latom_dim.append(natom)

        # Commit
        d_latom[dim] = latom_dim

    # DEBUG
    # print(f"{d_latom=}")

    # Finally, we draw uniformly on the dimension that still have atoms in d_latom,
    # until no atom remains.
    l_dim_dlatom = list(d_latom.keys())

    l_middle_scheme = []
    while l_dim_dlatom != []:
        # Draw a dim randomly
        dim = random.choice(l_dim_dlatom)

        # Pick the first atom of this dim
        natom = d_latom[dim][0]
        l_middle_scheme.append(natom)

        # Update d_latom by removing its first element
        d_latom[dim] = d_latom[dim][1:]

        # Update l_dim_dlatom is there is no more atom on this dim
        if d_latom[dim] == []:
            l_dim_dlatom.remove(dim)

    # 4) Full assembly of the scheme
    full_scheme = (
        l_mickern_scheme + lreuse_loop_scheme + l_middle_scheme + l_parallel_scheme
    )
    return full_scheme


# Main function for the Ttile selection with partial tiles, and whole microkernel
# Note that lambda is not supported with partial tile (yet?)
# The input is aligned with the "full_ttile_algorithm" function.
def ttile_partial_tile_algorithm(
    filename_mickern_info: str,
    dprob_sizes: dict[str, int],
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    lambda_dim: Optional[str],
    reuse_dim: str,
    lparallel_dim: List[str],
    loutput_array_name: List[str],
    threshold_mickern_perf_ratio: float,
    unroll_order: List[str],
    reuse_loop_min: int,
    nthread: Optional[int] = None,
) -> Optional[List[Atom]]:
    """
    Ttile scheme selection algorithm, based on microkernels and partial tiles.
      The only divisibility constraint of a tile size is in respect to its microkernel size
        (!= the tile size of its previous loop level)
      Note that lambda is not supported with partial tiles right now, so the microkernel
        strategy must be a single microkernel.

    Inputs:
     - filename_mickern_info: path where the microkernel database are stored
     - dprob_sizes : problem sizes
     - machine: targeted architecture
     - comp : computation being performed
     - loutput_array_name : list of output arrays of the computation

    [Options specific to the algorithms]
     - threshold_mickern_perf_ratio : from where are we considering microkernels?
     - unroll_order : unroll order in the microkernels
     - reuse_loop_min : minimum number of iteration of the reuse loop above the microkernel
     - nthread: either None if sequential, or the number of thread we wish to use for parallelisation

    Output:
     - If possible (depending on the available microkernels and how constraining are the option),
       output a (random) scheme in the Ttile search space, that uses partial tiles.
    """

    assert lambda_dim == None

    # Load the microkernel infos and select the good ones
    (machine_name, computation_name, ld_mickern_info) = load_microkernel_info(
        filename_mickern_info
    )
    assert machine_name == machine.name
    assert str(comp) == computation_name

    # Filter the microkernel infos
    ld_mickern_info = sort_microkernels_by_pperf(ld_mickern_info)
    best_perf_mickern = ld_mickern_info[-1]["peak_perf"]
    threashold_perf_mickern = best_perf_mickern * 0.85  # Arbitrary

    ld_mickern_info_filtered = []
    for d_mickern_info in ld_mickern_info:
        if is_faster_microkernel_than(d_mickern_info, threashold_perf_mickern):
            ld_mickern_info_filtered.append(d_mickern_info)

    # Microkernel selection, with no lambda
    num_elem_vector = int(machine.vector_size / comp.elem_size)
    l_mickern_strat = select_microkernel_ttile(
        ld_mickern_info_filtered,
        vector_dim,
        num_elem_vector,
        dprob_sizes,
        blambda=False,
    )

    if len(l_mickern_strat) == 0:
        print(
            "No compatible microkernel strategy was found, please relax the microkernel space !"
        )
        return None

    mickern_strat = random.choice(l_mickern_strat)

    # Complete the rest
    full_scheme = complete_scheme_from_mickern_ttile_nondiv(
        machine,
        comp,
        vector_dim,
        unroll_order,
        mickern_strat,
        dprob_sizes,
        reuse_dim,
        reuse_loop_min,
        loutput_array_name,
        lparallel_dim,
        nthread=nthread,
    )

    return full_scheme


# ====================================================================

# 4) Selection algorithms with tile level and cache occupancy


# TODO: have fun with other way to select? :/
# => One that selects cache level by cache level, respecting constraints on the footprint size? :/
# TODO: Parallel: option to insert above L2 ?
