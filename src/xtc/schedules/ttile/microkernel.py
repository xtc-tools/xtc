#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass

import math

import xtc.schedules.ttile.scheme as scheme
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.archi import Archi
from xtc.schedules.ttile.scheme_to_xtc import launch_and_measure_scheme_graph_interf

# =====================================================================


# Class regrouping the main characteristics attached to a microkernel
@dataclass
class Microkernel_summary:
    machine: Archi
    comp: Computation

    vector_dim: str  # Name of the dimension being vectorized

    unroll_order: List[str]
    unroll_sizes: List[int]  # Both list must be of same lengths

    def to_ttile_scheme(self) -> List[scheme.Atom]:
        lscheme = []

        vect_unroll_factor = int(self.machine.vector_size / self.comp.elem_size)
        vect_atom = scheme.new_vect_atom(self.vector_dim, vect_unroll_factor)
        lscheme.append(vect_atom)

        for i in range(len(self.unroll_order)):
            dim = self.unroll_order[i]
            unroll_ratio = self.unroll_sizes[i]
            unroll_atom = scheme.new_unroll_atom(dim, unroll_ratio)
            lscheme.append(unroll_atom)

        return lscheme

    def to_ttile_str(self) -> str:
        vect_unroll_factor = int(self.machine.vector_size / self.comp.elem_size)
        str_out = f"[V ({self.vector_dim}, {vect_unroll_factor})"
        for i in range(len(self.unroll_order)):
            dim = self.unroll_order[i]
            unroll_ratio = self.unroll_sizes[i]
            str_out += f"; U({dim}, {unroll_ratio})"
        str_out += "]"
        return str_out

    def __str__(self):
        str_out = f"({self.comp}; {self.machine.archi_type}, {self.machine.num_vect_reg} vreg) "
        str_out += self.to_ttile_str()
        return str_out


# perf: stored externally (database of values)?


# ====================================================================

# 1) Microkernel candidate space creation (Ttile version)
# Cf "mickernel_search.ml" for reference


# [Auxfunction] Gives the formula (in string) of the number of register used by the vector-register-inplace array
# of a computation (ex: "Output" for Conv, "C" for Matmult)
def _get_formula_volume_vect_inplace(comp: Computation) -> str:
    if comp.spec == Computation_spec.CONV:
        str_formula = "x * y * f"
    elif comp.spec == Computation_spec.MATMULT:
        str_formula = "i * j"
    else:
        raise ValueError("get_formula_volume_vect_inplace : unknown computation.")
    return str_formula


# [Auxfunction] Gives the formula (in string) of the number of register used by all vectorized arrays
# of a computation (ex: "Output + Kernel" for Conv, "C + B" for Matmult)
def _get_formula_volume_vect_total(comp: Computation) -> str:
    if comp.spec == Computation_spec.CONV:
        str_formula = "x * y * f + h * w * c * f"
    elif comp.spec == Computation_spec.MATMULT:
        str_formula = "i * j + k * j"
    else:
        raise ValueError("get_formula_volume_vect_total : unknown computation.")
    return str_formula


# TODO - Note: both previous functions could be infered (if one day, we extend "Computation_spec")


# Empirical formula, after some empirical observations (cf "stats_microkernel.py")
# - volume_vect_total should be between: [ 0.6 * num_vreg; 1.4 * num_vreg]
# - volume_vect_inplace should be between: [ 0.6 * num_vreg ; 0,875 * num_vreg]
# where num_vreg is the number of vector register of a core of the architecture.
ratio_min_vol_vect_total = 0.45  # 0.6
ratio_max_vol_vect_total = 1.15  # 1.4

ratio_min_vol_vect_inplace = 0  # 0.6
ratio_max_vol_vect_inplace = 1  # 0.875

# Note - here were the (old) bounds taken by Ttile to explore the microkernels:
# AVX2 / 16 vect registers : 8 <= total <= 20   | 8 <= out <= 14
# 	AVX512, Neon / 32 vect registers : 14 <= total <= 42 | 14 <= out <= 28
# After examination, these were too tight on the "spilling" part.


# Generate the list of microkernel candidate for conv2d (Ttile space)
# Note: uses the f/c (channel sizes) x/y (size of the image) w/h (size of the windows) convention
def _generate_conv_mickern(
    machine: Archi, comp: Computation
) -> List[Microkernel_summary]:
    assert comp.spec == Computation_spec.CONV
    vector_dim = "f"
    unroll_order_all = [
        "f",
        "x",
        "y",
        "c",
        "w",
        "h",
    ]  # Unroll order if all dimensions are activated

    # Volume of the inplace vectorized array
    formula_volume_vect_inplace = _get_formula_volume_vect_inplace(comp)

    # Total volume of the vectorized arrays
    formula_volume_vect_total = _get_formula_volume_vect_total(comp)

    # Bounds on these volumes
    min_vol_vect_inplace = int(
        math.ceil(machine.num_vect_reg * ratio_min_vol_vect_inplace)
    )
    max_vol_vect_inplace = int(
        math.floor(machine.num_vect_reg * ratio_max_vol_vect_inplace)
    )
    min_vol_vect_total = int(math.ceil(machine.num_vect_reg * ratio_min_vol_vect_total))
    max_vol_vect_total = int(
        math.floor(machine.num_vect_reg * ratio_max_vol_vect_total)
    )

    # Unroll_max
    unroll_max = 20

    # List of constraints the unroll factors that a conv2D microkernel should satisfy:
    # - All dims: Must be greater than 1, less than unroll_max
    # - volume vect inplace : within the bounds
    # - volume vect total : within the bounds
    # - H/W : If both of them are strictly greater than 1, they must be equal
    # - H/W : Value must be in {1, 3, 5, 7}.
    l_mic_kern_sizes = []
    for f in range(1, unroll_max):
        for x in range(1, unroll_max):
            for y in range(1, unroll_max):
                # Evaluate volume vect inplace here
                vol_vect_inplace = eval(
                    formula_volume_vect_inplace.replace("f", str(f))
                    .replace("x", str(x))
                    .replace("y", str(y))
                )
                if (vol_vect_inplace < min_vol_vect_inplace) or (
                    vol_vect_inplace > max_vol_vect_inplace
                ):
                    continue

                for c in range(1, unroll_max):
                    # h/w within the correct set of values
                    for h in [1, 3, 5, 7]:
                        for w in [1, 3, 5, 7]:
                            # Skip if h/w not equal
                            if h > 1 and w > 1 and h != w:
                                continue

                            # Evaluate volume vect total here
                            vol_vect_total = eval(
                                formula_volume_vect_total.replace("f", str(f))
                                .replace("x", str(x))
                                .replace("y", str(y))
                                .replace("h", str(h))
                                .replace("w", str(w))
                                .replace("c", str(c))
                            )
                            if (vol_vect_total < min_vol_vect_total) or (
                                vol_vect_total > max_vol_vect_total
                            ):
                                continue

                            # All good !
                            l_mic_kern_sizes.append([f, x, y, c, w, h])
                            # Note: unrolled order is ["f", "x", "y", "c", "w", "h"]

    # DEBUG
    # print(l_mic_kern_sizes)

    # Create the list of microkernel objects
    lmickern = []
    for mic_kern_sizes in l_mic_kern_sizes:
        # Filter the unroll factors/dims equal to 1
        unroll_order_filtered = []
        unroll_sizes_filtered = []

        assert len(unroll_order_all) == len(mic_kern_sizes)
        for ind in range(len(unroll_order_all)):
            if mic_kern_sizes[ind] > 1:
                unroll_order_filtered.append(unroll_order_all[ind])
                unroll_sizes_filtered.append(mic_kern_sizes[ind])

        # Create the microkernels here
        mickern = Microkernel_summary(
            machine, comp, vector_dim, unroll_order_filtered, unroll_sizes_filtered
        )
        lmickern.append(mickern)

    return lmickern


# Generate the list of microkernel candidate for matmul (Ttile space)
def _generate_matmult_mickern(
    machine: Archi, comp: Computation
) -> List[Microkernel_summary]:
    assert comp.spec == Computation_spec.MATMULT
    vector_dim = "j"
    unroll_order_all = ["j", "i", "k"]  # Unroll order if all dimensions are activated

    # Volume of the inplace vectorized array
    formula_volume_vect_inplace = _get_formula_volume_vect_inplace(comp)

    # Total volume of the vectorized arrays
    formula_volume_vect_total = _get_formula_volume_vect_total(comp)

    # Bounds on these volumes
    min_vol_vect_inplace = int(
        math.ceil(machine.num_vect_reg * ratio_min_vol_vect_inplace)
    )
    max_vol_vect_inplace = int(
        math.floor(machine.num_vect_reg * ratio_max_vol_vect_inplace)
    )
    min_vol_vect_total = int(math.ceil(machine.num_vect_reg * ratio_min_vol_vect_total))
    max_vol_vect_total = int(
        math.floor(machine.num_vect_reg * ratio_max_vol_vect_total)
    )

    # Unroll_max
    unroll_max = 20

    # List of constraints the unroll factors that a conv2D microkernel should satisfy:
    # - All dims: Must be greater than 1, less than unroll_max
    # - volume vect inplace : within the bounds
    # - volume vect total : within the bounds
    # - "k = 1"
    l_mic_kern_sizes = []
    for i in range(1, unroll_max):
        for j in range(1, unroll_max):
            vol_vect_inplace = eval(
                formula_volume_vect_inplace.replace("i", str(i)).replace("j", str(j))
            )
            if (vol_vect_inplace < min_vol_vect_inplace) or (
                vol_vect_inplace > max_vol_vect_inplace
            ):
                continue

            for k in [1]:
                # Evaluate volume vect total here
                vol_vect_total = eval(
                    formula_volume_vect_total.replace("i", str(i))
                    .replace("j", str(j))
                    .replace("k", str(k))
                )
                if (vol_vect_total < min_vol_vect_total) or (
                    vol_vect_total > max_vol_vect_total
                ):
                    continue

                # All good !
                l_mic_kern_sizes.append([j, i, k])
                # Note: unrolled order is also ["j", "i", "k"]

    # DEBUG
    # print(l_mic_kern_sizes)

    # Create the list of microkernel objects
    lmickern = []
    for mic_kern_sizes in l_mic_kern_sizes:
        # Filter the unroll factors/dims equal to 1
        unroll_order_filtered = []
        unroll_sizes_filtered = []

        assert len(unroll_order_all) == len(mic_kern_sizes)
        for ind in range(len(unroll_order_all)):
            if mic_kern_sizes[ind] > 1:
                unroll_order_filtered.append(unroll_order_all[ind])
                unroll_sizes_filtered.append(mic_kern_sizes[ind])

        # Create the microkernels here
        mickern = Microkernel_summary(
            machine, comp, vector_dim, unroll_order_filtered, unroll_sizes_filtered
        )
        lmickern.append(mickern)

    return lmickern


# From the infos about a machine and a computation, generate a list of µkernel to test
def generate_microkernels(
    machine: Archi, comp: Computation
) -> List[Microkernel_summary]:
    if comp.spec == Computation_spec.CONV:
        return _generate_conv_mickern(machine, comp)
    elif comp.spec == Computation_spec.MATMULT:
        return _generate_matmult_mickern(machine, comp)
    else:
        raise ValueError("generate_microkernels : unknown")


# ====================================================================

# 2) Microkernel testing and measurement
# Cf "mickernel_search.ml" for reference


# [Auxfunction] Gives the dimension of repetition of a microkernel
def _get_repetition_dim(comp: Computation) -> str:
    if comp.spec == Computation_spec.CONV:
        str_dim = "c"
    elif comp.spec == Computation_spec.MATMULT:
        str_dim = "k"
    else:
        raise ValueError("get_repetition_dim : unknown computation.")
    return str_dim


# Given a microkernel scheme, wrap it with a reduction/reuse loop (depending on the computation)
# launch its evaluation (code gen + measurement) and recover its peak performance
def launch_microkernel_testing(
    machine: Archi, comp: Computation, mickern: Microkernel_summary, backend: str
) -> float:
    """
    Launch the execution of a single microkernel in "ideal" situation

    Inputs:
    - machine : the architecture we are targeting
    - comp : the computation being run
    - mickern : the microkernel to be tested
    - backend : "mlir" or "tvm", the backend being use to measure the performance

    Output:
    - peak_perf : an measure of the microkernel run in isolation with a large repetition loop
    """

    # Wrap the scheme with a reduction loop
    mickern_scheme = mickern.to_ttile_scheme()
    # print(f"{mickern_scheme=}")

    repet_dim = _get_repetition_dim(comp)
    repet_loop_size = 512
    repet_atom = scheme.new_tile_atom(repet_dim, repet_loop_size)

    mickern_scheme.append(repet_atom)

    # DEBUG
    # print(scheme.convert_scheme_to_str(mickern_scheme))

    # Launch this!
    dsizes = scheme.get_unlambda_sizes_scheme(mickern_scheme)
    res_measurement = launch_and_measure_scheme_graph_interf(
        comp, machine, mickern_scheme, dsizes, backend
    )
    peak_perf = res_measurement["peak_perf"]

    return peak_perf


# ====================================================================

# 3) Microkernel performance database management


# Get the order of the dimensions inside the csv
def _get_csv_dim_order(comp: Computation) -> List[str]:
    if comp.spec == Computation_spec.CONV:
        ldim_order = ["f", "x", "y", "c", "h", "w"]
    elif comp.spec == Computation_spec.MATMULT:
        ldim_order = ["i", "j", "k"]
    else:
        raise ValueError(f"run_and_save_microkernel_info :: unknown computation {comp}")
    return ldim_order


# Given a machine and computation, get the list of microkernels, and progressively run them/save them
# "name_outfile" = name of the output file
# "b_savemecanism" = If activated, will count the number of entry in the save file to know where
# to continue. It assumes that the order of microkernels is always the same.
def run_and_save_microkernel_info(
    machine: Archi,
    comp: Computation,
    name_outfile: str,
    backend: str,
    b_savemecanism: bool,
    verbose: bool = True,
):
    """
    Launch all microkernels of a (stastically defined) space of interesting configurations.
    It produces a database of microkernels with their associated performance
    Warning: this will take time (depending on the amount of microkernels)

    Note that the process can be interrupted (option "b_savemecanism"): the number of lines
      of the output file is then used to know where to restart the process.

    Inputs:
    - machine : targetted architecture
    - comp : the computation being run
    - name_outfile : where the output (.csv file) will be stored
    - backend : "mlir" or "tvm", the backend compiler to be used to run the computation
    - b_savemecanism :
      If True, the outfile will be examined to know where to continue the runs.
      If False, we restart from the start (and erase the outfile)
    - verbose: activate some print to know where we are in the process.

    Output: None, but the file "name_outfile" will be modified/enriched.
    """

    # Retrieve the order of dims in the save file
    ldim_order = _get_csv_dim_order(comp)

    # Get the list of microkernels (recomputation should be fast)
    lmickern = generate_microkernels(machine, comp)

    # Since that time might be long, we have a save mechanism in place
    # Check the file which is potentially already there
    if b_savemecanism:
        try:
            with open(name_outfile, "r") as fcsv_out_r:
                i_start = 0
                for line in fcsv_out_r:
                    # Comment or column line
                    if line.startswith("#"):
                        continue
                    i_start = i_start + 1
        except OSError as _:
            # File not found
            i_start = 0
    else:
        i_start = 0

    # DEBUG
    # print(f"{i_start=}")

    # Open (or create) the new file "fcsv_out"
    if i_start == 0:
        # Start from a fresh file
        fcsv_out = open(name_outfile, "w")

        # Identification of the file
        fcsv_out.write(
            f"# Microkernel stats - machine = {machine.name} | comp = {comp} | num_mickern = {len(lmickern)}\n"
        )

        # Columns of the csv
        str_col_dimname = ""
        for dim_name in ldim_order:
            str_col_dimname += f"{dim_name},"
        fcsv_out.write(
            f"# {str_col_dimname}peak_perf\n"
        )  # Same order of unroll than the one used
    else:
        # Continue the previous file
        fcsv_out = open(name_outfile, "a")

    # Continue...
    for i in range(i_start, len(lmickern)):
        mickern = lmickern[i]

        if verbose:
            print(
                f"Running microkernel #{i}/{len(lmickern)} - {mickern.to_ttile_str()}"
            )

        # Dimensions
        str_sizes = ""
        for dim_name in ldim_order:
            if dim_name not in mickern.unroll_order:
                size_dim = 1
            else:
                ind_dim = mickern.unroll_order.index(dim_name)
                size_dim = mickern.unroll_sizes[ind_dim]
            str_sizes += f"{size_dim},"

        # Peak perf
        peak_perf = launch_microkernel_testing(machine, comp, mickern, backend)
        peak_perf_100 = round(peak_perf * 100, 3)

        fcsv_out.write(f"{str_sizes}{peak_perf_100}\n")

    fcsv_out.close()
    return


# Load a microkernel info CSV file and return its information
# (machine_name, computation_name, ld_mickern)
def load_microkernel_info(
    name_outfile: str,
) -> Tuple[str, str, List[Dict[str, int | float]]]:
    """
    Load the information of a database of microkernel perf
      (stored in a .csv file, produced by "run_and_save_microkernel_info")
    Input:
     - name_outfile: where the csv file is
    Output:
     - Tuple of machine_name/computation_name/content of the csv.
       The two first elements are great to confirm that we are using microkernels that are compatible with
       the currently considered problem (ie, the machine and the computation do match).
    """
    with open(name_outfile, "r") as fcsv_r:
        # Get the header infos
        # Example:
        # "# Microkernel stats - machine = Laptop_Gui | comp = matmul(f32) | num_mickern = 23"
        line1 = fcsv_r.readline()
        lsplit = line1.split("|")
        assert len(lsplit) == 3

        lsplit0 = lsplit[0].split("=")
        assert len(lsplit0) == 2
        machine_name = lsplit0[1].strip()

        lsplit1 = lsplit[1].split("=")
        assert len(lsplit1) == 2
        computation_name = lsplit1[1].strip()

        lsplit2 = lsplit[2].split("=")
        assert len(lsplit2) == 2
        num_mickern = int(lsplit2[1].strip())

        # Ignore the column info
        # Example: " # i,j,k,peak_perf "
        line2 = fcsv_r.readline()
        assert line2.startswith("#")
        line2 = line2.lstrip("# ")
        lcolumns = line2.split(",")

        for i in range(len(lcolumns)):
            lcolumns[i] = lcolumns[i].strip()

        # DEBUG
        # print(lcolumns)

        assert "peak_perf" in lcolumns

        # Get the informations
        ld_mickern = []
        for line in fcsv_r:
            # Note: it will continue after the second readline
            # But, in the case of this behavior is not stable...
            if line.startswith("#"):
                continue

            ldata = line.split(",")
            assert len(ldata) == len(lcolumns)

            d_mickern = dict()
            for i in range(len(ldata)):
                if lcolumns[i] == "peak_perf":
                    data = float(ldata[i].strip())
                else:
                    data = int(ldata[i].strip())
                d_mickern[lcolumns[i]] = data

            ld_mickern.append(d_mickern)

        # If this is not the case, the file is partial !
        assert len(ld_mickern) == num_mickern
        # print(f"len(ld_mickern) = {len(ld_mickern)} | num_mickern = {num_mickern}")

    return (machine_name, computation_name, ld_mickern)


# Convert back the microkernel info (dict containing the unroll factors) into a Microkernel structure
def convert_microkernel_info_to_microkernel(
    machine: Archi,
    comp: Computation,
    vector_dim: str,
    unroll_order: List[str],
    d_mickern: Dict[str, int],
) -> Microkernel_summary:
    unroll_sizes = []
    for dim in unroll_order:
        unroll_sizes.append(d_mickern[dim])
    mickern = Microkernel_summary(machine, comp, vector_dim, unroll_order, unroll_sizes)
    return mickern


# ====================================================================

# 4) Helper functions for the selection of microkernels, given a problem size.


# Check if a "d_mickern_info" dimensions is compatible with a problem size
def is_compatible_microkernel(d_mickern_info, dprob_sizes: Dict[str, int]) -> bool:
    """
    Check if the sizes of a microkernel are compatible (i.e. divides) with the problem size
    """

    for dim in d_mickern_info.keys():
        # All keys are dimensions, except for "peak_perf"
        if dim == "peak_perf":
            continue

        size_mickern = d_mickern_info[dim]
        if size_mickern > 1:
            assert dim in dprob_sizes  # Incompatibility or dim naming issue

            size_prob = dprob_sizes[dim]
            if size_prob % size_mickern != 0:
                return False

    return True


# Check if a microkernel info is quicker than a peak perf threshold
def is_faster_microkernel_than(d_mickern_info, pperf_threshold: float) -> bool:
    """
    Is a given microkernel faster than the peak performance threshold?
    """

    return d_mickern_info["peak_perf"] >= pperf_threshold


# Sort the microkernels by their peak performance (in increasing performance: last ones are the best ones)
def sort_microkernels_by_pperf(ld_mickern_info):
    """
    Sort the microkernels according to their peak performance
    """
    ld_mickern_info_sorted = ld_mickern_info.copy()
    ld_mickern_info_sorted = sorted(
        ld_mickern_info_sorted, key=lambda e: e["peak_perf"]
    )
    return ld_mickern_info_sorted
