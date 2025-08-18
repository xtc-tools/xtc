#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import tempfile

from typing import List, Dict, Tuple, Any

from xtc.schedules.ttile.archi import Archi, laptop_guillaume_machine, pinocchio_machine

from xtc.schedules.ttile.computation import (
    Computation,
    Computation_spec,
    compute_number_ops,
)
from xtc.schedules.ttile.computation import get_ldims_computation

from xtc.schedules.ttile.scheme import (
    build_scheme_from_str,
    Atom,
    AtomType,
    convert_scheme_to_str,
)
from xtc.schedules.ttile.scheme import get_sizes_scheme
from xtc.schedules.ttile.scheme import (
    stringify_lambda_choice,
    recover_all_branchid_from_stringified,
)
from xtc.schedules.ttile.scheme import (
    recover_branchid_from_stringified,
    remove_dim_from_stringified,
    get_list_dims_str_lambda_loc,
)

# Backends
import xtc.graphs.xtc.op as O
from xtc.itf.back.backend import Backend
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as MlirBackend
from xtc.backends.tvm import TVMBackend as TVMBackend

from xtc.schedules.descript import descript_scheduler

from xtc.utils.cpu import cpu_peak_time

# NOTE: Don't forget to activate the venv of XTC

# Convert an input "Ttile-style" scheme into an input of xdsl-transform
#   Also contain the interfacing to launch xdsl-transform on it

# ===========================================================================

# 1) Helper function to rename the dimensions of a conv2D (since there are 2 existing conventions)


# [Aux functions] Replace the F/C/X/Y/H/W notation to F/C/H/W/R/S for conv2D
def _rename_xyhw_to_hwrs(dim_name: str) -> str:
    if dim_name == "x":
        return "h"
    elif dim_name == "y":
        return "w"
    elif dim_name == "h":
        return "r"
    elif dim_name == "w":
        return "s"
    else:
        return dim_name


def subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme: List[Atom]) -> List[Atom]:
    nscheme = []
    for atom in scheme:
        atom.dim = _rename_xyhw_to_hwrs(atom.dim)
        nscheme.append(atom)
    return nscheme


def subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes: dict[str, int]) -> dict[str, int]:
    dsizes_renamed = dict()
    for k in dsizes:
        nk = _rename_xyhw_to_hwrs(k)
        dsizes_renamed[nk] = dsizes[k]
    return dsizes_renamed


# ===========================================================================

# 2) Conversion fo a Scheme into an input for "xdsl-transform"


# [Aux function] Factorisation of the generation of a string for an atom with a ratio
def _get_spec_atom_ratio(
    dim_atom: str,
    size_atom: int,
    b_lastatom_dim: bool,
    b_unroll: bool,
    b_paral: bool,
    b_is_graph_interface: bool,
) -> str:
    # If we are at the top level, we directly use the dim name
    if not b_lastatom_dim:
        str_size = f"#{size_atom}"
    else:
        str_size = ""

    str_spec_atom = f'"{dim_atom}{str_size}"'

    # Attribute
    if b_unroll:
        if b_is_graph_interface:
            str_unroll = f'"unroll" : {size_atom}'
        else:
            str_unroll = f'"unroll" = {size_atom}'
    else:
        str_unroll = ""

    if b_unroll and b_paral:
        str_comma = ", "
    else:
        str_comma = ""

    if b_paral:
        str_paral = '"parallelize": None'
    else:
        str_paral = ""

    if b_unroll or b_paral:
        if b_is_graph_interface:
            str_modif = ": {" + str_unroll + str_comma + str_paral + "}"
        else:
            str_modif = "= {" + str_unroll + str_comma + str_paral + "}"
    else:
        if b_is_graph_interface:
            str_modif = ": {}"
        else:
            str_modif = ""

    str_spec_atom = str_spec_atom + str_modif + ",\n"
    return str_spec_atom


# [Main function] Generate a string corresponding to the schedule descriptor, from the Ttile specification
#
# Inputs:
#  - scheme: the Ttile scheme (str)
#  - comp: the considered computation
#  - machine: the considered architecture
#  - b_is_graph_interface : True if we using the graph interface, False if we are using the mlir-loop interface
#   (there are slight changes in the syntax of a schedule between both interfaces)
# Output:
#  - str_descr_sched, ready to be "eval()"
def get_descr_sched(
    scheme: List[Atom], comp: Computation, machine: Archi, b_is_graph_interface: bool
):
    ldims = get_ldims_computation(comp)

    # Dictionnary: [lambda_loc] --> string of spec (to be "eval(...)")
    d_current_str_desc = dict()
    d_current_str_desc[""] = ""

    # Size (note: computation will be similar to "scheme::get_sizes_scheme")
    d_lsizes = dict()
    for dim in ldims:
        d_lsizes[dim] = [1]

    # Quick pass to know if an atom is the last one
    lb_lastatom_dim = []
    l_encountered_dim = []
    for i in range(len(scheme)):
        atom = scheme[len(scheme) - i - 1]

        # Case where there are no "atom.dim"
        if atom.type == AtomType.HOIST:
            lb_lastatom_dim.append(False)
            continue

        if atom.dim not in l_encountered_dim:
            lb_lastatom_dim.append(True)
            l_encountered_dim.append(atom.dim)
        else:
            lb_lastatom_dim.append(False)
    lb_lastatom_dim.reverse()

    assert len(lb_lastatom_dim) == len(scheme)
    # DEBUG
    # print(f"scheme={convert_scheme_to_str(scheme)}")
    # print(f"{lb_lastatom_dim=}")

    # For each atom, inner to outer
    for i_atom in range(len(scheme)):
        atom = scheme[i_atom]

        # DEBUG
        # print(f"Current atom = {atom}")
        # print(f"  - {d_lsizes=}")
        # print(f"  - {d_current_str_desc=}")

        dim_atom = atom.dim

        if atom.type == AtomType.VECT:
            # Vector size (in elements)
            size_vec = int(machine.vector_size / comp.elem_size)
            if b_is_graph_interface:
                str_spec_atom = f'"{dim_atom}#{size_vec}": ' + '{"vectorize" : None}\n'
            else:
                str_spec_atom = f'"{dim_atom}#{size_vec}"= ' + '{ "vectorize" }\n'
            # Note: no comma at the end since vect is suppose to be the first atom (last in the list)

            # We check that we are the first one
            assert "" in d_current_str_desc.keys()
            assert d_current_str_desc[""] == ""

            # Commit
            d_current_str_desc[""] = str_spec_atom
            d_lsizes[dim_atom] = [size_vec]

        elif atom.type == AtomType.HOIST:
            continue  # Skip

        elif atom.type in [
            AtomType.UNROLL,
            AtomType.TILE,
            AtomType.TILE_PARAL,
            AtomType.TILE_PARTIAL,
        ]:
            # Lambda loc stays the same

            # Gather infos
            if atom.type != AtomType.TILE_PARTIAL:
                ratio = atom.ratio
                b_unroll = atom.type == AtomType.UNROLL
                b_paral = atom.type == AtomType.TILE_PARAL

                # We update d_lsizes[dim_atom]
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = lsize_prev[i] * atom.ratio
                # Note: no need to update due to pointer magic in Python :)
            else:
                # We have a size (instead of a ratio) specified
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = atom.size

            # For each lambda loc...
            for str_lambda_loc in d_current_str_desc.keys():
                str_desc = d_current_str_desc[str_lambda_loc]

                # Note: ibr=0 if the dimension is not inside str_lambda_loc
                ibr = recover_branchid_from_stringified(str_lambda_loc, dim_atom)
                size_atom = lsize_prev[ibr]

                str_spec_atom = _get_spec_atom_ratio(
                    dim_atom,
                    size_atom,
                    lb_lastatom_dim[i_atom],
                    b_unroll,
                    b_paral,
                    b_is_graph_interface,
                )

                # Commit
                n_str_desc = str_spec_atom + str_desc
                d_current_str_desc[str_lambda_loc] = n_str_desc

            # Note: d_current_str_desc was correctly updated - no need for a commit here

        elif atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
            # Lambda_loc might change: we check this
            str_lambda_loc_rand = list(d_current_str_desc.keys())[0]
            ldim_prev_lambda_loc = get_list_dims_str_lambda_loc(str_lambda_loc_rand)

            if dim_atom in ldim_prev_lambda_loc:
                # Case where the lambda on this dim is not the first one (lambda_loc are already ok)

                # Gather infos
                b_unroll = atom.type == AtomType.ULAMBDA
                b_paral = False

                # We update d_lsizes[dim_atom]
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = lsize_prev[i] * atom.lratios[i]
                # Note: no need to update due to pointer magic in Python :)

                # For each lambda loc, that stays the same
                for str_lambda_loc in d_current_str_desc.keys():
                    str_desc = d_current_str_desc[str_lambda_loc]

                    ibr = recover_branchid_from_stringified(str_lambda_loc, dim_atom)
                    size_atom = lsize_prev[ibr]

                    str_spec_atom = _get_spec_atom_ratio(
                        dim_atom,
                        size_atom,
                        lb_lastatom_dim[i_atom],
                        b_unroll,
                        b_paral,
                        b_is_graph_interface,
                    )

                    # Commit
                    n_str_desc = str_spec_atom + str_desc
                    d_current_str_desc[str_lambda_loc] = n_str_desc
            else:
                n_branches = len(atom.lratios)

                # The lambda is new: we need to adapt & duplicate the str
                assert len(d_lsizes[dim_atom]) == 1
                size_prev = d_lsizes[dim_atom][0]

                # Update the size in this dimension
                nlsize = []
                for i in range(n_branches):
                    nlsize.append(size_prev * atom.lratios[i])
                d_lsizes[dim_atom] = nlsize

                # Create the new lambda loc
                n_d_current_str_desc = dict()
                for str_lambda_loc_prev in d_current_str_desc.keys():
                    d_lambda_loc_prev = recover_all_branchid_from_stringified(
                        str_lambda_loc_prev
                    )

                    for i_br in range(n_branches):
                        d_curr_lambda = d_lambda_loc_prev.copy()
                        d_curr_lambda[dim_atom] = i_br
                        str_lambda_loc = stringify_lambda_choice(ldims, d_curr_lambda)

                        # Schedule
                        str_desc = d_current_str_desc[str_lambda_loc_prev]
                        size_atom = nlsize[i_br]

                        str_spec_atom = _get_spec_atom_ratio(
                            dim_atom,
                            size_atom,
                            lb_lastatom_dim[i_atom],
                            b_unroll,
                            b_paral,
                            b_is_graph_interface,
                        )

                        n_str_desc = str_spec_atom + str_desc

                        # Commit
                        n_d_current_str_desc[str_lambda_loc] = n_str_desc

                # Update
                d_current_str_desc = n_d_current_str_desc

        elif atom.type == AtomType.SEQ:
            # The lambda loc reduces

            # Computing the new size list for this dim
            lsizes_dim_prev = d_lsizes[dim_atom]  # Keeping it in memory
            n_branches = len(lsizes_dim_prev)

            # Updating the d_lsizes
            n_size_dim = sum(lsizes_dim_prev)
            d_lsizes[dim_atom] = [n_size_dim]

            # Regrouping the entry of d_current_str_desc by new lambda
            d_lbranches: dict[str, List[str]] = dict()
            for str_lambda_loc_prev in d_current_str_desc.keys():
                str_lambda_loc = remove_dim_from_stringified(
                    str_lambda_loc_prev, dim_atom, ldims
                )
                i_br = recover_branchid_from_stringified(str_lambda_loc_prev, dim_atom)

                if str_lambda_loc in d_lbranches.keys():
                    d_lbranches[str_lambda_loc][i_br] = d_current_str_desc[
                        str_lambda_loc_prev
                    ]
                else:
                    nl_str_desc = [""] * n_branches
                    nl_str_desc[i_br] = d_current_str_desc[str_lambda_loc_prev]
                    d_lbranches[str_lambda_loc] = nl_str_desc

            # Creating the new d_current_str_desc from d_lbranches
            d_current_str_desc = dict()
            for str_lambda_loc in d_lbranches.keys():
                lbranches = d_lbranches[str_lambda_loc]

                str_desc = ""
                start_branch = 0
                for i_br in range(len(lbranches)):
                    str_br = lbranches[i_br]
                    end_branch = start_branch + lsizes_dim_prev[i_br]

                    str_desc = (
                        str_desc
                        + f'"{dim_atom}[{start_branch}:{end_branch}]" :'
                        + "{"
                        + str_br
                        + "}"
                    )

                    if i_br < (len(lbranches) - 1):
                        str_desc = str_desc + ",\n"

                    start_branch = end_branch
                str_desc = str_desc + "\n"

                d_current_str_desc[str_lambda_loc] = str_desc
        else:
            raise ValueError(f"get_descr_sched :: Unrecognized operator {atom.type}")

    # At the end, only "" should remain
    str_descr_sched = d_current_str_desc[""]
    str_descr_sched = "{\n" + str_descr_sched + "}"
    return str_descr_sched


# ===========================================================================

# 3) Launch scheme execution & measurement through xdsl-transform "mlir-loops"
# Interface with Hugo's "mlir-loop" that generates transform dialect, must go through MLIR compilation pipeline


# Replicate the "avx2_matmult.mlir" from xdsl-transform
# MATMULT: linalg.matmul
#     https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop
def build_xdsl_module_string_matmul(
    comp: Computation,
    machine: Archi,
    scheme: List[Atom],
    dsizes: dict[str, int],
    b_initOut: bool = False,
) -> str:
    # Recover the individual problem sizes
    default_sizes = {"i": 1, "j": 1, "k": 1}
    dsizes = default_sizes | dsizes
    size_i = dsizes["i"]
    size_j = dsizes["j"]
    size_k = dsizes["k"]

    if comp.elem_size == 4:
        fXX = "f32"
    elif comp.elem_size == 8:
        fXX = "f64"
    else:
        raise ValueError(
            f"build_xdsl_module_op_matmul :: unknown type of float of size {comp.elem_size} octets"
        )

    num_elem_per_vec = int(machine.vector_size / comp.elem_size)  # Both are in octets

    # Model: "avx2_matmul.mlir"
    str_output = f"func.func @myfun(\n"
    str_output += f"  %A: memref<{size_i}x{size_k}x{fXX}>,\n"
    str_output += f"  %B: memref<{size_k}x{size_j}x{fXX}>,\n"
    str_output += f"  %C: memref<{size_i}x{size_j}x{fXX}>\n"
    str_output += ") {\n"

    # Initialization of the output
    if b_initOut:
        str_output += f"  %cst = arith.constant 0.000 : {fXX}\n"
        str_output += f"  linalg.fill\n"
        str_output += "    {\n"
        str_output += '      loop.dims = ["i", "j"],\n'
        # str_output += f"      loop.parallel_dims = [\"i\",\"j\"],\n"
        # str_output += f"      loop.reduction_dims = [],\n"
        str_output += "      loop.tiles_names = {"
        str_output += f'"j" = ["j1"]'
        str_output += "},\n"
        str_output += "      loop.tiles_sizes = {"
        str_output += f"j1 = {num_elem_per_vec}"
        str_output += "},\n"
        str_output += f'      loop.interchange = ["i","j","j1"],\n'
        str_output += f'      loop.vectorize = ["j1"]\n'
        # Bad idea to add a loop parallelize on i here  (tank the perfs)
        str_output += "    }\n"
        str_output += f"    ins(%cst : {fXX})\n"
        str_output += f"    outs(%C : memref<{size_i}x{size_j}x{fXX}>)\n"

    str_output += f"  linalg.matmul\n"
    str_output += "    {\n"

    # This is the part where we plug the scheme in the text
    str_output += '      loop.dims = ["i","j","k"],\n'
    str_loop_sched = get_descr_sched(scheme, comp, machine, False)
    str_output += f"      loop.schedule = {str_loop_sched}\n"
    str_output += "    }\n"
    str_output += f"    ins(%A, %B : memref<{size_i}x{size_k}x{fXX}>, memref<{size_k}x{size_j}x{fXX}>)\n"
    str_output += f"    outs(%C : memref<{size_i}x{size_j}x{fXX}>)\n"
    str_output += f"  return\n"
    str_output += "}\n"

    return str_output


# Same than above, but for convolution, for xdsl-transform
# CONV: linalg.conv_2d_nhwc_hwcf
#     https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nhwc_hwcf-linalgconv2dnhwchwcfop
def build_xdsl_module_string_conv(
    comp, machine, scheme, dsizes, b_initOut=False
) -> str:
    # There are several naming notation for the conv2D image/kernel dimensions:
    #   X / Y + H / W
    # or
    #   H / W + R / S
    # => Preprocessing the first set into the second one, to uniform notations
    b_conv_dim_renaming = False
    for atom in scheme:
        # Ignore the atom with no dims
        if atom in [AtomType.HOIST]:
            continue
        if atom.dim in ["x", "y"]:
            b_conv_dim_renaming = True
    if b_conv_dim_renaming:
        scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
        dsizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes)

    if "strx" in dsizes:
        assert dsizes["strx"] == 1  # TODO: Stride not managed yet.
    if "stry" in dsizes:
        assert dsizes["stry"] == 1  # TODO: Stride not managed yet.

    # Recover the individual problem sizes
    size_n = 1  # Batch dim == 1

    default_sizes = {"n": 1, "f": 1, "c": 1, "h": 1, "w": 1, "r": 1, "s": 1}
    dsizes = default_sizes | dsizes

    # Output/Input feature dimensions
    size_f = dsizes["f"]
    size_c = dsizes["c"]
    size_h = dsizes["h"]
    size_w = dsizes["w"]
    size_r = dsizes["r"]
    size_s = dsizes["s"]

    if comp.elem_size == 4:
        fXX = "f32"
    elif comp.elem_size == 8:
        fXX = "f64"
    else:
        raise ValueError(
            f"build_xdsl_module_op_matmul :: unknown type of float of size {comp.elem_size} octets"
        )

    num_elem_per_vec = int(machine.vector_size / comp.elem_size)  # Both are in octets

    # Let's do it!
    str_output = f"func.func @myfun(\n"
    str_output += f"  %I: memref<{size_n}x{(size_h + size_r - 1)}x{(size_w + size_s - 1)}x{size_c}x{fXX}>,\n"
    str_output += f"  %K: memref<{size_r}x{size_s}x{size_c}x{size_f}x{fXX}>,\n"
    str_output += f"  %O: memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>\n"
    str_output += ") {\n"

    # Initialization of the output
    if b_initOut:
        str_output += f"  %cst = arith.constant 0.000 : {fXX}\n"
        str_output += f"  linalg.fill\n"
        str_output += "    {\n"
        str_output += '      loop.dims = ["n","h","w","f"],\n'
        # str_output += f"      loop.parallel_dims = [\"n\",\"h\",\"w\",\"f\"],\n"
        # str_output += f"      loop.reduction_dims = [],\n"
        str_output += "      loop.tiles_names = {"
        str_output += f'"f" = ["f1"]'
        str_output += "},\n"
        str_output += "      loop.tiles_sizes = {"
        str_output += f"f1 = {num_elem_per_vec}"
        str_output += "},\n"
        str_output += f'      loop.interchange = ["n","h","w","f","f1"],\n'
        str_output += f'      loop.vectorize = ["f1"]\n'
        # Note: this is a bad idea to add a loop.parallelize on n/h here
        str_output += "    }\n"
        str_output += f"    ins(%cst : {fXX})\n"
        str_output += (
            f"    outs(%O : memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>)\n"
        )

    # Version direct operator (not available yet in xdsl)
    # str_output += f"  linalg.conv_2d_nhwc_hwcf {\n"

    # [MOD] Version of "conv_2d_nhwc_hwcf" with a generic
    str_output += "  linalg.generic {\n"
    str_output += "      indexing_maps = [\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],\n"
    str_output += (
        '      iterator_types = ["parallel", "parallel", "parallel", "parallel",\n'
    )
    str_output += '          "reduction", "reduction", "reduction"]\n'
    str_output += "    }\n"
    str_output += f"    ins(%I, %K : memref<{size_n}x{(size_h + size_r - 1)}x{(size_w + size_s - 1)}x{size_c}x{fXX}>,"
    str_output += f" memref<{size_r}x{size_s}x{size_c}x{size_f}x{fXX}>)\n"
    str_output += f"    outs(%O : memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>)\n"
    str_output += "     attrs = {\n"

    # This is the part where we plug the scheme in the text
    str_output += '      loop.dims = ["n","h","w","f","r","s","c"],\n'
    str_loop_sched = get_descr_sched(scheme, comp, machine, False)
    str_output += f"      loop.schedule = {str_loop_sched}\n"
    str_output += "     }\n"

    # [MOD] Version of "conv_2d_nhwc_hwcf" with a generic
    str_output += "  {\n"
    str_output += f"    ^bb0(%0: {fXX}, %1: {fXX}, %2: {fXX}) :\n"
    str_output += f"      %3 = arith.mulf %0, %1 : {fXX}\n"
    str_output += f"      %4 = arith.addf %2, %3 : {fXX}\n"
    str_output += f"      linalg.yield %4 : {fXX}\n"
    str_output += "  }\n"
    str_output += f"  return\n"
    str_output += "}\n"

    return str_output


# Launch through the "mlir_loop.py" Python script - old syntax (since April 25)
def launch_and_measure_scheme(
    comp: Computation, machine: Archi, scheme: List[Atom], dsizes: dict[str, int]
):
    # Recover the text of the input
    if comp.spec == Computation_spec.MATMULT:
        str_input_xdsltransf = build_xdsl_module_string_matmul(
            comp, machine, scheme, dsizes
        )
    elif comp.spec == Computation_spec.CONV:
        str_input_xdsltransf = build_xdsl_module_string_conv(
            comp, machine, scheme, dsizes
        )
    else:
        raise ValueError("launch_and_measure_scheme : Unknown computation spec.")

    # Write it inside a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete_on_close=False) as finput:
        finput.write(str_input_xdsltransf)
        finput.close()

        # Number of elements inside a vector
        vector_size_in_elem = int(machine.vector_size / comp.elem_size)

        # Launch mlir-loop (from xdsl-transform) on this input, with the right options
        with tempfile.NamedTemporaryFile(mode="r", delete_on_close=False) as ftime_meas:
            cmd_xdsl_transf = f"mlir-loop {finput.name}"
            # cmd_xdsl_transf += f" --llvm-dir={llvm_build_installation}"
            cmd_xdsl_transf += f" --evaluate --no-alias --init-zero --vectors-size {vector_size_in_elem} "
            cmd_xdsl_transf += f" > {ftime_meas.name}"

            # DEBUG
            # print(cmd_xdsl_transf)
            os.system(cmd_xdsl_transf)

            # Note: possible to subst "mlir_loop" by "mlir_loop.py" if add the correct 2 lines at the end of its script.

            # Recover the time
            res_measurement = dict()
            for line in ftime_meas:
                res_measurement["time"] = float(line)
    # Delete both temporary files

    # Get the peak_perf
    # xdsl_transform got utils.py :: cpu_peak_time(ops: int, dtype: str, threads: int = 1)
    num_ops = compute_number_ops(comp, dsizes)
    if comp.elem_size == 4:
        dtype = "float32"
    elif comp.elem_size == 8:
        dtype = "float64"
    else:
        raise ValueError(
            f"launch_and_measure_scheme :: unknown type of float of size {comp.elem_size} octets"
        )
    peak_time = cpu_peak_time(num_ops, dtype)

    # DEBUG
    # print(f"measurement = {res_measurement['time']}")
    # print(f"{peak_time=}")

    # Compute peak_perf
    peak_perf = peak_time / res_measurement["time"]
    res_measurement["peak_perf"] = peak_perf

    return res_measurement


# ===========================================================================

# 4) Launch scheme execution & measurement through the XTC graph interface (in Python)
# Inspired from the "test_conv2d_r181_mlir.py" file from XTC


# Schedule to be specified here (convert Ttile to TVM-like instructions)
# Note: "xrc/itf/schd/scheduler.py"
def build_schedule_from_ttile(
    impl: Backend, comp: Computation, machine: Archi, scheme: List[Atom]
):
    sch = impl.get_scheduler()

    # ldims: the dims that appears in the scheme
    ldims = []
    for atom in scheme:
        if atom.dim not in ldims:
            ldims.append(atom.dim)

    name_op = str(comp.spec)

    str_descr_sched = get_descr_sched(scheme, comp, machine, True)
    spec_schedule = eval(str_descr_sched)

    # DEBUG
    # print(f"{spec_schedule=}")
    # print(f"{ldims=}")

    descript_scheduler(
        scheduler=sch, node_name=name_op, abstract_axis=ldims, spec=spec_schedule
    )

    # And run it!
    sched = sch.schedule()
    return sched


# Launch scheme execution & measurement through xdsl-transform script (higher level)
def launch_and_measure_scheme_graph_interf(
    comp: Computation,
    machine: Archi,
    scheme: List[Atom],
    dsizes: dict[str, int],
    backend: str,
    pmu_counters: list[str] = ["clocks"],
) -> dict[str, float]:
    # 1) Computation - described as a graph
    dtype = f"float{comp.elem_size * 8}"

    if comp.spec == Computation_spec.CONV:
        scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
        dsizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes)

        # Getting the sizes
        default_sizes = {
            "n": 1,
            "f": 1,
            "c": 1,
            "h": 1,
            "w": 1,
            "r": 1,
            "s": 1,
            "strx": 1,
            "stry": 1,
        }
        dsizes = default_sizes | dsizes

        size_n = dsizes["n"]
        size_f = dsizes["f"]
        size_c = dsizes["c"]
        size_h = dsizes["h"]
        size_w = dsizes["w"]
        size_r = dsizes["r"]
        size_s = dsizes["s"]

        assert dsizes["strx"] == 1  # TODO: For now...
        assert dsizes["stry"] == 1

        # Specifying the computation
        a = O.tensor((size_n, size_h + size_r - 1, size_w + size_s - 1, size_c), dtype)
        b = O.tensor((size_r, size_s, size_c, size_f), dtype)
        with O.graph(name="ttile_xdsl_conv2D_graph") as gb:
            O.conv2d(a, b, stride=(dsizes["strx"], dsizes["stry"]), name="Out")
    elif comp.spec == Computation_spec.MATMULT:
        # Getting the sizes
        default_sizes = {"i": 1, "j": 1, "k": 1}
        dsizes = default_sizes | dsizes
        size_i = dsizes["i"]
        size_j = dsizes["j"]
        size_k = dsizes["k"]

        # Specifying the computation
        a = O.tensor((size_i, size_k), dtype)
        b = O.tensor((size_k, size_j), dtype)
        with O.graph(name="matmul") as gb:
            O.matmul(a, b, name="C")
    else:
        raise ValueError(f"Unrecognized computation {comp}")

    # Build the graph
    graph = gb.graph

    # DEBUG
    # print(graph)

    # Backend
    if backend == "mlir":
        impl_backend: MlirBackend | TVMBackend = MlirBackend(
            graph, always_vectorize=True, no_alias=True
        )
    elif backend == "tvm":
        impl_backend = TVMBackend(graph, always_vectorize=True, no_alias=True)
    else:
        raise ValueError(f"Unrecognized backend: {backend}")

    # Schedule
    sched = build_schedule_from_ttile(impl_backend, comp, machine, scheme)

    # Compiler & Execute it
    compiler_args: dict[str, Any] = dict()
    evaluate_args: dict[str, Any] = dict()

    compiler = impl_backend.get_compiler(
        shared_lib=True,
        dump_file=f"compiled_file_{backend}",
        # *compiler_args
        # Options for debugging: print_source_ir=True,  print_transformed_ir=True
    )
    module = compiler.compile(sched)
    evaluator = module.get_evaluator(
        validate=True,
        # pmu_counters=pmu_counters
        # *evaluate_args
    )
    results, code, error = evaluator.evaluate()

    assert len(results) == 1
    time = results[0]

    # [OLD CODE]
    # For debugging
    # compiler_args["print_source_ir"] = True
    # time = impl_backend.evaluate(sched, compiler_args, evaluate_args)

    res_measurement = dict()
    res_measurement["time"] = float(time)

    # Get the peak_perf
    # xdsl_transform got utils.py :: cpu_peak_time(ops: int, dtype: str, threads: int = 1)
    num_ops = compute_number_ops(comp, dsizes)
    peak_time = cpu_peak_time(num_ops, dtype)

    # DEBUG
    # print(f"measurement = {res_measurement['time']}")
    # print(f"{peak_time=}")

    # Compute peak_perf
    peak_perf = peak_time / res_measurement["time"]
    res_measurement["peak_perf"] = peak_perf

    return res_measurement
