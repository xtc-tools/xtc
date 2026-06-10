#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from typing import List, Tuple
from typing_extensions import override
from enum import Enum
from dataclasses import dataclass, replace

# Manipulation of scheme, which describes the configuration choices made.

# ====================================================================


# Enumeration of the types of atoms
class AtomType(Enum):
    VECT = 1
    UNROLL = 2
    TILE = 3
    TILE_PARTIAL = 4
    HOIST = 5  # Not sure if needed in the codegen? :/
    ULAMBDA = 6  # Apply a different value to an unroll, depending on the sequential branch it is
    TLAMBDA = (
        7  # Apply a different value to a tile, depending on the sequential branch it is
    )
    SEQ = 8  # To be placed above a lambda (sequential composition)
    TILE_PARAL = 9  # For parallelism
    # Note "Ttile Lambda_apply" = Ulambda on the µkernel side + Tlambda above + Seq

    # TODO: add other? Packing/Padding ?
    #  SW Prefetch ?

    @override
    def __str__(self):
        match self:
            case AtomType.VECT:
                return "V"
            case AtomType.UNROLL:
                return "U"
            case AtomType.TILE:
                return "T"
            case AtomType.TILE_PARTIAL:
                return "Tpart"
            case AtomType.HOIST:
                return "Hoist_var"
            case AtomType.ULAMBDA:
                return "UL"
            case AtomType.TLAMBDA:
                return "TL"
            case AtomType.SEQ:
                return "Seq"
            case AtomType.TILE_PARAL:
                return "Tparal"
            case _:
                raise ValueError("Unknown AtomType in AtomType::__str__")


# Definition of an atom, as a single element of a scheme
@dataclass
class Atom:
    type: AtomType  # Type of an atom

    # Note: at most one of these is not "None"
    _dim: str | None = None  # Name of the dimension
    _ratio: int | None = None  # Ratio of the atom (for VECT, UNROLL, TILE, TILE_PARAL)
    _size: int | None = None  # Size of the atom (for TILE_PARTIAL)
    _lvars_hoist: List[str] | None = None  # HOIST: list of variables to be hoisted
    _lratios: List[int] | None = None  # ULAMBDA/TLAMBDA: list of ratio

    @property
    def dim(self) -> str:
        assert self._dim is not None
        return self._dim

    @property
    def ratio(self) -> int:
        assert self._ratio is not None
        return self._ratio

    @property
    def size(self) -> int:
        assert self._size is not None
        return self._size

    @property
    def lvars_hoist(self) -> list[str]:
        assert self._lvars_hoist is not None
        return self._lvars_hoist

    @property
    def lratios(self) -> list[int]:
        assert self._lratios is not None
        return self._lratios

    def with_dim(self, dim: str) -> "Atom":
        assert self.has_dim()
        return replace(self, _dim=dim)

    def has_dim(self) -> bool:
        return self.type != AtomType.HOIST

    @override
    def __str__(self):
        if self.type in [
            AtomType.VECT,
            AtomType.UNROLL,
            AtomType.TILE,
            AtomType.TILE_PARAL,
        ]:
            return f"{self.type}({self.dim},{self.ratio})"
        elif self.type == AtomType.TILE_PARTIAL:
            return f"{self.type}({self.dim},{self.size})"
        elif self.type == AtomType.HOIST:
            return f"{self.type}({self.lvars_hoist})"
        elif (self.type == AtomType.ULAMBDA) or (self.type == AtomType.TLAMBDA):
            return f"{self.type}({self.dim}, {self.lratios})"
        elif self.type == AtomType.SEQ:
            return f"{self.type}({self.dim})"
        else:
            raise ValueError("Unknown AtomTYPE in Atom::__str__")


# Constructors to use for atoms
def new_vect_atom(dim: str, vector_len: int) -> Atom:
    atom = Atom(AtomType.VECT, dim, _ratio=vector_len)
    return atom


def new_unroll_atom(dim: str, ratio: int) -> Atom:
    atom = Atom(AtomType.UNROLL, dim, _ratio=ratio)
    return atom


def new_tile_atom(dim: str, ratio: int) -> Atom:
    atom = Atom(AtomType.TILE, dim, _ratio=ratio)
    return atom


def new_partialtile_atom(dim: str, size: int) -> Atom:
    atom = Atom(AtomType.TILE_PARTIAL, dim, _size=size)
    return atom


def new_hoist_atom(lvars_hoist: List[str]) -> Atom:
    atom = Atom(AtomType.HOIST, _lvars_hoist=lvars_hoist)
    return atom


def new_unrollLambda_atom(dim: str, lratios: List[int]) -> Atom:
    atom = Atom(AtomType.ULAMBDA, dim, _lratios=lratios)
    return atom


def new_tileLambda_atom(dim: str, lratios: List[int]) -> Atom:
    atom = Atom(AtomType.TLAMBDA, dim, _lratios=lratios)
    return atom


def new_seq_atom(dim: str) -> Atom:
    atom = Atom(AtomType.SEQ, dim)
    return atom


def new_paralleltile_atom(dim: str, ratio: int) -> Atom:
    atom = Atom(AtomType.TILE_PARAL, dim, _ratio=ratio)
    return atom


# Scheme = list of atoms (because single statement case)


# MAIN FUNCTION for pretty-printing
def convert_scheme_to_str(scheme: List[Atom]) -> str:
    str_scheme = "["
    b_first = True
    for atom in scheme:
        if b_first:
            str_scheme += str(atom)
            b_first = False
        else:
            str_scheme += "; " + str(atom)
    str_scheme += "]"
    return str_scheme


def print_scheme(scheme: List[Atom]):
    print(convert_scheme_to_str(scheme))
    return


# ====================================================================

# 2) Converting a Ttile string into a Scheme object


# Auxilliary function that manage the string preprocessing, in order to extract
# the string corresponding to each atom
def _preprocess_scheme_str(str_scheme: str) -> List[str]:
    # Remove the leading/finishing empty chars
    str_scheme = str_scheme.strip()

    # Assume first and last char are [ and ]
    assert str_scheme[0] == "["
    assert str_scheme[-1] == "]"
    str_scheme = str_scheme[1:-1]

    # Consider each atom one by one
    latoms_str = str_scheme.split(";")

    # DEBUG
    # print(latoms_str)

    return latoms_str


# Aux function, parse atoms of the shape "U(d, 42)", "T(d, 20)", Tpart(d, 43) or "V(d, 16)"
def _parse_V_T_U_Tpart_Tpara_atom(str_atom: str) -> Tuple[str, int]:
    # Get the content of the parenthesis
    lstr_atom = str_atom.split("(")
    assert len(lstr_atom) == 2
    lstr_atom = lstr_atom[1].split(")")
    assert len(lstr_atom) <= 2

    # print(lstr_atom)

    lstr_atom = lstr_atom[0].split(",")
    dim = lstr_atom[0].strip().lower()
    ratio = int(lstr_atom[1].strip())

    return (dim, ratio)


# Aux function, parse atoms of the shape "Hoist_var([O, I])"
def _parse_hoistvar_atom(str_atom: str) -> List[str]:
    # Get the content of the brackets
    lstr_atom = str_atom.split("[")
    assert len(lstr_atom) == 2
    lstr_atom = lstr_atom[1].split("]")
    assert len(lstr_atom) <= 2

    # Get the elements of the list of variables
    lstr_vars = lstr_atom[0].split(",")
    lvars_hoist = [elem.strip().strip("'\"") for elem in lstr_vars]
    return lvars_hoist


# Aux function, parse atoms of the shape "UL(d, [3,4])" or "TL(d, [ 12, 14 ])"
def _parse_Ulambda_Tlambda_atom(str_atom: str) -> Tuple[str, List[int]]:
    # Get the content of the parenthesis
    lstr_atom = str_atom.split("(")
    assert len(lstr_atom) == 2
    lstr_atom = lstr_atom[1].split(")")
    assert len(lstr_atom) <= 2
    str_temp = lstr_atom[0].strip()

    # Recover the dim (text before the first "," )
    ind = str_temp.find(",")
    str_dim = str_temp[:ind]
    str_dim = str_dim.strip().lower()
    str_lvalues = str_temp[ind + 1 :]

    # Get the content of the brackets
    lstr_lvalues = str_lvalues.split("[")
    assert len(lstr_lvalues) == 2
    lstr_lvalues = lstr_lvalues[1].split("]")
    assert len(lstr_lvalues) <= 2

    # Get the elements of the list of variables
    lstr_vars = lstr_lvalues[0].split(",")
    lvalues = [int(elem) for elem in lstr_vars]
    return str_dim, lvalues


# Aux function, parse atoms of the shape "Seq(d)"
def _parse_seq_atom(str_atom: str) -> str:
    # Get the content of the parenthesis
    lstr_atom = str_atom.split("(")
    assert len(lstr_atom) == 2
    lstr_atom = lstr_atom[1].split(")")
    assert len(lstr_atom) <= 2

    dim = lstr_atom[0].strip().lower()
    return dim


# MAIN FUNCTION for parsing
def build_scheme_from_str(ttile_str: str) -> List[Atom]:
    scheme = []

    latom_str = _preprocess_scheme_str(ttile_str)

    for str_atom in latom_str:
        # Quick textual preprocessing
        str_atom = str_atom.strip()
        if str_atom.startswith("("):
            assert str_atom[-1] == ")"
            str_atom = str_atom[1:-1]
            str_atom = str_atom.strip()

        # DEBUG
        # print(f"str_atom = \"{str_atom}\"")

        if (str_atom[0] in ["V", "U", "T"]) and (
            not str_atom.startswith("UL") and (not str_atom.startswith("TL"))
        ):
            dim, ratio = _parse_V_T_U_Tpart_Tpara_atom(str_atom)

            if str_atom.startswith("Tpart"):
                # Actually not a ratio, but a size
                atom = new_partialtile_atom(dim, ratio)
            elif str_atom.startswith("Tparal"):
                atom = new_paralleltile_atom(dim, ratio)
            elif str_atom[0] == "V":
                atom = new_vect_atom(dim, ratio)
            elif str_atom[0] == "U":
                atom = new_unroll_atom(dim, ratio)
            elif str_atom[0] == "T":
                atom = new_tile_atom(dim, ratio)
            else:
                raise ValueError(
                    "build_scheme_from_str - atom should be a V,U,T,Tpart here"
                )
            scheme.append(atom)

        elif str_atom.startswith("Hoist_var"):
            lvars_hoist = _parse_hoistvar_atom(str_atom)
            atom = new_hoist_atom(lvars_hoist)
            scheme.append(atom)

        elif str_atom.startswith("Seq"):
            dim = _parse_seq_atom(str_atom)
            atom = new_seq_atom(dim)
            scheme.append(atom)

        elif str_atom.startswith("UL"):
            dim, lratios = _parse_Ulambda_Tlambda_atom(str_atom)
            atom = new_unrollLambda_atom(dim, lratios)
            scheme.append(atom)
        elif str_atom.startswith("TL"):
            dim, lratios = _parse_Ulambda_Tlambda_atom(str_atom)
            atom = new_tileLambda_atom(dim, lratios)
            scheme.append(atom)

        else:
            raise ValueError("build_scheme_from_str - atom not supported: {str_atom}")

    return scheme


# Normalize a scheme, by removing unroll/tiling of ratio 1
def normalize_scheme(scheme: List[Atom]) -> List[Atom]:
    nscheme = []
    for atom in scheme:
        if (atom.type == AtomType.UNROLL) and atom.ratio == 1:
            continue
        if (atom.type == AtomType.TILE) and atom.ratio == 1:
            continue
        if (atom.type == AtomType.TILE_PARAL) and atom.ratio == 1:
            continue

        # Default: keep it
        nscheme.append(atom)

    return nscheme


# ====================================================================

# 3) Coherency check


# MAIN FUNCTION to check if a given scheme is well-built
# Rules are:
# - V : must be the first one
# - U/UL : only after V,U,UL
# - UL/TL/Seq: if there is at least an UL/TL on a dimension, a Seq must also be here
# - The size of the list in UL/TL for a given dim must be the same
# - Seq: at most one Seq per dimension (for the ones with UL/TL)
# => 2 groups of non-interleavered UL/TL/Seq on the same dim is not allowed (hypothesis used in full_assoc_model)
# - Tpart: no mix on the dimension where there is a UL/TL (note: we might be fine if it happens before the UL/TL)
def check_coherency_scheme(scheme: List[Atom], verbose: bool = False) -> bool:
    b_above_mickern = False
    d_num_ratios: dict[str, int] = dict()
    l_dimseq = []

    for i in range(len(scheme)):
        atom = scheme[i]

        if (atom.type == AtomType.VECT) and (i > 0):
            if verbose:
                print(f"Atom {str(atom)} is a V which is not at the lowest dimension.")
            return False

        if atom.type in [
            AtomType.TILE,
            AtomType.TILE_PARTIAL,
            AtomType.TLAMBDA,
            AtomType.TILE_PARAL,
        ]:
            b_above_mickern = True
        if (atom.type in [AtomType.UNROLL, AtomType.ULAMBDA]) and b_above_mickern:
            if verbose:
                print(f"Atom {str(atom)} is an unroll above at least a tile dimension.")
            return False

        if atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
            if atom.dim in d_num_ratios:
                old_num_ratios = d_num_ratios[atom.dim]
                if len(atom.lratios) != old_num_ratios:
                    if verbose:
                        print(
                            f"Atom {str(atom)} has {len(atom.lratios)} branches, {old_num_ratios} branches were expected."
                        )
                    return False
            else:
                d_num_ratios[atom.dim] = len(atom.lratios)
        if atom.type == AtomType.SEQ:
            if atom.dim not in d_num_ratios:
                if verbose:
                    print(
                        f"Atom {str(atom)} does not have any corresponding UL/TL on this dimension."
                    )
                return False
            else:
                if atom.dim in l_dimseq:
                    if verbose:
                        print(f"Atom {str(atom)} is the second Seq on this dimension.")
                    return False
                l_dimseq.append(atom.dim)

        if atom.type == AtomType.TILE_PARTIAL:
            if atom.dim in d_num_ratios:
                if verbose:
                    print(
                        f"Atom {str(atom)} is a partial tiling on a dimension with an UL/TL."
                    )
                return False

    # At the end of the scheme:
    # Check that all dims with a UL/TL had a corresponding Seq
    for dim in d_num_ratios.keys():
        if dim not in l_dimseq:
            if verbose:
                print(f"Dimension {dim} has UL/TL, but no corresponding Seq.")
            return False

    # All checks passed!
    return True


# ====================================================================

# 4) Stringified lambda localisation utilities


# [Aux function] Stringify the choice of branch for multiple lambda
# This is to use these string as the key of a "dict()" in the footprint
def stringify_lambda_choice(
    ldims: List[str], d_dim_i_lambda_branch: dict[str, int]
) -> str:
    # Check that entries of d_dim_i_lambda_branch are in ldims
    for k in d_dim_i_lambda_branch.keys():
        if k not in ldims:
            raise ValueError(
                f'stringify_lambda_choice :: unknown dimension in lambda choice "{k}"'
            )

    # We use ldims to impose an order on the different dimensions of the choice
    str_lambda_loc = ""
    for d in ldims:
        if d in d_dim_i_lambda_branch:
            str_lambda_loc = str_lambda_loc + f"{d}*{d_dim_i_lambda_branch[d]}" + ","

    # Remove the last ",":
    str_lambda_loc = str_lambda_loc[:-1]
    return str_lambda_loc


# [Aux function] Opposite of stringify: given a string, recover the dim/branch id
def recover_all_branchid_from_stringified(str_lambda_loc: str) -> dict[str, int]:
    if len(str_lambda_loc) == 0:
        return dict()

    ldimbr = str_lambda_loc.split(",")
    d_dim_brid = dict()
    for str_dimbr in ldimbr:
        linfo_str_dimbr = str_dimbr.split("*")
        assert len(linfo_str_dimbr) == 2
        dim = linfo_str_dimbr[0]
        brid = linfo_str_dimbr[1]

        d_dim_brid[dim] = int(brid)
    return d_dim_brid


# [Aux function] Opposite of stringify: given a string and a dim, recover
# the branch id of this dimension.
#  If this dimension is not in str_lambda_loc, return "0" (single unsplited branch along this dim)
def recover_branchid_from_stringified(str_lambda_loc: str, dim: str) -> int:
    d_dim_brid = recover_all_branchid_from_stringified(str_lambda_loc)
    if dim in d_dim_brid.keys():
        return d_dim_brid[dim]
    else:
        return 0


# [Aux function] Remove a dimension from str_lambda_loc
def remove_dim_from_stringified(
    str_lambda_loc: str, dim: str, ldims_ref: List[str]
) -> str:
    d_dim_brid = recover_all_branchid_from_stringified(str_lambda_loc)
    assert dim in d_dim_brid
    del d_dim_brid[dim]
    n_str_lambda_loc = stringify_lambda_choice(ldims_ref, d_dim_brid)

    return n_str_lambda_loc


# [Aux function] Given a list of lambda_loc (typically the keys of a d_footprint) and a "dim",
# find the max number of lambda branches along this dimension.
def get_max_num_lambda_branch(l_str_lambda_loc: List[str], dim: str) -> int:
    n_lambda_id = 0
    for str_lambda_loc in l_str_lambda_loc:
        lambda_id = recover_branchid_from_stringified(str_lambda_loc, dim)
        n_lambda_id = max(n_lambda_id, lambda_id)
    return n_lambda_id + 1  # Because id start on "0"


# [Aux function] Recover the list of dimensions used in a str_lambda_loc
def get_list_dims_str_lambda_loc(str_lambda_loc: str) -> List[str]:
    if len(str_lambda_loc) == 0:
        return []

    ldimbr = str_lambda_loc.split(",")

    ldims_lambda_loc = []
    for str_dimbr in ldimbr:
        linfo_str_dimbr = str_dimbr.split("*")
        assert len(linfo_str_dimbr) == 2

        ldims_lambda_loc.append(linfo_str_dimbr[0])
    return ldims_lambda_loc


# [Aux function] Check if the branch infos of a stringified lambda loc is a subset of another stringified
# Return True iff the branches choices matches, and all dims of str_lambda_loc_1 are in str_lambda_loc_2
def is_stringify_subset(str_lambda_loc_1: str, str_lambda_loc_2: str) -> bool:
    # Corner case
    if str_lambda_loc_1 == "":
        return True

    ldimbr_1 = str_lambda_loc_1.split(",")
    ldimbr_2 = str_lambda_loc_2.split(",")

    for str_dimbr_1 in ldimbr_1:
        if str_dimbr_1 not in ldimbr_2:
            return False
    return True


# [Aux function] Identify the stringified lambda from "l_str_lloc" whose infos are a subset of "str_lloc_upp_set"
# Return the list of such subsets
def identify_subset_in_list_stringify(
    l_str_lloc: List[str], str_lloc_upp_set: str
) -> List[str]:
    d_dim_brid_upp_set = recover_all_branchid_from_stringified(str_lloc_upp_set)

    l_ret = []
    for str_lloc_cand in l_str_lloc:
        d_dim_brid_cand = recover_all_branchid_from_stringified(str_lloc_cand)

        # We check if all the entries of "d_dim_brid_cand" are a subset of "d_dim_brid_upp_set"
        b_is_subset = True
        for dim, brid_cand in d_dim_brid_cand.items():
            # Entry is here?
            if dim not in d_dim_brid_upp_set:
                b_is_subset = False
                break
            else:
                # Value corresponds?
                if brid_cand != d_dim_brid_upp_set[dim]:
                    b_is_subset = False
                    break

        # Commit
        if b_is_subset:
            l_ret.append(str_lloc_cand)

    return l_ret


# ====================================================================

# 5) Size of a scheme


# Compute the size of a scheme (to be compared with the size of the problem)
# Returns a dict whose keys are the dimension, and values is the list of sizes (at least with a single element)
#
# This function also works on prefix of scheme (ex: if there is not matching Seq to a UL/TL in a dim)
# When a lambda is not closed, this is the case where the list of sizes can contains more than a single element
#
# Returns "d_lsizes" : [dim_name] |--> [list of sizes, depending on the branch of lambda for this dim]
# If no lambda on a dim, the list is of size 1.
def get_sizes_scheme(scheme: List[Atom]) -> dict[str, List[int]]:
    d_lsizes = dict()

    for atom in scheme:
        if atom.type in [AtomType.HOIST]:
            continue

        # New dimension encountered
        if atom.dim not in d_lsizes:
            d_lsizes[atom.dim] = [1]

        lsizes = d_lsizes[atom.dim]
        if atom.type in [
            AtomType.VECT,
            AtomType.UNROLL,
            AtomType.TILE,
            AtomType.TILE_PARAL,
        ]:
            lsizes = [(elem * atom.ratio) for elem in lsizes]
            d_lsizes[atom.dim] = lsizes

        elif atom.type == AtomType.TILE_PARTIAL:
            assert len(lsizes) == 1  # No mix between partial tiles and lambda
            d_lsizes[atom.dim] = [atom.size]

        elif atom.type == AtomType.HOIST:
            assert False  # This case was supposed to be managed above
            continue

        elif atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
            dim = atom.dim
            lratios = atom.lratios

            if len(lsizes) == 1:
                # First lambda encountered
                lsizes = [(lsizes[0] * ratio) for ratio in atom.lratios]
                d_lsizes[atom.dim] = lsizes
            else:
                # Not the first lambda in this dimension, sizes must match
                assert len(lsizes) == len(atom.lratios)
                lsizes = [(lsizes[i] * atom.lratios[i]) for i in range(len(lsizes))]
                d_lsizes[atom.dim] = lsizes

        elif atom.type == AtomType.SEQ:
            acc = 0
            for elem in lsizes:
                acc += elem
            d_lsizes[atom.dim] = [acc]

        else:
            raise ValueError(f"get_sizes_scheme - Unrecognized atom {atom}")

    return d_lsizes


# [Aux function] Call get_sizes_scheme, then assume that all elements of the dict have a single element
def get_unlambda_sizes_scheme(compl_scheme: List[Atom]) -> dict[str, int]:
    d_lsizes = get_sizes_scheme(compl_scheme)

    d_ul_sizes = dict()
    for k in d_lsizes:
        lsizes = d_lsizes[k]
        if len(lsizes) != 1:
            raise ValueError(
                f"get_unlambda_sizes_scheme :: dimension {k} still have a lambda: {lsizes}"
            )
        d_ul_sizes[k] = lsizes[0]

    return d_ul_sizes


# Utility function - extract the vect/unrolled innermost part of a scheme
def extract_microkernel_scheme(scheme: List[Atom]) -> List[Atom]:
    subscheme = []
    for atom in scheme:
        if atom.type not in [AtomType.VECT, AtomType.UNROLL, AtomType.ULAMBDA]:
            break
        subscheme.append(atom)
    return subscheme


# MAIN FUNCTION to check if a scheme is coherent in term of dimension
# Rules are:
# - Tpart: size must be a multiple of the microkernel (U/UL dimension)
# - All: dimension of scheme must match the dimension of a problem
#   l_ignored_dims : entries of d_prob_dims to be ignored (ex: strides)
def check_dim_coherency_scheme(
    scheme: List[Atom],
    d_prob_dims: dict[str, int],
    l_ignored_dims: List[str] = [],
    verbose: bool = False,
):
    mickern_scheme = extract_microkernel_scheme(scheme)
    dsize_mickern = get_sizes_scheme(mickern_scheme)
    for atom in scheme:
        if atom.type == AtomType.TILE_PARTIAL:
            # If dimension not even in the microkernel: no issue
            if atom.dim not in dsize_mickern:
                continue

            lsize_dim = dsize_mickern[atom.dim]
            assert len(lsize_dim) == 1
            if (atom.size % lsize_dim[0]) != 0:
                if verbose:
                    print(
                        f"Atom {str(atom)} is a partial tile that cuts the microkernel (of size {lsize_dim[0]})."
                    )
                return False

    # Comparison with d_prob_dims
    dsize = get_sizes_scheme(scheme)
    for dim in d_prob_dims.keys():
        if dim in l_ignored_dims:
            continue

        if dim not in dsize.keys():
            # Dimension is not in the scheme => assume that its size is 1
            if d_prob_dims[dim] != 1:
                if verbose:
                    print(
                        f"Dimension {dim} is of size >1 while not being used in the scheme."
                    )
                return False
            else:
                continue

        if (d_prob_dims[dim] != 1) and (dim not in dsize):
            if verbose:
                print(f"Dimension {dim} was not found in the scheme.")
            return False

        assert len(dsize[dim]) == 1  # Scheme is complete (all lambda were closed)

        if dsize[dim][0] != d_prob_dims[dim]:
            if verbose:
                print(
                    f"On dimension {dim}, the scheme is of size {dsize[dim]}, while the problem is of size {d_prob_dims[dim]}."
                )
            return False
    for dim in dsize.keys():
        if dim not in d_prob_dims:
            if verbose:
                print(
                    f"Dimension {dim} was found in the scheme but not in the problem sizes."
                )
            return False

    # All checks passed!
    return True
