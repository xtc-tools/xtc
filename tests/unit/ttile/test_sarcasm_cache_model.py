
from xtc.schedules.ttile.scheme import build_scheme_from_str, get_sizes_scheme
from xtc.schedules.ttile.scheme_to_xtc import subst_dimname_xyhw_to_hwrs_conv2D_scheme
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.computation import get_list_array_contiguous_alloc, get_default_sizes
from xtc.schedules.ttile.cache_model.full_assoc_model import ReuseLoopStrat
from xtc.schedules.ttile.prob_sizes import ddsizes_Yolo, ddsizes_matmul, subst_dimname_xyhw_to_hwrs_conv2D_dsizes

from xtc.schedules.ttile.cache_model.sarcasm_set_assoc_model import build_maccess_func_coeff, build_bound_tiles_cacheline
from xtc.schedules.ttile.cache_model.sarcasm_set_assoc_model import repeat_and_rotate_dfp, sum_with_shift
from xtc.schedules.ttile.cache_model.sarcasm_set_assoc_model import dl_fp_direct_computation, arrange_ld_btcl_combi_dims
from xtc.schedules.ttile.cache_model.sarcasm_set_assoc_model import periodic_extra_cacheset_estimation_lvl, compute_cacheset_aware_comm_vol

# Test of the "build_maccess_func_coeff" function
def test_build_maccess_func_coeff_1():
	comp = Computation(Computation_spec.CONV, 4)
	prob_sizes = {"n" : 1, "f" :  64, "c":  64, "h": 112, "w": 112, "r": 3, "s": 3, "strx": 2, "stry": 2}
	cache_line_size = 16

	maccess_func_dim_name_coeff = build_maccess_func_coeff(comp, prob_sizes, cache_line_size)

	expected_result = { 'O': [(50176, 'n'), (448, 'h'), (4, 'w'), (1, 'f')],
		'I': [(204304, 'n'), (1808, 'h'), (904, 'r'), (8, 'w'), (4, 's'), (1, 'c')],
		'K': [(768, 'r'), (256, 's'), (4, 'c'), (1, 'f')] }
	assert(maccess_func_dim_name_coeff == expected_result)

	return


# Tests of the "build_bound_tiles_cacheline" function
def test_build_bound_tiles_cacheline_1():
	comp = Computation(Computation_spec.CONV, 4)
	prob_sizes = {"n" : 1, "f" :  64, "c":  64, "h": 112, "w": 112, "r": 3, "s": 3, "strx": 2, "stry": 2}
	cache_line_size = 16

	maccess_func_dim_name_coeff = build_maccess_func_coeff(comp, prob_sizes, cache_line_size)

	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	d_ltilesizes = get_sizes_scheme(scheme)
	d_def_sizes = get_default_sizes(comp)
	d_ltilesizes = d_def_sizes | d_ltilesizes

	llstride_dimname_O = maccess_func_dim_name_coeff["O"]
	ld_bound_tiles_cacheline_O = build_bound_tiles_cacheline(comp, llstride_dimname_O, d_ltilesizes, cache_line_size)

	# Result
	assert(d_ltilesizes=={'n': [1], 'h': [1], 'w': [9, 14], 'f': [64], 'r': [3], 's': [1], 'c': [8], 'strx': [1], 'stry': [1]})
	assert(llstride_dimname_O == [(50176, 'n'), (448, 'h'), (4, 'w'), (1, 'f')])
	assert(ld_bound_tiles_cacheline_O == [{'w*0': 1, 'w*1': 1}, {'w*0': 1, 'w*1': 1},
		{'w*0': 9, 'w*1': 14}, {'w*0': 4, 'w*1': 4}])

	return

def test_build_bound_tiles_cacheline_2():
	comp = Computation(Computation_spec.CONV, 4)
	prob_sizes = {"n" : 1, "f" :  64, "c":  64, "h": 112, "w": 112, "r": 3, "s": 3, "strx": 2, "stry": 2}
	cache_line_size = 16

	maccess_func_dim_name_coeff = build_maccess_func_coeff(comp, prob_sizes, cache_line_size)

	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	d_ltilesizes = get_sizes_scheme(scheme)
	d_def_sizes = get_default_sizes(comp)
	d_ltilesizes = d_def_sizes | d_ltilesizes

	llstride_dimname_I = maccess_func_dim_name_coeff["I"]
	ld_bound_tiles_cacheline_I = build_bound_tiles_cacheline(comp, llstride_dimname_I, d_ltilesizes, cache_line_size)

	# Result
	assert(d_ltilesizes == {'n': [1], 'h': [1], 'w': [9, 14], 'f': [64], 'r': [3], 's': [1], 'c': [8],
			'strx': [1], 'stry': [1]})
	assert(llstride_dimname_I == [(204304, 'n'), (1808, 'h'), (904, 'r'), (8, 'w'), (4, 's'), (1, 'c')])
	assert(ld_bound_tiles_cacheline_I ==[{'w*0': 1, 'w*1': 1}, {'w*0': 1, 'w*1': 1},
			{'w*0': 3, 'w*1': 3}, {'w*0': 9, 'w*1': 14},
			{'w*0': 1, 'w*1': 1}, {'w*0': 1, 'w*1': 1}])

	return


# Tests of the "repeat_and_rotate_dfp" function
def test_repeat_and_rotate_dfp_1():	
	dl_fp = {"" : [1,0,0,0]}
	last_dl_fp = {"" : [1,0,0,0]}
	d_ratio = {"" : 3}
	d_shift_dim = {"" : 1}
	Nset = 4

	ndl_fp = repeat_and_rotate_dfp(dl_fp, d_ratio, last_dl_fp, d_shift_dim, "d", Nset)
	assert(ndl_fp == {'': [1, 1, 1, 0]})

	return

def test_repeat_and_rotate_dfp_2():
	dl_fp = {"i*0" : [1,0,0,0], "i*1" : [0,1,0,0]}
	last_dl_fp = {"i*0" : [1,0,0,0], "i*1" : [0,1,0,0]}
	d_ratio = {"i*0" : 3, "i*1" : 6}
	d_shift_dim = {"i*0" : 1, "i*1" : 2}
	Nset = 4

	ndl_fp = repeat_and_rotate_dfp(dl_fp, d_ratio, last_dl_fp, d_shift_dim, "d", Nset)
	assert(ndl_fp == {'i*0': [1, 1, 1, 0], 'i*1': [0, 3, 0, 3]})

	return


# Test of the "sum_with_shift" function
def test_sum_with_shift_1():
	dl_fp = {"" : [42, 42, 42, 42] }
	last_dl_fp = {"i*0" : [1,0,0,0], "i*1" : [0,4,0,0]}
	d_lshift = {"" : [0, 1]}
	dim_seq = "i"
	Nset = 4

	ndl_fp = sum_with_shift(dl_fp, last_dl_fp, d_lshift, dim_seq, Nset)
	assert(ndl_fp == {'': [1, 0, 4, 0]})

	return


# Test of the "dl_fp_direct_computation" function
def test_dl_fp_direct_computation_1():
	comp = Computation(Computation_spec.CONV, 4)
	prob_sizes = {"n" : 1, "f" :  64, "c":  64, "h": 112, "w": 112, "r": 3, "s": 3, "strx": 2, "stry": 2}
	cache_line_size = 16

	maccess_func_dim_name_coeff = build_maccess_func_coeff(comp, prob_sizes, cache_line_size)

	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	d_ltilesizes = get_sizes_scheme(scheme)
	d_def_sizes = get_default_sizes(comp)
	d_ltilesizes = d_def_sizes | d_ltilesizes

	arr_name = "O"
	llstride_dimname_O = maccess_func_dim_name_coeff[arr_name]
	ld_bound_tiles_cacheline_O = build_bound_tiles_cacheline(comp, llstride_dimname_O, d_ltilesizes, cache_line_size)

	num_cache_set = 16
	lcont_arr_order = get_list_array_contiguous_alloc(comp)

	dl_fp = dl_fp_direct_computation(
		llstride_dimname_O, ld_bound_tiles_cacheline_O,
		comp, lcont_arr_order, arr_name, prob_sizes, num_cache_set, cache_line_size,
		ldim_ignore=[], starting_dl_fp=None
	)
	
	assert(dl_fp == {'w*0': [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
			'w*1': [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3]})

	return


# Tests of the "arrange_ld_btcl_combi_dims" function
def test_arrange_ld_btcl_combi_dims_1():
	ldims_combi = ['h', 'r']
	ll_stride_dimname = [[204304, 'n'], [1808, 'h'], [904, 'r'], [8, 'w'], [4, 's'], [1, 'c']]

	# Not consecutive (holes inside)
	ld_bound_tiles_cacheline = [{'': 1}, {'':10}, {'':1}, {'':1}, {'':1}, {'': 2}]
	
	ld_bound_tiles_cacheline = arrange_ld_btcl_combi_dims(ll_stride_dimname, ld_bound_tiles_cacheline, ldims_combi)
	assert(ld_bound_tiles_cacheline == [{'': 1}, {'': 10}, {'': 1}, {'': 1}, {'': 1}, {'': 2}])

	return

def test_arrange_ld_btcl_combi_dims_2():
	ldims_combi = ['h', 'r']
	ll_stride_dimname = [[204304, 'n'], [1808, 'h'], [904, 'r'], [8, 'w'], [4, 's'], [1, 'c']]

	# Consecutive interval
	ld_bound_tiles_cacheline = [{'': 1}, {'':10}, {'':3}, {'':1}, {'':1}, {'': 2}]

	ld_bound_tiles_cacheline = arrange_ld_btcl_combi_dims(ll_stride_dimname, ld_bound_tiles_cacheline, ldims_combi)
	assert(ld_bound_tiles_cacheline == [{'': 1}, {'': 1}, {'': 21}, {'': 1}, {'': 1}, {'': 2}])

	return

def test_arrange_ld_btcl_combi_dims_3():
	ldims_combi = ['h', 'r']
	ll_stride_dimname = [[45, 'n'], [9, 'h'], [3, 'r'], [3, 'w'], [1, 's'], [1, 'c']]

	# Consecutive interval
	ld_bound_tiles_cacheline = [{'': 1}, {'':5}, {'':1}, {'':1}, {'':1}, {'': 1}]

	ld_bound_tiles_cacheline = arrange_ld_btcl_combi_dims(ll_stride_dimname, ld_bound_tiles_cacheline, ldims_combi)
	assert(ld_bound_tiles_cacheline == [{'': 1}, {'': 5}, {'': 1}, {'': 1}, {'': 1}, {'': 1}])

	return


# Tests of the "periodic_extra_cacheset_estimation_lvl" function
def test_periodic_extra_cacheset_estimation_lvl_1():
	str_scheme = "[V(F,16); U(F,2); T(C,16)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	prob_sizes = {"n" : 1, "f" :  32, "c":  16, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 3, "stry": 3}

	comp = Computation(Computation_spec.CONV, 4)

	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	num_cache_set = 8
	cache_line_size = 16

	lddl_fp = periodic_extra_cacheset_estimation_lvl(scheme, comp, prob_sizes, lcont_arr_order, num_cache_set,
		cache_line_size, b_sanity_check=True)
	assert(lddl_fp==[
		{'O': {'': [1, 0, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 1, 0, 0, 0, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 1, 1, 0, 0, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [4, 4, 4, 4, 4, 4, 4, 4]}}])

	#lddl_fp = [[[
	#* Loop Level 0:
	#  - Array O :
	#    - "" : [1, 0, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 1, 0, 0, 0, 0]
	#* Loop Level 1:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 1, 1, 0, 0, 0]
	#* Loop Level 2:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [4, 4, 4, 4, 4, 4, 4, 4]
	#]]]

	return

def test_periodic_extra_cacheset_estimation_lvl_2():
	str_scheme = "[V(F,16); U(F,2); T(C,16); T(X,5)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	prob_sizes = {"n" : 1, "f" :  32, "c":  16, "h": 5, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}

	comp = Computation(Computation_spec.CONV, 4)

	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	num_cache_set = 8
	cache_line_size = 16

	lddl_fp = periodic_extra_cacheset_estimation_lvl(scheme, comp, prob_sizes, lcont_arr_order, num_cache_set,
		cache_line_size, b_sanity_check=True)
	assert(lddl_fp==[
		{'O': {'': [1, 0, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 0, 0, 0, 0, 1]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [1, 0, 0, 0, 0, 0, 0, 1]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [4, 4, 4, 4, 4, 4, 4, 4]}},
		{'O': {'': [2, 2, 1, 1, 1, 1, 1, 1]}, 'I': {'': [0, 0, 1, 1, 1, 1, 1, 0]}, 'K': {'': [4, 4, 4, 4, 4, 4, 4, 4]}}])
	#lddl_fp = [[[
	#* Loop Level 0:
	#  - Array O :
	#    - "" : [1, 0, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 0, 0, 0, 0, 1]
	#* Loop Level 1:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [1, 0, 0, 0, 0, 0, 0, 1]
	#* Loop Level 2:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [4, 4, 4, 4, 4, 4, 4, 4]
	#* Loop Level 3:
	#  - Array O :
	#    - "" : [2, 2, 1, 1, 1, 1, 1, 1]
	#  - Array I :
	#    - "" : [0, 0, 1, 1, 1, 1, 1, 0]
	#  - Array K :
	#    - "" : [4, 4, 4, 4, 4, 4, 4, 4]
	#]]]

	return

def test_periodic_extra_cacheset_estimation_lvl_3():
	str_scheme = "[V(F,16); U(F,2); T(C,16); T(F,2)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	prob_sizes = {"n" : 1, "f" :  64, "c":  16, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}

	comp = Computation(Computation_spec.CONV, 4)

	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	num_cache_set = 8
	cache_line_size = 16

	lddl_fp = periodic_extra_cacheset_estimation_lvl(scheme, comp, prob_sizes, lcont_arr_order, num_cache_set,
		cache_line_size, b_sanity_check=True)
	assert(lddl_fp==[
		{'O': {'': [1, 0, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 0, 0, 1, 0, 0, 0]}, 'K': {'': [0, 0, 0, 0, 0, 1, 0, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 0, 0, 1, 0, 0, 0]}, 'K': {'': [0, 0, 0, 0, 0, 1, 1, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 0, 0, 1, 0, 0, 0]}, 'K': {'': [0, 8, 8, 0, 0, 8, 8, 0]}},
		{'O': {'': [1, 1, 1, 1, 0, 0, 0, 0]}, 'I': {'': [0, 0, 0, 0, 1, 0, 0, 0]}, 'K': {'': [8, 8, 8, 8, 8, 8, 8, 8]}}])
	#lddl_fp = [[[
	#* Loop Level 0:
	#  - Array O :
	#    - "" : [1, 0, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 0, 0, 1, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 0, 0, 1, 0, 0]
	#* Loop Level 1:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 0, 0, 1, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 0, 0, 1, 1, 0]
	#* Loop Level 2:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 0, 0, 1, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 8, 8, 0, 0, 8, 8, 0]
	#* Loop Level 3:
	#  - Array O :
	#    - "" : [1, 1, 1, 1, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 0, 0, 1, 0, 0, 0]
	#  - Array K :
	#    - "" : [8, 8, 8, 8, 8, 8, 8, 8]
	#]]]

	return

def test_periodic_extra_cacheset_estimation_lvl_4():
	# Lambda/seq test
	str_scheme = "[V(F,16); U(F,2); T(C,16); TL(F,[2,3]); TL(F,[5,5]); Seq(F)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	prob_sizes = {"n" : 1, "f" :  800, "c":  16, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}

	comp = Computation(Computation_spec.CONV, 4)

	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	num_cache_set = 8
	cache_line_size = 16

	lddl_fp = periodic_extra_cacheset_estimation_lvl(scheme, comp, prob_sizes, lcont_arr_order, num_cache_set,
		cache_line_size, b_sanity_check=True)
	assert(lddl_fp==[
		{'O': {'': [1, 0, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 1, 0, 0, 0, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 1, 1, 0, 0, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [4, 4, 4, 4, 4, 4, 4, 4]}},
		{'O': {'f*0': [1, 1, 1, 1, 0, 0, 0, 0], 'f*1': [1, 1, 1, 1, 1, 1, 0, 0]},
			'I': {'f*0': [0, 0, 1, 0, 0, 0, 0, 0], 'f*1': [0, 0, 1, 0, 0, 0, 0, 0]},
			'K': {'f*0': [8, 8, 8, 8, 8, 8, 8, 8], 'f*1': [12, 12, 12, 12, 12, 12, 12, 12]}
		},
		{'O': {'f*0': [3, 3, 3, 3, 2, 2, 2, 2], 'f*1': [4, 4, 4, 4, 4, 4, 3, 3]},
			'I': {'f*0': [0, 0, 1, 0, 0, 0, 0, 0], 'f*1': [0, 0, 1, 0, 0, 0, 0, 0]},
			'K': {'f*0': [40, 40, 40, 40, 40, 40, 40, 40], 'f*1': [60, 60, 60, 60, 60, 60, 60, 60]}
		},
		{'O': {'': [7, 7, 6, 6, 6, 6, 6, 6]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [100, 100, 100, 100, 100, 100, 100, 100]}}])
	#lddl_fp = [[[
	#* Loop Level 0:
	#  - Array O :
	#    - "" : [1, 0, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 1, 0, 0, 0, 0]
	#* Loop Level 1:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [0, 0, 0, 1, 1, 0, 0, 0]
	#* Loop Level 2:
	#  - Array O :
	#    - "" : [1, 1, 0, 0, 0, 0, 0, 0]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [4, 4, 4, 4, 4, 4, 4, 4]
	#* Loop Level 3:
	#  - Array O :
	#    - "f*0" : [1, 1, 1, 1, 0, 0, 0, 0]
	#    - "f*1" : [1, 1, 1, 1, 1, 1, 0, 0]
	#  - Array I :
	#    - "f*0" : [0, 0, 1, 0, 0, 0, 0, 0]
    #	 - "f*1" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "f*0" : [8, 8, 8, 8, 8, 8, 8, 8]
	#    - "f*1" : [12, 12, 12, 12, 12, 12, 12, 12]
	#* Loop Level 4:
	#  - Array O :
	#    - "f*0" : [3, 3, 3, 3, 2, 2, 2, 2]
	#    - "f*1" : [4, 4, 4, 4, 4, 4, 3, 3]
	#  - Array I :
	#    - "f*0" : [0, 0, 1, 0, 0, 0, 0, 0]
    #	 - "f*1" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "f*0" : [40, 40, 40, 40, 40, 40, 40, 40]
	#    - "f*1" : [60, 60, 60, 60, 60, 60, 60, 60]
	#* Loop Level 5:
	#  - Array O :
	#    - "" : [7, 7, 6, 6, 6, 6, 6, 6]
	#  - Array I :
	#    - "" : [0, 0, 1, 0, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [100, 100, 100, 100, 100, 100, 100, 100]
	#]]]

	return

def test_periodic_extra_cacheset_estimation_lvl_5():
	# Normal "T" after a lambda
	str_scheme = "[V(F,16); U(F,2); T(C,16); TL(F,[2,3]); T(C,2); TL(F,[5,5]); Seq(F); T(C,2)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	prob_sizes = {"n" : 1, "f" :  800, "c":  64, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}

	comp = Computation(Computation_spec.CONV, 4)

	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	num_cache_set = 8
	cache_line_size = 16

	lddl_fp = periodic_extra_cacheset_estimation_lvl(scheme, comp, prob_sizes, lcont_arr_order, num_cache_set,
		cache_line_size, b_sanity_check=True)
	assert(lddl_fp==[
		{'O': {'': [1, 0, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 0, 0, 0, 1, 0]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [0, 0, 0, 0, 0, 0, 1, 1]}},
		{'O': {'': [1, 1, 0, 0, 0, 0, 0, 0]}, 'I': {'': [0, 0, 1, 0, 0, 0, 0, 0]}, 'K': {'': [4, 4, 4, 4, 4, 4, 4, 4]}},
		{'O': {'f*0': [1, 1, 1, 1, 0, 0, 0, 0], 'f*1': [1, 1, 1, 1, 1, 1, 0, 0]},
			'I': {'f*0': [0, 0, 1, 0, 0, 0, 0, 0], 'f*1': [0, 0, 1, 0, 0, 0, 0, 0]},
			'K': {'f*0': [8, 8, 8, 8, 8, 8, 8, 8], 'f*1': [12, 12, 12, 12, 12, 12, 12, 12]}
		},
		{'O': {'f*0': [1, 1, 1, 1, 0, 0, 0, 0], 'f*1': [1, 1, 1, 1, 1, 1, 0, 0]},
			'I': {'f*0': [0, 0, 1, 1, 0, 0, 0, 0], 'f*1': [0, 0, 1, 1, 0, 0, 0, 0]},
			'K': {'f*0': [16, 16, 16, 16, 16, 16, 16, 16], 'f*1': [24, 24, 24, 24, 24, 24, 24, 24]}
		},
		{'O': {'f*0': [3, 3, 3, 3, 2, 2, 2, 2], 'f*1': [4, 4, 4, 4, 4, 4, 3, 3]},
			'I': {'f*0': [0, 0, 1, 1, 0, 0, 0, 0], 'f*1': [0, 0, 1, 1, 0, 0, 0, 0]},
			'K': {'f*0': [80, 80, 80, 80, 80, 80, 80, 80], 'f*1': [120, 120, 120, 120, 120, 120, 120, 120]}
		},
		{'O': {'': [7, 7, 6, 6, 6, 6, 6, 6]}, 'I': {'': [0, 0, 1, 1, 0, 0, 0, 0]}, 'K': {'': [200, 200, 200, 200, 200, 200, 200, 200]}},
		{'O': {'': [7, 7, 6, 6, 6, 6, 6, 6]}, 'I': {'': [0, 0, 1, 1, 1, 1, 0, 0]}, 'K': {'': [400, 400, 400, 400, 400, 400, 400, 400]}}])
	# Results start to be too big: here is the 3 last levels:
	#* Loop Level 5:
	#  - Array O :
	#    - "f*0" : [3, 3, 3, 3, 2, 2, 2, 2]
	#    - "f*1" : [4, 4, 4, 4, 4, 4, 3, 3]
	#  - Array I :
	#    - "f*0" : [0, 0, 1, 1, 0, 0, 0, 0]
	#    - "f*1" : [0, 0, 1, 1, 0, 0, 0, 0]
	#  - Array K :
	#    - "f*0" : [80, 80, 80, 80, 80, 80, 80, 80]
	#    - "f*1" : [120, 120, 120, 120, 120, 120, 120, 120]
	#* Loop Level 6:
	#  - Array O :
	#    - "" : [7, 7, 6, 6, 6, 6, 6, 6]
	#  - Array I :
	#    - "" : [0, 0, 1, 1, 0, 0, 0, 0]
	#  - Array K :
	#    - "" : [200, 200, 200, 200, 200, 200, 200, 200]
	#* Loop Level 7:
	#  - Array O :
	#    - "" : [7, 7, 6, 6, 6, 6, 6, 6]
	#  - Array I :
	#    - "" : [0, 0, 1, 1, 1, 1, 0, 0]
	#  - Array K :
	#    - "" : [400, 400, 400, 400, 400, 400, 400, 400]

	return


# Tests of the "compute_cacheset_aware_comm_vol" function
def test_compute_cacheset_aware_comm_vol_1_Max1Reuse():
	# First test - no lambda
	str_scheme = "[V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(H,3); T(C,8); T(Y,34); T(W,3); T(C,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses==[75069024,71294484])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	# - Cache L1 : 75069024
	# - Cache L2 : 71294484
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 75059232
	#- Cache L2 : 70511940

	# Reference points:
	# - [Data from Dinero] din_L1 : 63 925 112 | din_L2 : 44 215 402
	# - [Measured CM] CM_L1 : 65 540 279 | CM_L2 : 34 570 219

	return

def test_compute_cacheset_aware_comm_vol_1_AllReuse():
	# First test - no lambda
	str_scheme = "[V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(H,3); T(C,8); T(Y,34); T(W,3); T(C,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses==[75059232,70511940])

	return

def test_compute_cacheset_aware_comm_vol_2_Max1Reuse():
	# Second test - lambda
	str_scheme = "[V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(H,3); T(X,4); T(Y,2); T(W,3); T(Y,4); TL(Y,[2,1]); Seq(Y)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses == [8328384,1531296])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	# - Cache L1 : 8328384
	# - Cache L2 : 1531296
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 8284032
	#- Cache L2 : 1531296

	# Reference points:
	# - [Data from Dinero] din_L1 : 6 720 833 | din_L2 : 1 364 046
	# - [Measured CM] CM_L1 : 7 845 484 | CM_L2 : 1 902 496

	return

def test_compute_cacheset_aware_comm_vol_2_AllReuse():
	# Second test - lambda
	str_scheme = "[V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(H,3); T(X,4); T(Y,2); T(W,3); T(Y,4); TL(Y,[2,1]); Seq(Y)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y)]"
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses == [8284032,1531296])

	return

def test_compute_cacheset_aware_comm_vol_3_Max1Reuse():
	# Third test - debug of a case where I had a factor 26 between Dinero and model
	# Old Data point:
	#{'name_conv': 'Yolo9000_02', 'perf': 59.01, 'din_l1cache': 3740266, 'din_l2cache': 2220933,
	#  'dinfa_l1cache': 2227392, 'dinfa_l2cache': 2227392, 'l1fassoc': 2227392, 'l2fassoc': 2227392,
	#  'l1camodel': 100871664, 'l2camodel': 3481536, 'l1cm': 5481357, 'l2cm': 5007044,
	# 'str_scheme': '[(V F); (U (1, F)); (U (8, X)); (U (2, Y)); (U (3, W)); (T (16, C));  (Hoist_vars [C]);
	#		(T (2, Y)); (T (4, F)); (T (34, X)); (T (17, Y));  (T (2, C)); (T (2, Y)); (T (3, H)); (T (2, Y))]'}
	
	str_scheme = "[V(F,16); U(F,1); U(X,8); U(Y,2); U(H,3); T(C,16); T(Y,2); T(F,4); T(X,34); T(Y,17); T(C,2); T(Y,2); T(W,3); T(Y,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,1); U(X,8); U(Y,2); U(W,3); T(C,16); T(Y,2); T(F,4); T(X,34); T(Y,17); T(C,2); T(Y,2); T(H,3); T(Y,2)]"
	
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses==[100871664,3481536])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 100871664
	#- Cache L2 : 3481536
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 100427760
	#- Cache L2 : 3481536

	# Reference points:
	# - [Data from Dinero] din_L1 : 3 740 266 | din_L2 : 2 220 933
	# - [Measured CM] CM_L1 : 5 481 357 | CM_L2 : 5 007 044
	#
	# Check of the repartition in our algo:
	# ==> L1: Number of comm on cache set #0 is 1000 times the comms of all others cache sets

	return

def test_compute_cacheset_aware_comm_vol_3_AllReuse():
	str_scheme = "[V(F,16); U(F,1); U(X,8); U(Y,2); U(H,3); T(C,16); T(Y,2); T(F,4); T(X,34); T(Y,17); T(C,2); T(Y,2); T(W,3); T(Y,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,1); U(X,8); U(Y,2); U(W,3); T(C,16); T(Y,2); T(F,4); T(X,34); T(Y,17); T(C,2); T(Y,2); T(H,3); T(Y,2)]"
	
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses==[100427760,3481536])

	return

def test_compute_cacheset_aware_comm_vol_4_Max1Reuse():
	# Forth test - case where it goes perfectly compared to Dinero
	# Old Data point:
	#{'name_conv': 'Yolo9000_02', 'perf': 45.79, 'din_l1cache': 5333816, 'din_l2cache': 5332248,
	# 'dinfa_l1cache': 5333760, 'dinfa_l2cache': 5333760, 'l1fassoc': 5333760, 'l2fassoc': 5333760,
	# 'l1camodel': 5333760, 'l2camodel': 6043680, 'l1cm': 5674411, 'l2cm': 9056055,
	# 'str_scheme': '[(V F); (U (2, F)); (ULambda Y); (T (32, C)); (Hoist_vars [C]); (T (272, X));
	#	(T (4, Y)); (T (3, W));  (Lambda_apply (Y, [((Iter 5), (Arg 11)); ((Iter 1), (Arg 13))])); (T (3, H)); (T (2, F))]'}	
	#
	str_scheme = "[V(F,16); U(F,2); UL(Y,[11,13]); T(C,32); T(X,272); T(Y,4); T(H,3); TL(Y,[5,1]); Seq(Y); T(W,3); T(F,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); UL(Y,[11,13]); T(C,32); T(X,272); T(Y,4); T(W,3); TL(Y,[5,1]); Seq(Y); T(H,3); T(F,2)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses==[5333760,6043680])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 5333760
	#- Cache L2 : 6043680
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 5333760
	#- Cache L2 : 6043680
	# (SAME)

	# Selected because Dinero =(almost)= Model (to check what happens)
	# => L1: Number of comm on cache set #0 is much more equilibrated across cache sets

	return

def test_compute_cacheset_aware_comm_vol_4_AllReuse():
	str_scheme = "[V(F,16); U(F,2); UL(Y,[11,13]); T(C,32); T(X,272); T(Y,4); T(H,3); TL(Y,[5,1]); Seq(Y); T(W,3); T(F,2)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); UL(Y,[11,13]); T(C,32); T(X,272); T(Y,4); T(W,3); TL(Y,[5,1]); Seq(Y); T(H,3); T(F,2)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses==[5333760, 6043680])

	return

def test_compute_cacheset_aware_comm_vol_5_Max1Reuse():
	# Fifth test - pre merge of all models in the old implem
	# Old Data point:
	# {'name_conv': 'Yolo9000_08', 'perf': 86.423, 'din_l1cache': 8786688, 'din_l2cache': 450584,
	#  'l1cm': 9418900, 'l2cm': 612512,
	#  'str_scheme': '[(V F); (U (2, F)); (U (2, X)); (ULambda Y); (U (3, W)); (T (128, C));  (Hoist_vars [C]);
	#         (T (17, X)); (T (4, F)); (T (2, F)); (T (4, Y));  (T (3, H)); (T (2, X));
	#         (Lambda_apply (Y,     [((Iter 1), (Arg 5));       ((Iter 2), (Arg 6))]     ))  ]',
	#	'din_fa_l1cache': 9517824, 'din_fa_l2cache': 449936, 'l1fassoc': 9517824, 'l2fassoc': 450048}
	#
	str_scheme = "[V(F,16); U(F,2); U(X,2); UL(Y,[5,6]); U(H,3); T(C,128); T(X,17); T(F,4); T(F,2); T(Y,4); T(W,3); T(X,2); TL(Y,[1,2]); Seq(Y)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); U(X,2); UL(Y,[5,6]); U(W,3); T(C,128); T(X,17); T(F,4); T(F,2); T(Y,4); T(H,3); T(X,2); TL(Y,[1,2]); Seq(Y)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_08"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses==[9124608,450048])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 9124608
	#- Cache L2 : 450048
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 9124608
	#- Cache L2 : 450048
	# (SAME)

	return

def test_compute_cacheset_aware_comm_vol_5_AllReuse():
	str_scheme = "[V(F,16); U(F,2); U(X,2); UL(Y,[5,6]); U(H,3); T(C,128); T(X,17); T(F,4); T(F,2); T(Y,4); T(W,3); T(X,2); TL(Y,[1,2]); Seq(Y)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); U(X,2); UL(Y,[5,6]); U(W,3); T(C,128); T(X,17); T(F,4); T(F,2); T(Y,4); T(H,3); T(X,2); TL(Y,[1,2]); Seq(Y)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_08"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses==[9124608,450048])

	return

def test_compute_cacheset_aware_comm_vol_6_Max1Reuse():
	# Sixth test - pre merge of all models in the old implem
	# Old Data point:
	# {'name_conv': 'Yolo9000_02', 'perf': 23.43, 'din_l1cache': 5752230, 'din_l2cache': 1357743,
	#	'dinfa_l1cache': 5405184, 'dinfa_l2cache': 1336128, 'l1fassoc': 5640192, 'l2fassoc': 1336128,
	#	'l1camodel': 5716080, 'l2camodel': 1336128, 'l1cm': 5853425, 'l2cm': 1566415,
	# 'str_scheme': '[(V F); (U (2, F)); (ULambda Y); (T (2, C)); (Hoist_vars [C]); (T (2, X));  (T (8, C)); (T (2, C));
	#     (Lambda_apply (Y,     [((Iter 9), (Arg 9));       ((Iter 5), (Arg 11))]     ));  (T (4, X));
	#     (T (2, F)); (T (3, H)); (T (2, X)); (T (2, Y)); (T (17, X));  (T (3, W))]'}
	str_scheme = "[V(F,16); U(F,2); UL(Y,[9,11]); T(C,2); T(X,2); T(C,8); T(C,2); TL(Y,[9,5]); Seq(Y); T(X,4); T(F,2); T(W,3); T(X,2); T(Y,2); T(X,17); T(H,3)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); UL(Y,[9,11]); T(C,2); T(X,2); T(C,8); T(C,2); TL(Y,[9,5]); Seq(Y); T(X,4); T(F,2); T(H,3); T(X,2); T(Y,2); T(X,17); T(W,3)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses == [6019632, 1336128])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 6019632
	#- Cache L2 : 1336128

	# Note: This is a match on the L2, but the L1 differ  (old result: 5716080 )
	# The dfp computation is identical on both sides, so it is coming from the fully assocative model
	#   The saturation happens at the level right below the Seq, so it might be a slight change in algo here
	#   (the new full assoc model algorithm is more general)

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 5471280
	#- Cache L2 : 1336128

	return

def test_compute_cacheset_aware_comm_vol_6_AllReuse():
	str_scheme = "[V(F,16); U(F,2); UL(Y,[9,11]); T(C,2); T(X,2); T(C,8); T(C,2); TL(Y,[9,5]); Seq(Y); T(X,4); T(F,2); T(W,3); T(X,2); T(Y,2); T(X,17); T(H,3)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation
	#str_scheme = "[V(F,16); U(F,2); UL(Y,[9,11]); T(C,2); T(X,2); T(C,8); T(C,2); TL(Y,[9,5]); Seq(Y); T(X,4); T(F,2); T(H,3); T(X,2); T(Y,2); T(X,17); T(W,3)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_02"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses == [5471280, 1336128])

	return

def test_compute_cacheset_aware_comm_vol_7_Max1Reuse():
	# Seventh test - pre merge of all models in the old implem
	# Old Data point:
	# {'name_conv': 'Yolo9000_19', 'perf': 37.88, 'din_l1cache': 1731281, 'din_l2cache': 1715918,
	#	'dinfa_l1cache': 1828384, 'dinfa_l2cache': 1698912, 'l1fassoc': 1828384, 'l2fassoc': 1698912,
	#	'l1camodel': 6738120, 'l2camodel': 4233408, 'l1cm': 1768651, 'l2cm': 1722812,
	#	'str_scheme': '[(V F); (U (4, F)); (ULambda Y); (T (1024, C)); (Hoist_vars [C]); (T (2, F));  (T (2, F));
	#       (T (2, F));  (Lambda_apply (Y,     [((Iter 1), (Arg 5));       ((Iter 2), (Arg 6))]     ));  (T (17, X))]'}

	str_scheme = "[V(F,16); U(F,4); UL(Y,[5,6]); T(C,64); T(C,16); T(F,2); T(F,2); T(F,2); TL(Y,[1,2]); Seq(Y); T(X,17)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation. But H=W=1 : no problem
	#str_scheme = "[V(F,16); U(F,4); UL(Y,[5,6]); T(C,64); T(C,16); T(F,2); T(F,2); T(F,2); TL(Y,[1,2]); Seq(Y); T(X,17)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_19"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses == [6847328,4233408])

	### reuse_strat = MAX1 (corresponding to the old implem)
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 6847328
	#- Cache L2 : 4233408
	# ====> MATCH !

	### reuse_strat = UNLIMITED_REUSE
	#lcachemisses (in number of cache line) =
	#- Cache L1 : 6840800
	#- Cache L2 : 4231096

	return

def test_compute_cacheset_aware_comm_vol_7_AllReuse():
	str_scheme = "[V(F,16); U(F,4); UL(Y,[5,6]); T(C,64); T(C,16); T(F,2); T(F,2); T(F,2); TL(Y,[1,2]); Seq(Y); T(X,17)]"
	# Note: careful, there is a permutation on the H/W compared to the old implementation. But H=W=1 : no problem
	#str_scheme = "[V(F,16); U(F,4); UL(Y,[5,6]); T(C,64); T(C,16); T(F,2); T(F,2); T(F,2); TL(Y,[1,2]); Seq(Y); T(X,17)]"

	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)

	comp = Computation(Computation_spec.CONV, 4)
	d_prob_sizes = ddsizes_Yolo["Yolo9000_19"]
	d_prob_sizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(d_prob_sizes)
	#print(f"{d_prob_sizes=}")
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True)

	assert(lcachemisses == [6840800,4231096])

	return

def test_compute_cacheset_aware_comm_vol_8():
	# Test with matmult
	str_scheme = "[V(J,16); U(J,2); U(I,8); T(K,512); T(K,4); T(I,32); T(J,16); T(J,16)]"
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4)
	d_prob_sizes = ddsizes_matmul["Gemma2B_QKV"]
	lcont_arr_order = get_list_array_contiguous_alloc(comp)
	
	# Pinocchio L1 and L2 cache sizes
	lcachesizes = [8192, 262144]  # In float (= 4 octets)
	lassoc_cache = [8, 16]
	lnum_cache_set = [64, 1024]
	cache_line_size = 16
	
	# Let's go!
	lcachemisses = compute_cacheset_aware_comm_vol(scheme, comp, d_prob_sizes, lcont_arr_order,
		lcachesizes, lassoc_cache, lnum_cache_set, cache_line_size,
		b_sanity_check=True) #, reuse_strat_full_assoc=ReuseLoopStrat.MAX1_LOOP_REUSE)

	assert(lcachemisses == [444334080, 43155456])

	# Note: lots of power of 2 in the problem size, lots of imprecision from the model since
	#	everything goes to a few cache sets in the first iteration.

	return

