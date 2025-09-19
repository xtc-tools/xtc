
from xtc.schedules.ttile.scheme import build_scheme_from_str
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.computation import get_array_accesses, get_ldims_computation
from xtc.schedules.ttile.scheme_to_xtc import subst_dimname_xyhw_to_hwrs_conv2D_scheme

from xtc.schedules.ttile.cache_model.full_assoc_model import compute_footprint_for_each_level, ReuseLoopStrat
from xtc.schedules.ttile.cache_model.full_assoc_model import find_saturation_level, compute_full_assoc_cache_misses


# Tests of "compute_footprint_for_each_level"
def test_compute_footprint_for_each_level_1():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}  # Technically do not need all the dims, just the strides
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...


	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)

	#print_ldd_footprint(scheme, ldd_footprint)

	# Pretty-printer
	expected_res = [
		{'': {'O': 16, 'I': 16, 'K': 16, '_Total': 48}},
		{'': {'O': 32, 'I': 16, 'K': 32, '_Total': 80}},
		{'': {'O': 256, 'I': 128, 'K': 32, '_Total': 416}},
		{'': {'O': 256, 'I': 128, 'K': 64, '_Total': 448}},
		{'': {'O': 17408, 'I': 8704, 'K': 64, '_Total': 26176}},
		{'': {'O': 69632, 'I': 34816, 'K': 64, '_Total': 104512}},
		{'': {'O': 139264, 'I': 34816, 'K': 128, '_Total': 174208}},
		{'': {'O': 139264, 'I': 43520, 'K': 384, '_Total': 183168}},
		{'': {'O': 139264, 'I': 43520, 'K': 3072, '_Total': 185856}},
		{'': {'O': 4734976, 'I': 1192448, 'K': 3072, '_Total': 5930496}},
		{'': {'O': 4734976, 'I': 1201216, 'K': 9216, '_Total': 5945408}},
		{'': {'O': 4734976, 'I': 2402432, 'K': 18432, '_Total': 7155840}}]

	assert(ldd_footprint == expected_res)

	""" Results:
	- Loop lvl 1 - Last scheme atom: V(f,16):
	    LBranch "" -> {'O': 16, 'I': 16, 'K': 16, '_Total': 48}
	- Loop lvl 2 - Last scheme atom: U(f,2):
	    LBranch "" -> {'O': 32, 'I': 16, 'K': 32, '_Total': 80}
	- Loop lvl 3 - Last scheme atom: U(w,8):
	    LBranch "" -> {'O': 256, 'I': 128, 'K': 32, '_Total': 416}
	- Loop lvl 4 - Last scheme atom: T(c,2):
	    LBranch "" -> {'O': 256, 'I': 128, 'K': 64, '_Total': 448}
	- Loop lvl 5 - Last scheme atom: T(h,68):
	    LBranch "" -> {'O': 17408, 'I': 8704, 'K': 64, '_Total': 26176}
	- Loop lvl 6 - Last scheme atom: T(h,4):
	    LBranch "" -> {'O': 69632, 'I': 34816, 'K': 64, '_Total': 104512}
	- Loop lvl 7 - Last scheme atom: T(f,2):
	    LBranch "" -> {'O': 139264, 'I': 34816, 'K': 128, '_Total': 174208}
	- Loop lvl 8 - Last scheme atom: T(s,3):
	    LBranch "" -> {'O': 139264, 'I': 35072, 'K': 384, '_Total': 174720}
	- Loop lvl 9 - Last scheme atom: T(c,8):
	    LBranch "" -> {'O': 139264, 'I': 35072, 'K': 3072, '_Total': 177408}
	- Loop lvl 10 - Last scheme atom: T(w,34):
	    LBranch "" -> {'O': 4734976, 'I': 1192448, 'K': 3072, '_Total': 5930496}
	- Loop lvl 11 - Last scheme atom: T(r,3):
	    LBranch "" -> {'O': 4734976, 'I': 1201216, 'K': 9216, '_Total': 5945408}
	- Loop lvl 12 - Last scheme atom: T(c,2):
	    LBranch "" -> {'O': 4734976, 'I': 2402432, 'K': 18432, '_Total': 7155840}
	"""
	return

def test_compute_footprint_for_each_level_2():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}  # Technically do not need all the dims, just the strides
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)

	# Pretty-printer
	#print_ldd_footprint(scheme, ldd_footprint)

	expected_res = [{'': {'O': 16, 'I': 16, 'K': 16, '_Total': 48}},
		{'': {'O': 64, 'I': 16, 'K': 64, '_Total': 144}},
		{'w*0': {'O': 320, 'I': 80, 'K': 64, '_Total': 464}, 'w*1': {'O': 448, 'I': 112, 'K': 64, '_Total': 624}},
		{'w*0': {'O': 320, 'I': 80, 'K': 128, '_Total': 528}, 'w*1': {'O': 448, 'I': 112, 'K': 128, '_Total': 688}},
		{'w*0': {'O': 320, 'I': 80, 'K': 1024, '_Total': 1424}, 'w*1': {'O': 448, 'I': 112, 'K': 1024, '_Total': 1584}},
		{'w*0': {'O': 10880, 'I': 2720, 'K': 1024, '_Total': 14624}, 'w*1': {'O': 15232, 'I': 3808, 'K': 1024, '_Total': 20064}},
		{'w*0': {'O': 21760, 'I': 5440, 'K': 1024, '_Total': 28224}, 'w*1': {'O': 30464, 'I': 7616, 'K': 1024, '_Total': 39104}},
		{'w*0': {'O': 43520, 'I': 10880, 'K': 1024, '_Total': 55424}, 'w*1': {'O': 60928, 'I': 15232, 'K': 1024, '_Total': 77184}},
		{'w*0': {'O': 43520, 'I': 21760, 'K': 2048, '_Total': 67328}, 'w*1': {'O': 60928, 'I': 30464, 'K': 2048, '_Total': 93440}},
		{'w*0': {'O': 43520, 'I': 26112, 'K': 6144, '_Total': 75776}, 'w*1': {'O': 60928, 'I': 34816, 'K': 6144, '_Total': 101888}},
		{'w*0': {'O': 174080, 'I': 104448, 'K': 6144, '_Total': 284672}, 'w*1': {'O': 243712, 'I': 139264, 'K': 6144, '_Total': 389120}},
		{'w*0': {'O': 348160, 'I': 191488, 'K': 6144, '_Total': 545792}, 'w*1': {'O': 487424, 'I': 261120, 'K': 6144, '_Total': 754688}},
		{'w*0': {'O': 348160, 'I': 192896, 'K': 18432, '_Total': 559488}, 'w*1': {'O': 487424, 'I': 263040, 'K': 18432, '_Total': 768896}},
		{'w*0': {'O': 1392640, 'I': 718976, 'K': 18432, '_Total': 2130048}, 'w*1': {'O': 1949696, 'I': 999552, 'K': 18432, '_Total': 2967680}},
		{'w*0': {'O': 2785280, 'I': 1420416, 'K': 18432, '_Total': 4224128}, 'w*1': {'O': 1949696, 'I': 999552, 'K': 18432, '_Total': 2967680}},
		{'': {'O': 4734976, 'I': 2402432, 'K': 18432, '_Total': 7155840}}]

	assert(ldd_footprint == expected_res)

	""" Results:
	- Loop lvl 1 - Last scheme atom: V(f,16):
	    LBranch "" -> {'O': 16, 'I': 16, 'K': 16, '_Total': 48}
	- Loop lvl 2 - Last scheme atom: U(f,4):
	    LBranch "" -> {'O': 64, 'I': 16, 'K': 64, '_Total': 144}
	- Loop lvl 3 - Last scheme atom: UL(w, [5, 7]):
	    LBranch "w*0" -> {'O': 320, 'I': 80, 'K': 64, '_Total': 464}
	    LBranch "w*1" -> {'O': 448, 'I': 112, 'K': 64, '_Total': 624}
	- Loop lvl 4 - Last scheme atom: U(c,2):
	    LBranch "w*0" -> {'O': 320, 'I': 80, 'K': 128, '_Total': 528}
	    LBranch "w*1" -> {'O': 448, 'I': 112, 'K': 128, '_Total': 688}
	- Loop lvl 5 - Last scheme atom: T(c,8):
	    LBranch "w*0" -> {'O': 320, 'I': 80, 'K': 1024, '_Total': 1424}
	    LBranch "w*1" -> {'O': 448, 'I': 112, 'K': 1024, '_Total': 1584}
	- Loop lvl 6 - Last scheme atom: T(h,34):
	    LBranch "w*0" -> {'O': 10880, 'I': 2720, 'K': 1024, '_Total': 14624}
	    LBranch "w*1" -> {'O': 15232, 'I': 3808, 'K': 1024, '_Total': 20064}
	- Loop lvl 7 - Last scheme atom: T(h,2):
	    LBranch "w*0" -> {'O': 21760, 'I': 5440, 'K': 1024, '_Total': 28224}
	    LBranch "w*1" -> {'O': 30464, 'I': 7616, 'K': 1024, '_Total': 39104}
	- Loop lvl 8 - Last scheme atom: T(w,2):
	    LBranch "w*0" -> {'O': 43520, 'I': 10880, 'K': 1024, '_Total': 55424}
	    LBranch "w*1" -> {'O': 60928, 'I': 15232, 'K': 1024, '_Total': 77184}
	- Loop lvl 9 - Last scheme atom: T(c,2):
	    LBranch "w*0" -> {'O': 43520, 'I': 21760, 'K': 2048, '_Total': 67328}
	    LBranch "w*1" -> {'O': 60928, 'I': 30464, 'K': 2048, '_Total': 93440}
	- Loop lvl 10 - Last scheme atom: T(s,3):
	    LBranch "w*0" -> {'O': 43520, 'I': 22400, 'K': 6144, '_Total': 72064}
	    LBranch "w*1" -> {'O': 60928, 'I': 31360, 'K': 6144, '_Total': 98432}
	- Loop lvl 11 - Last scheme atom: T(h,4):
	    LBranch "w*0" -> {'O': 174080, 'I': 87680, 'K': 6144, '_Total': 267904}
	    LBranch "w*1" -> {'O': 243712, 'I': 122752, 'K': 6144, '_Total': 372608}
	- Loop lvl 12 - Last scheme atom: T(w,2):
	    LBranch "w*0" -> {'O': 348160, 'I': 175360, 'K': 6144, '_Total': 529664}
	    LBranch "w*1" -> {'O': 487424, 'I': 245504, 'K': 6144, '_Total': 739072}
	- Loop lvl 13 - Last scheme atom: T(r,3):
	    LBranch "w*0" -> {'O': 348160, 'I': 192896, 'K': 18432, '_Total': 559488}
	    LBranch "w*1" -> {'O': 487424, 'I': 263040, 'K': 18432, '_Total': 768896}
	- Loop lvl 14 - Last scheme atom: T(w,4):
	    LBranch "w*0" -> {'O': 1392640, 'I': 718976, 'K': 18432, '_Total': 2130048}
	    LBranch "w*1" -> {'O': 1949696, 'I': 999552, 'K': 18432, '_Total': 2967680}
	- Loop lvl 15 - Last scheme atom: TL(w, [2, 1]):
	    LBranch "w*0" -> {'O': 2785280, 'I': 1420416, 'K': 18432, '_Total': 4224128}
	    LBranch "w*1" -> {'O': 1949696, 'I': 999552, 'K': 18432, '_Total': 2967680}
	- Loop lvl 16 - Last scheme atom: Seq(w):
	    LBranch "" -> {'O': 4734976, 'I': 2402432, 'K': 18432, '_Total': 7155840}
	"""
	return


# Tests of "find_saturation_level"
def test_saturation_level_1_L1():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 8192  # L1 cache size

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	assert(d_sat_loop_lvl == {'': 4})

	return

def test_saturation_level_1_L2():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 262144 # L2 cache size

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	assert(d_sat_loop_lvl == {'': 9})

	return

def test_saturation_level_2_L1():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 8192

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	assert(d_sat_loop_lvl == {'w*0': 5, 'w*1': 5})

	return

def test_saturation_level_2_L2():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 262144 # L2 cache size

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	assert(d_sat_loop_lvl == {'w*0': 10, 'w*1': 10})

	return

# More tricky cases
def test_saturation_level_3():
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); TL(K,[1024,16]); Seq(I); Seq(K) ]"
	d_full_sizes = { }
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 4800

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	#print_ldd_footprint(scheme, ldd_footprint)
	assert(d_sat_loop_lvl == {'i*0,k*0': 2, 'i*0,k*1': 2, 'i*1,k*0': 3, 'i*1,k*1': 4})

	return

def test_saturation_level_4():
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); Seq(I); TL(K,[1024,16]); Seq(K) ]"
	d_full_sizes = { }
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	cachesize = 4800

	ldd_footprint = compute_footprint_for_each_level(scheme, comp, d_full_sizes, d_arrays_accs, prog_dims, cache_line_size)
	d_sat_loop_lvl = find_saturation_level(ldd_footprint, cachesize, comp)

	#print_ldd_footprint(scheme, ldd_footprint)
	assert(d_sat_loop_lvl == {'i*0,k*0': 2, 'i*0,k*1': 2, 'i*1,k*0': 3, 'i*1,k*1': 3})

	return


# Tests of "compute_full_assoc_cache_misses"
def test_full_assoc_model_1_NoReuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 68
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 68
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 68
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [68])

	# === Detailed calculations ===
	# ldd_footprint (in elements):
	#- Loop lvl 1 - Last scheme atom: V(j,16):
	#    LBranch "" -> {'A': 16, 'B': 16, 'C': 16, '_Total': 48}
	#- Loop lvl 2 - Last scheme atom: T(j,2):
	#    LBranch "" -> {'A': 16, 'B': 32, 'C': 32, '_Total': 80}
	#- Loop lvl 3 - Last scheme atom: T(k,4):
	#    LBranch "" -> {'A': 16, 'B': 128, 'C': 32, '_Total': 176}
	#- Loop lvl 4 - Last scheme atom: T(i,3):
	#    LBranch "" -> {'A': 48, 'B': 128, 'C': 96, '_Total': 272}
	#- Loop lvl 5 - Last scheme atom: T(k,4):
	#    LBranch "" -> {'A': 48, 'B': 512, 'C': 96, '_Total': 656}
	#
	# Saturation lvl happens at lvl 4 (atom "T(i,3)" is the one that triggers it)
	# - Array A: 3 cache misses at saturation lvl, repeated 4 times by "T(k,4)" ==> Total for A: 12
	# - Array B: 8 cache misses at saturation lvl, repeated 4 times by "T(k,4)" ==> Total for B: 32
	# - Array C: 6 cache misses, repeated 4 times by "T(k,4)" ==> Total for C: 24
	# ==> Total cache misses across all arrays: 68

	# Dinero measurement:
	# 	Option "-l1-dsize 1024 -l1-dbsize 64 -l1-dassoc 16 -l1-drepl p" => 75
	#	Option "-l1-dsize 1024 -l1-dbsize 64 -l1-dassoc 16 -l1-drepl l") => 68 (MATCH!)

	return

def test_full_assoc_model_1_Max1Reuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 68
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 68
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 68
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [68])

	return

def test_full_assoc_model_1_AllReuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 68
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 68
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 68
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [68])

	return

def test_full_assoc_model_1B_NoReuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4); T(K,8) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 544
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 544
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 544
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [544])

	# === Detailed calculations ===
	# ldd_footprint (in elements):
	#- Loop lvl 1 - Last scheme atom: V(j,16):
	#    LBranch "" -> {'A': 1, 'B': 1, 'C': 1, '_Total': 3} [in CL]
	#- Loop lvl 2 - Last scheme atom: T(j,2):
	#    LBranch "" -> {'A': 1, 'B': 2, 'C': 2, '_Total': 5} [in CL]
	#- Loop lvl 3 - Last scheme atom: T(k,4):
	#    LBranch "" -> {'A': 1, 'B': 8, 'C': 2, '_Total': 11} [in CL]
	#- Loop lvl 4 - Last scheme atom: T(i,3):
	#    LBranch "" -> {'A': 3, 'B': 8, 'C': 6, '_Total': 17} [in CL]
	#- Loop lvl 5 - Last scheme atom: T(k,4):
	#    LBranch "" -> {'A': 3, 'B': 32, 'C': 6, '_Total': 41} [in CL]
	#- Loop lvl 6 - Last scheme atom: T(k,8):
	#    LBranch "" -> {'A': 24, 'B': 256, 'C': 6, '_Total': 286} [in CL]
	#
	# Saturation lvl happens at lvl 4 (atom "T(i,3)" is the one that triggers it)
	# - Array A: 3 cache misses at saturation lvl, repeated 32 times ==> Total for A: 96
	# - Array B: 32 cache misses at saturation lvl, repeated 8 times ==> Total for B: 256
	# - Array C: 6 cache misses at saturation lvl, repeated 32 times ==> Total for C: 192
	# ==> Total cache misses across all arrays: 544

	# Dinero measurement:
	#	Option "-l1-dsize 1024 -l1-dbsize 64 -l1-dassoc 16 -l1-drepl l") => 544 (MATCH!)

	return

def test_full_assoc_model_1B_Max1Reuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4); T(K,8) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 544
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 544
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 544
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [544])

	return

def test_full_assoc_model_1B_AllReuse():
	# This is the example of the Sarcasm article
	str_scheme = "[ V(J,16); T(J,2); T(K,4); T(I,3); T(K,4); T(K,8) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]
	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 544
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 544
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 544
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [544])

	return

def test_full_assoc_model_2_NoReuse():
	# More complicated test with a simple lambda behavior (saturation below a lambda)
	str_scheme = "[ V(J,16); TL(J,[2,4]); TL(K,[4,8]); T(I,3); Seq(J); T(K,4); Seq(K) ]"
	d_full_sizes = { }
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 992
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 992
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 992
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [992])

	# Dinero measurement:
	#	Option "-l1-dsize 1024 -l1-dbsize 64 -l1-dassoc 16 -l1-drepl l") => 992 (MATCH!)

	# === Detailed calculations ===
	# ldd_footprint (in elements):
	#- Loop lvl 1 - Last scheme atom: V(j,16):
	#    LBranch "" -> {'A': 1, 'B': 1, 'C': 1, '_Total': 3} [in CL]
	#- Loop lvl 2 - Last scheme atom: TL(j, [2, 4]):
	#    LBranch "j*0" -> {'A': 1, 'B': 2, 'C': 2, '_Total': 5} [in CL]
	#    LBranch "j*1" -> {'A': 1, 'B': 4, 'C': 4, '_Total': 9} [in CL]
	#- Loop lvl 3 - Last scheme atom: TL(k, [4, 8]):
	#    LBranch "j*0,k*0" -> {'A': 1, 'B': 8, 'C': 2, '_Total': 11} [in CL]
	#    LBranch "j*1,k*0" -> {'A': 1, 'B': 16, 'C': 4, '_Total': 21} [in CL]
	#    LBranch "j*0,k*1" -> {'A': 1, 'B': 16, 'C': 2, '_Total': 19} [in CL]
	#    LBranch "j*1,k*1" -> {'A': 1, 'B': 32, 'C': 4, '_Total': 37} [in CL]
	#- Loop lvl 4 - Last scheme atom: T(i,3):
	#    LBranch "j*0,k*0" -> {'A': 3, 'B': 8, 'C': 6, '_Total': 17} [in CL]
	#    LBranch "j*1,k*0" -> {'A': 3, 'B': 16, 'C': 12, '_Total': 31} [in CL]
	#    LBranch "j*0,k*1" -> {'A': 3, 'B': 16, 'C': 6, '_Total': 25} [in CL]
	#    LBranch "j*1,k*1" -> {'A': 3, 'B': 32, 'C': 12, '_Total': 47} [in CL]
	#- Loop lvl 5 - Last scheme atom: Seq(j):
	#    LBranch "k*0" -> {'A': 3, 'B': 24, 'C': 18, '_Total': 45} [in CL]
	#    LBranch "k*1" -> {'A': 3, 'B': 48, 'C': 18, '_Total': 69} [in CL]
	#- Loop lvl 6 - Last scheme atom: T(k,4):
	#    LBranch "k*0" -> {'A': 3, 'B': 96, 'C': 18, '_Total': 117} [in CL]
	#    LBranch "k*1" -> {'A': 6, 'B': 192, 'C': 18, '_Total': 216} [in CL]
	#- Loop lvl 7 - Last scheme atom: Seq(k):
	#    LBranch "" -> {'A': 9, 'B': 288, 'C': 18, '_Total': 315} [in CL]
	#
	# Saturation levels:
	#   {'j*0,k*0': 4, 'j*0,k*1': 3, 'j*1,k*0': 3, 'j*1,k*1': 3}
	#
	# Summations:
	#  For array A:
	#    - For lambda location j*0,k*0:
	#      3 [FP_CL] * 4 [REP] = 12 [CM]
	#    - For lambda location j*0,k*1:
	#      1 [FP_CL] * 12 [REP] = 12 [CM]
	#    - For lambda location j*1,k*0:
	#      1 [FP_CL] * 12 [REP] = 12 [CM]
	#    - For lambda location j*1,k*1:
	#      1 [FP_CL] * 12 [REP] = 12 [CM]
	#  => Total for array A = 48
	#  For array B:
	#    - For lambda location j*0,k*0:
	#      8 [FP_CL] * 4 [REP] = 32 [CM]
	#    - For lambda location j*0,k*1:
	#      16 [FP_CL] * 12 [REP] = 192 [CM]
	#    - For lambda location j*1,k*0:
	#      16 [FP_CL] * 12 [REP] = 192 [CM]
	#    - For lambda location j*1,k*1:
	#      32 [FP_CL] * 12 [REP] = 384 [CM]
	#  => Total for array B = 288
	#  For array C:
	#    - For lambda location j*0,k*0:
	#      6 [FP_CL] * 4 [REP] = 24 [CM]
	#    - For lambda location j*0,k*1:
	#      2 [FP_CL] * 12 [REP] = 24 [CM]
	#    - For lambda location j*1,k*0:
	#      4 [FP_CL] * 12 [REP] = 48 [CM]
	#    - For lambda location j*1,k*1:
	#      4 [FP_CL] * 12 [REP] = 48 [CM]
	#  => Total for array C = 144
	#=> Total across all arrays = 992

	return

def test_full_assoc_model_2_Max1Reuse():
	# More complicated test with a simple lambda behavior (saturation below a lambda)
	str_scheme = "[ V(J,16); TL(J,[2,4]); TL(K,[4,8]); T(I,3); Seq(J); T(K,4); Seq(K) ]"
	d_full_sizes = { }
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 992
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 992
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 992
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [992])

	return

def test_full_assoc_model_2_AllReuse():
	# More complicated test with a simple lambda behavior (saturation below a lambda)
	str_scheme = "[ V(J,16); TL(J,[2,4]); TL(K,[4,8]); T(I,3); Seq(J); T(K,4); Seq(K) ]"
	d_full_sizes = { }
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 16*cache_line_size ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 992
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 992
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 992
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [992])

	return

def test_full_assoc_model_3_NoReuse():
	# More complicated test with tricky lambda behavior (saturation on a Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); TL(K,[1024,16]); Seq(I); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 316574
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 316574
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 316574
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[316574])

	# Dinero measurement:
	#	Option "-l1-dsize 19200 -l1-dbsize 64 -l1-dassoc 300 -l1-drepl l" => 316572

	# === Detailed calculations ===
	# ldd_footprint (in elements):
	#- Loop lvl 1 - Last scheme atom: V(j,16):
	#    LBranch "" -> {'A': 16, 'B': 16, 'C': 16, '_Total': 48}
	#- Loop lvl 2 - Last scheme atom: U(j,2):
	#    LBranch "" -> {'A': 16, 'B': 32, 'C': 32, '_Total': 80}
	#- Loop lvl 3 - Last scheme atom: UL(i, [100, 6]):
	#    LBranch "i*0" -> {'A': 1600, 'B': 32, 'C': 3200, '_Total': 4832}
	#    LBranch "i*1" -> {'A': 96, 'B': 32, 'C': 192, '_Total': 320}
	#- Loop lvl 4 - Last scheme atom: TL(k, [1024, 16]):
	#    LBranch "i*0,k*0" -> {'A': 102400, 'B': 32768, 'C': 3200, '_Total': 138368}
	#    LBranch "i*1,k*0" -> {'A': 6144, 'B': 32768, 'C': 192, '_Total': 39104}
	#    LBranch "i*0,k*1" -> {'A': 1600, 'B': 512, 'C': 3200, '_Total': 5312}
	#    LBranch "i*1,k*1" -> {'A': 96, 'B': 512, 'C': 192, '_Total': 800}
	#- Loop lvl 5 - Last scheme atom: Seq(i):
	#    LBranch "k*0" -> {'A': 108544, 'B': 32768, 'C': 3392, '_Total': 144704}
	#    LBranch "k*1" -> {'A': 1696, 'B': 512, 'C': 3392, '_Total': 5600}
	#- Loop lvl 6 - Last scheme atom: Seq(k):
	#    LBranch "" -> {'A': 110240, 'B': 33280, 'C': 3392, '_Total': 146912}
	#
	# Saturation levels:
	#   {'i*0,k*0': 3, 'i*0,k*1': 3, 'i*1,k*0': 4, 'i*1,k*1': 5}
	# After update due to special cases (a saturation level of a branch happening at a Seq:
	#   {'i*0,k*0': 3, 'i*0,k*1': 3, 'i*1,k*0': 4, 'i*1,k*1': 4}
	#
	# Summations:
	#  For array A:
	#    - For lambda location i*0,k*0:
	#      100 [FP_CL] * 1024 [REP] = 102400 [CM]
	#    - For lambda location i*0,k*1:
	#      100 [FP_CL] * 16 [REP] = 1600 [CM]
	#    - For lambda location i*1,k*0:
	#      384 [FP_CL] * 1 [REP] = 384 [CM]
	#    - For lambda location i*1,k*1:
	#      6 [FP_CL] * 1 [REP] = 6 [CM]
	#  => Total for array A = 104390
	#  For array B:
	#    - For lambda location i*0,k*0:
	#      2 [FP_CL] * 1024 [REP] = 2048 [CM]
	#    - For lambda location i*0,k*1:
	#      2 [FP_CL] * 16 [REP] = 32 [CM]
	#    - For lambda location i*1,k*0:
	#      2048 [FP_CL] * 1 [REP] = 2048 [CM]
	#    - For lambda location i*1,k*1:
	#      32 [FP_CL] * 1 [REP] = 32 [CM]
	#  => Total for array B = 4160
	#  For array C:
	#    - For lambda location i*0,k*0:
	#      200 [FP_CL] * 1024 [REP] = 204800 [CM]
	#    - For lambda location i*0,k*1:
	#      200 [FP_CL] * 16 [REP] = 3200 [CM]
	#    - For lambda location i*1,k*0:
	#      12 [FP_CL] * 1 [REP] = 12 [CM]
	#    - For lambda location i*1,k*1:
	#      12 [FP_CL] * 1 [REP] = 12 [CM]
	#  => Total for array C = 208024
	#=> Total cache misses = 316574

	return

def test_full_assoc_model_3_Max1Reuse():
	# More complicated test with tricky lambda behavior (saturation on a Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); TL(K,[1024,16]); Seq(I); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 316574
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 316574
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 316574
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[316574])

	return

def test_full_assoc_model_3_AllReuse():
	# More complicated test with tricky lambda behavior (saturation on a Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); TL(K,[1024,16]); Seq(I); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 316574
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 316574
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 316574
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[316574])

	return

def test_full_assoc_model_4_NoReuse():
	# More complicated test with tricky lambda behavior (interleaved Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); Seq(I); TL(K,[1024,16]); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 334880
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 334880
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 330728
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [334880])

	# Dinero measurement:
	#	Option "-l1-dsize 19200 -l1-dbsize 64 -l1-dassoc 300 -l1-drepl l" => 332800

	return

def test_full_assoc_model_4_Max1Reuse():
	# More complicated test with tricky lambda behavior (interleaved Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); Seq(I); TL(K,[1024,16]); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 334880
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 334880
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 330728
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [334880])

	return

def test_full_assoc_model_4_AllReuse():
	# More complicated test with tricky lambda behavior (interleaved Seq)
	str_scheme = "[ V(J,16); U(J,2); UL(I,[100,6]); Seq(I); TL(K,[1024,16]); Seq(K) ]"
	d_full_sizes = {}
	scheme = build_scheme_from_str(str_scheme)

	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 4800 ]

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = 334880
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = 334880
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = 330728
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [330728])

	return


# Real examples
def test_full_assoc_model_real_1_NoReuse():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [64078848, 2223936]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [63961344, 2223936] # Checked original implem
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [63961344, 2223936]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [64078848, 2223936])
	# Note: Dinero (set-assoc): [ 63925112 , 44215402 ]

	return

def test_full_assoc_model_real_1_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [64078848, 2223936]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [63961344, 2223936] # Checked original implem
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [63961344, 2223936]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [63961344, 2223936])

	return

def test_full_assoc_model_real_1_AllReuse():
	str_scheme = "[ V(F,16); U(F,2); U(Y,8); T(C,2); T(X,68); T(X,4); T(F,2); T(W,3); T(C,8); T(Y,34); T(H,3); T(C,2) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [64078848, 2223936]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [63961344, 2223936] # Checked original implem
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [63961344, 2223936]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [63961344, 2223936])

	return

def test_full_assoc_model_real_2_NoReuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [7100928, 1437696]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [6879744, 1423872]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [6769152, 1423872]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [7100928, 1437696])
	# Original implem: [ 6879744, 1348800 ]  (MAX1_LOOP_REUSE - Match when interversion between r/s (bug in old implem) )
	# Dinero (set-assoc): [ 6720833, 1364046 ]

	return

def test_full_assoc_model_real_2_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [7100928, 1437696]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [6879744, 1423872]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [6769152, 1423872]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [6879744, 1423872])

	return

def test_full_assoc_model_real_2_All1Reuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(W,3); T(X,4); T(Y,2); T(H,3); T(Y,4); TL(Y,[2,1]); Seq(Y) ]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [7100928, 1437696]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [6879744, 1423872]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [6769152, 1423872]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [6769152, 1423872])

	return

def test_full_assoc_model_real_3_NoReuse():
	str_scheme = "[ V(F,16); U(F,2); U(X,4); U(Y,2); U(H,3); T(C,16); T(W,3); T(Y,4); T(F,2); T(X,4); T(C,2); T(X,17); T(Y,34)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [3532736, 447240]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [3463376, 447240]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [3463376, 447240]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [3532736, 447240])
	# Original implem: [ 3463376, 447240 ]  (MAX1_LOOP_REUSE - Match!)

	return

def test_full_assoc_model_real_3_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,2); U(X,4); U(Y,2); U(H,3); T(C,16); T(W,3); T(Y,4); T(F,2); T(X,4); T(C,2); T(X,17); T(Y,34)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [3532736, 447240]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [3463376, 447240]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [3463376, 447240]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [3463376, 447240])

	return

def test_full_assoc_model_real_3_AllReuse():
	str_scheme = "[ V(F,16); U(F,2); U(X,4); U(Y,2); U(H,3); T(C,16); T(W,3); T(Y,4); T(F,2); T(X,4); T(C,2); T(X,17); T(Y,34)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [3532736, 447240]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [3463376, 447240]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [3463376, 447240]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [3463376, 447240])

	return

def test_full_assoc_model_real_4_NoReuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[7,8]); U(C,2); T(C,2); T(X,17); TL(Y,[4,5]); Seq(Y); T(X,8); T(H,3); T(C,4); T(C,2); T(Y,2); T(F,2); T(C,2); T(F,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [32108544, 31970304]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [32108544, 31970304]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [31967232, 31970304]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [32108544, 31970304])
	# Original implem: [ 32034816, 31970304 ]
	# TODO: check (minor) difference?

	return

def test_full_assoc_model_real_4_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[7,8]); U(C,2); T(C,2); T(X,17); TL(Y,[4,5]); Seq(Y); T(X,8); T(H,3); T(C,4); T(C,2); T(Y,2); T(F,2); T(C,2); T(F,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [32108544, 31970304]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [32108544, 31970304]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [31967232, 31970304]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [32108544, 31970304])

	return

def test_full_assoc_model_real_4_AllReuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[7,8]); U(C,2); T(C,2); T(X,17); TL(Y,[4,5]); Seq(Y); T(X,8); T(H,3); T(C,4); T(C,2); T(Y,2); T(F,2); T(C,2); T(F,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [32108544, 31970304]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [32108544, 31970304]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [31967232, 31970304]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [31967232, 31970304])

	return

def test_full_assoc_model_real_5_NoReuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[4,6]); U(C,2); T(C,64); T(Y,4); T(X,4); T(X,2); TL(Y,[1,5]); Seq(Y); T(X,17)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [1893120, 222464]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [1893120, 222464]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [1893120, 222464]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [ 1893120, 222464 ])

	return

def test_full_assoc_model_real_5_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[4,6]); U(C,2); T(C,64); T(Y,4); T(X,4); T(X,2); TL(Y,[1,5]); Seq(Y); T(X,17)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [1893120, 222464]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [1893120, 222464]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [1893120, 222464]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [ 1893120, 222464 ])

	return

def test_full_assoc_model_real_5_AllReuse():
	str_scheme = "[ V(F,16); U(F,4); UL(Y,[4,6]); U(C,2); T(C,64); T(Y,4); T(X,4); T(X,2); TL(Y,[1,5]); Seq(Y); T(X,17)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [1893120, 222464]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [1893120, 222464]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [1893120, 222464]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [ 1893120, 222464 ])

	return

def test_full_assoc_model_real_6_NoReuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3); T(X,136); TL(Y,[6,1]); Seq(Y); T(X,2); T(C,4); T(Y,2); T(Y,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [4516608, 4470528]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [4470528, 4470528]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [4454400, 4454400]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[4516608, 4470528])
	# Original implem: [ 4640256, 4474368 ] (MAX1_LOOP_REUSE - Match when there was a bug with a permutation of r/s)
	
	return

def test_full_assoc_model_real_6_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3); T(X,136); TL(Y,[6,1]); Seq(Y); T(X,2); T(C,4); T(Y,2); T(Y,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [4516608, 4470528]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [4470528, 4470528]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [4454400, 4454400]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[4470528, 4470528])
	
	return

def test_full_assoc_model_real_6_AllReuse():
	str_scheme = "[ V(F,16); U(F,2); UL(Y,[9,14]); T(C,8); T(F,2); T(H,3); T(X,136); TL(Y,[6,1]); Seq(Y); T(X,2); T(C,4); T(Y,2); T(Y,2); T(W,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [4516608, 4470528]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [4470528, 4470528]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [4454400, 4454400]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss==[4454400, 4454400])
	
	return
	
def test_full_assoc_model_real_7_NoReuse():
	str_scheme = "[ V(F,16); U(F,4); U(X,2); U(Y,2); U(C,4); T(C,16); T(F,2); T(X,7); T(Y,14); T(X,2); T(W,3); T(H,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [987840, 14480]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [987840, 14480]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [987840, 14480]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [987840, 14480])
	# Original implem: [ 987840, 14480 ]  (scheme cache resident !)

	return

def test_full_assoc_model_real_7_Max1Reuse():
	str_scheme = "[ V(F,16); U(F,4); U(X,2); U(Y,2); U(C,4); T(C,16); T(F,2); T(X,7); T(Y,14); T(X,2); T(W,3); T(H,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [987840, 14480]
	reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [987840, 14480]
	#reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [987840, 14480]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [987840, 14480])

	return

def test_full_assoc_model_real_7_AllReuse():
	str_scheme = "[ V(F,16); U(F,4); U(X,2); U(Y,2); U(C,4); T(C,16); T(F,2); T(X,7); T(Y,14); T(X,2); T(W,3); T(H,3)]"
	d_full_sizes = {"strx":1, "stry":1}
	scheme = build_scheme_from_str(str_scheme)
	scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
	# Note: subst needed else the access functions are looking at the wrong sizes here...

	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	prog_dims = get_ldims_computation(comp)
	cache_line_size = 16
	
	lcachesize = [ 8192, 262144 ] # L1/L2 sizes

	#reuse_strat = ReuseLoopStrat.NO_LOOP_REUSE        # ==> Result = [987840, 14480]
	#reuse_strat = ReuseLoopStrat.MAX1_LOOP_REUSE      # ==> Result = [987840, 14480]
	reuse_strat = ReuseLoopStrat.UNLIMITED_LOOP_REUSE # ==> Result = [987840, 14480]
	lcache_miss = compute_full_assoc_cache_misses(scheme, d_full_sizes, lcachesize, cache_line_size, comp, reuse_strat)

	assert(lcache_miss == [987840, 14480])

	return



