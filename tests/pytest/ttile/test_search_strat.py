from pathlib import Path
import random

from xtc.schedules.ttile.scheme import convert_scheme_to_str, check_coherency_scheme, check_dim_coherency_scheme, normalize_scheme
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.archi import pinocchio_machine, laptop_guillaume_machine
from xtc.schedules.ttile.microkernel import load_microkernel_info
from xtc.schedules.ttile.prob_sizes import ddsizes_matmul, ddsizes_RN18

from xtc.schedules.ttile.search_strat import find_affine_combination_dividing
from xtc.schedules.ttile.search_strat import select_microkernel_ttile, convert_microkernel_strat_to_scheme
from xtc.schedules.ttile.search_strat import complete_scheme_from_mickern_ttile_div, full_ttile_algorithm
from xtc.schedules.ttile.search_strat import ttile_partial_tile_algorithm


# Test of the search of a/b/k such that (a*x1 + b*x2)*k = size_x
def test_find_affine_combination_dividing():
	x1 = 7
	x2 = 3
	size_x = 20
	l_abk = find_affine_combination_dividing(x1, x2, size_x)

	assert(l_abk == [ [2, 2, 1], [1, 1, 2] ])

	#Combination found:
	#- (2*7 + 2*3) * 1 = 20
	#- (1*7 + 1*3) * 2 = 20

	return


# Test of the selection of microkernels, using Ttile algorithm.
def test_select_microkernel_ttile():
	# Recovering the microkernel database
	name_outfile = str(Path(__file__).parent / "mickern_xtctvm_pinocchio_matmul_f32.csv")
	(machine_name, computation_name, ld_mickern_info) = load_microkernel_info(name_outfile)

	# Problem specification
	machine = pinocchio_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	assert(machine_name == machine.name)
	assert(str(comp) == computation_name)

	dprob_sizes = {'i': 28, 'j': 32, 'k': 64 }
	vector_dim = "j"
	unroll_order = ["j", "i", "k"]
	
	num_elem_vector = int(machine.vector_size / comp.elem_size)
	lmicker_strat = select_microkernel_ttile(ld_mickern_info, vector_dim, num_elem_vector, dprob_sizes,
		blambda=True, lambda_dim="i")

	# Expected result
	lexp_res_micker_strat = ["Single({'i': 7, 'j': 2, 'k': 1}|perf = 91.825)",
		"Single({'i': 14, 'j': 1, 'k': 1}|perf = 71.43)",
		"Single({'i': 14, 'j': 2, 'k': 1}|perf = 97.255)",
		"Lambda(2*{'i': 7, 'j': 2, 'k': 1} + 1*{'i': 14, 'j': 2, 'k': 1}|perf=93.635)",
		"Lambda(1*{'i': 8, 'j': 2, 'k': 1} + 2*{'i': 10, 'j': 2, 'k': 1}|perf=95.634)",
		"Lambda(2*{'i': 8, 'j': 2, 'k': 1} + 1*{'i': 12, 'j': 2, 'k': 1}|perf=95.217)",
		"Lambda(2*{'i': 9, 'j': 2, 'k': 1} + 1*{'i': 10, 'j': 2, 'k': 1}|perf=95.759)",
		"Lambda(1*{'i': 12, 'j': 2, 'k': 1} + 1*{'i': 16, 'j': 2, 'k': 1}|perf=70.911)",
		"Lambda(1*{'i': 13, 'j': 2, 'k': 1} + 1*{'i': 15, 'j': 2, 'k': 1}|perf=71.272)"]

	assert(len(lmicker_strat) == len(lexp_res_micker_strat))
	for i in range(len(lmicker_strat)):
		assert( str(lmicker_strat[i]) == lexp_res_micker_strat[i])

	# Convertion to scheme
	lexp_res_scheme_micker = [
		"[V(j,16); U(j,2); U(i,7); U(k,1)]",
		"[V(j,16); U(j,1); U(i,14); U(k,1)]",
		"[V(j,16); U(j,2); U(i,14); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [7, 14]); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [8, 10]); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [8, 12]); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [9, 10]); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [12, 16]); U(k,1)]",
		"[V(j,16); U(j,2); UL(i, [13, 15]); U(k,1)]"]
	for i in range(len(lmicker_strat)):
		scheme_mickern = convert_microkernel_strat_to_scheme(machine, comp, vector_dim, unroll_order, lmicker_strat[i])
		assert(lexp_res_scheme_micker[i] == convert_scheme_to_str(scheme_mickern))

	return


# Test of the drawing of a compete scheme from a microkernel.
def test_complete_scheme_from_mickern_ttile_div():
	# Deterministic test
	random.seed(42105105)

	name_outfile = str(Path(__file__).parent / "mickern_xtctvm_dummyresult_matmul_f32.csv")
	(machine_name, computation_name, ld_mickern_info) = load_microkernel_info(name_outfile)

	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	assert(machine_name == machine.name)
	assert(str(comp) == computation_name)

	dprob_sizes = {'i': 112, 'j': 32, 'k': 64 }
	lambda_dim = "i"
	vector_dim = "j"
	reuse_dim = "k"
	unroll_order = ["j", "i", "k"]
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	num_elem_vector = int(machine.vector_size / comp.elem_size)
	lmicker_strat = select_microkernel_ttile(ld_mickern_info, vector_dim, num_elem_vector,
		dprob_sizes, blambda=True, lambda_dim=lambda_dim)

	expected_res_full_scheme = [
		"[V(j,8); U(j,4); U(i,7); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); T(i,4); Tparal(i,2); Tparal(i,2)]",
		"[V(j,8); U(j,2); U(i,14); U(k,1); T(k,32); Hoist_var(['C']); T(i,2); T(k,2); T(i,2); Tparal(i,2); Tparal(j,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [2, 17]); Seq(i); T(k,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [8, 12]); Seq(i); T(k,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [4, 6]); Seq(i); T(k,2); Tparal(i,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [2, 3]); Seq(i); Tparal(i,2); Tparal(i,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [14, 7]); Seq(i)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [2, 1]); Seq(i); Tparal(i,7)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [20, 2]); Seq(i)]",
		"[V(j,8); U(j,4); UL(i, [5, 6]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [10, 1]); Seq(i); Tparal(i,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 7]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [7, 11]); Seq(i); T(k,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 7]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [14, 6]); Seq(i)]",
		"[V(j,8); U(j,4); UL(i, [5, 7]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [7, 3]); Seq(i); T(k,2); Tparal(i,2)]",
		"[V(j,8); U(j,4); UL(i, [5, 7]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [21, 1]); Seq(i)]",
		"[V(j,8); U(j,4); UL(i, [6, 7]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [7, 10]); Seq(i)]",
		"[V(j,8); U(j,4); UL(i, [6, 7]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [14, 4]); Seq(i); T(k,2)]",
		"[V(j,8); U(j,4); UL(i, [6, 7]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [7, 2]); Seq(i); T(k,2); Tparal(i,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 11]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [9, 2]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 12]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [4, 6]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 12]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [2, 3]); Seq(i); Tparal(j,2); Tparal(i,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 12]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [10, 1]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 13]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [6, 4]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 13]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [3, 2]); Seq(i); Tparal(j,2); Tparal(i,2)]",
		"[V(j,8); U(j,2); UL(i, [10, 14]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [7, 3]); Seq(i); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [11, 12]); U(k,1); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [8, 2]); Seq(i); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [11, 12]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [4, 1]); Seq(i); T(k,2); Tparal(i,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [11, 13]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [9, 1]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [12, 13]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [5, 4]); Seq(i); T(k,2); Tparal(j,2)]",
		"[V(j,8); U(j,2); UL(i, [12, 14]); U(k,1); T(k,32); Hoist_var(['C']); TL(i, [7, 2]); Seq(i); T(k,2); Tparal(j,2)]"]

	assert(len(lmicker_strat) == len(expected_res_full_scheme))

	# Select one microkernel
	for i in range(len(lmicker_strat)):
		micker_strat = lmicker_strat[i]
		full_scheme = complete_scheme_from_mickern_ttile_div(machine, comp, vector_dim, unroll_order, micker_strat,
			dprob_sizes, reuse_dim, 32, loutput_array_name, lparallel_dim, nthread=4)

		assert(convert_scheme_to_str(full_scheme) == expected_res_full_scheme[i])

	return


# Test of the Ttile algorithm
def test_full_ttile_algorithm():
	# Deterministic test
	random.seed(42105105)

	# Problem specification
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_dummyresult_matmul_f32.csv")
	dprob_sizes = {'i': 112, 'j': 32, 'k': 64 }
	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	nthread = None
	
	# Information about the scheme related to the computation	
	vector_dim = "j"
	lambda_dim = "i"
	reuse_dim = "k"
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["j", "i", "k"]
	reuse_loop_min = 32
	threshold_mickern_perf_ratio = 0.85
	
	# Let's go
	full_scheme = full_ttile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
			vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
			threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
			nthread=None)

	# Check that the scheme is well-built
	assert(check_coherency_scheme(full_scheme))
	assert(check_dim_coherency_scheme(full_scheme, dprob_sizes))
	
	full_scheme = normalize_scheme(full_scheme)

	expected_full_scheme = "[V(j,8); U(j,4); UL(i, [6, 7]); T(k,32); Hoist_var(['C']); T(k,2); TL(i, [7, 10]); Seq(i)]"
	assert(convert_scheme_to_str(full_scheme) == expected_full_scheme)

	return


# Test of the Ttile partial tile algorithm
def test_ttile_partial_tile_algorithm():
	# Deterministic test
	random.seed(42105105)

	# Problem 
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_dummyresult_matmul_f32.csv")
	dprob_sizes = {'i': 112, 'j': 32, 'k': 64 }
	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	nthread = None
	
	# Information about the scheme related to the computation	
	vector_dim = "j"
	lambda_dim = None   # No lambda with partial tiles
	reuse_dim = "k"
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["j", "i", "k"]
	reuse_loop_min = 32
	threshold_mickern_perf_ratio = 0.85
	
	# Let's go
	full_scheme = ttile_partial_tile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
			vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
			threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
			nthread=None)

	assert(full_scheme!=None)

	# Check that the scheme is well-built
	assert(check_coherency_scheme(full_scheme))
	assert(check_dim_coherency_scheme(full_scheme, dprob_sizes, verbose=True))

	expected_full_scheme = "[V(j,8); U(j,4); U(i,7); U(k,1); T(k,32); Hoist_var(['C']); Tpart(i,98); Tpart(k,64); Tpart(i,105); Tpart(i,112)]"
	assert(convert_scheme_to_str(full_scheme) == expected_full_scheme)

	return

# Test of the Ttile partial tile algorithm
def test_ttile_partial_tile_algorithm_parallel():
	# Deterministic test
	random.seed(42105105)

	# Problem 
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_dummyresult_matmul_f32.csv")
	dprob_sizes = {'i': 112, 'j': 32, 'k': 64 }
	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	nthread = None
	
	# Information about the scheme related to the computation	
	vector_dim = "j"
	lambda_dim = None   # No lambda with partial tiles
	reuse_dim = "k"
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["j", "i", "k"]
	reuse_loop_min = 32
	threshold_mickern_perf_ratio = 0.85
	
	# Let's go
	full_scheme = ttile_partial_tile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
			vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
			threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
			nthread=4)

	assert(full_scheme!=None)

	# Check that the scheme is well-built
	assert(check_coherency_scheme(full_scheme))
	assert(check_dim_coherency_scheme(full_scheme, dprob_sizes, verbose=True))
	
	expected_full_scheme = "[V(j,8); U(j,4); U(i,7); U(k,1); T(k,32); Hoist_var(['C']); Tpart(k,64); Tpart(i,21); Tpart(i,28); Tparal(i,2); Tparal(i,2)]"
	assert(convert_scheme_to_str(full_scheme) == expected_full_scheme)

	return


# Test of specific cases for the Ttile algorithm
def test_full_ttile_algorithm_pinocchio_mm_PolybLarge_1():
	# Deterministic test
	random.seed(42105105)

	# Problem specification
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_pinocchio_matmul_f32.csv")
	dprob_sizes = ddsizes_matmul["Polybench_large"]
	machine = pinocchio_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	nthread = None

	# Information about the scheme related to the computation       
	vector_dim = "j"
	lambda_dim = "i" # None
	reuse_dim = "k"
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["j", "i", "k"]
	reuse_loop_min = 16
	threshold_mickern_perf_ratio = 0.75

	expected_full_scheme = [
		"[V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(k,15); T(j,23); TL(i, [244, 3]); Seq(i)]",
		"[V(j,16); U(j,3); UL(i, [4, 9]); U(k,1); T(k,16); Hoist_var(['C']); T(k,3); T(j,23); T(k,25); TL(i, [241, 4]); Seq(i)]",
		"[V(j,16); U(j,3); UL(i, [4, 5]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [30, 176]); Seq(i); T(j,23); T(k,3); T(k,5); T(k,5)]",
		"[V(j,16); U(j,3); UL(i, [5, 7]); U(k,1); T(k,16); Hoist_var(['C']); T(k,25); T(k,3); TL(i, [123, 55]); Seq(i); T(j,23)]",
		"[V(j,16); U(j,3); UL(i, [5, 6]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [14, 5]); Seq(i); T(j,23); T(k,75); T(i,10)]",
		"[V(j,16); U(j,3); UL(i, [4, 7]); U(k,1); T(k,16); Hoist_var(['C']); T(k,75); T(j,23); TL(i, [138, 64]); Seq(i)]",
		"[V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,2); T(k,25); TL(i, [67, 29]); Seq(i); T(k,3)]",
		"[V(j,16); U(j,3); UL(i, [4, 9]); U(k,1); T(k,16); Hoist_var(['C']); T(k,3); T(k,5); TL(i, [41, 4]); Seq(i); T(j,23); T(k,5); T(i,5)]",
		"[V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [28, 11]); Seq(i); T(j,23); T(k,25); T(k,3); T(i,5)]",
		"[V(j,16); U(j,3); UL(i, [4, 6]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [122, 2]); Seq(i); T(j,23); T(i,2); T(k,75)]"]

	# Let's go
	for i in range(10):
		full_scheme = full_ttile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
							vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
							threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
							nthread=None)

		assert(full_scheme!=None)

		# Check that the scheme is well-built
		assert(check_coherency_scheme(full_scheme))
		assert(check_dim_coherency_scheme(full_scheme, dprob_sizes))

		assert(convert_scheme_to_str(full_scheme) == expected_full_scheme[i])

	return

def test_full_ttile_algorithm_pinocchio_mm_PolybLarge_2():
	# Deterministic test
	random.seed(42105105)

	# Problem specification
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_pinocchio_matmul_f32.csv")
	dprob_sizes = ddsizes_matmul["Polybench_large"]
	machine = pinocchio_machine
	comp = Computation(Computation_spec.MATMULT, 4)
	nthread = None

	# Information about the scheme related to the computation       
	vector_dim = "j"
	lambda_dim = None
	reuse_dim = "k"
	lparallel_dim = ["i","j"]
	loutput_array_name = [ "C" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["j", "i", "k"]
	reuse_loop_min = 16
	threshold_mickern_perf_ratio = 0.75

	expected_full_scheme = [
		"[V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(k,15); T(j,23); T(i,200)]",
		"[V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(k,3); T(i,5); T(i,25); T(k,25)]",
		"[V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,100); T(k,3); T(k,25); T(i,2)]",
		"[V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,25); T(k,15); T(i,5); T(k,5)]",
		"[V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,125); T(k,15); T(k,5)]",
		"[V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(i,125); T(j,23); T(k,5); T(k,3); T(i,2)]",
		"[V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,5); T(i,10); T(k,5); T(k,15); T(i,4)]",
		"[V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(i,5); T(i,5); T(k,5); T(k,15); T(i,5); T(j,23)]",
		"[V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(k,75); T(i,50); T(i,5)]",
		"[V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(k,15); T(i,2); T(k,5); T(i,125); T(j,23)]"]

	# Let's go
	for i in range(10):
		full_scheme = full_ttile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
							vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
							threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
							nthread=None)

		assert(full_scheme!=None)

		# Check that the scheme is well-built
		assert(check_coherency_scheme(full_scheme))
		assert(check_dim_coherency_scheme(full_scheme, dprob_sizes))

		assert(convert_scheme_to_str(full_scheme) == expected_full_scheme[i])

	return



	# === Result when "lambda_dim = None"
	# full_scheme = [V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(k,15); T(j,23); T(i,200)]
	# full_scheme = [V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(k,3); T(i,5); T(i,25); T(k,25)]
	# full_scheme = [V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,100); T(k,3); T(k,25); T(i,2)]
	# full_scheme = [V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,25); T(k,15); T(i,5); T(k,5)]
	# full_scheme = [V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,125); T(k,15); T(k,5)]
	# full_scheme = [V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(i,125); T(j,23); T(k,5); T(k,3); T(i,2)]
	# full_scheme = [V(j,16); U(j,3); U(i,5); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,5); T(i,10); T(k,5); T(k,15); T(i,4)]
	# full_scheme = [V(j,16); U(j,3); U(i,8); U(k,1); T(k,16); Hoist_var(['C']); T(i,5); T(i,5); T(k,5); T(k,15); T(i,5); T(j,23)]
	# full_scheme = [V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(k,75); T(i,50); T(i,5)]
	# full_scheme = [V(j,16); U(j,3); U(i,4); U(k,1); T(k,16); Hoist_var(['C']); T(k,15); T(i,2); T(k,5); T(i,125); T(j,23)]

	# === Result when "lambda_dim = i"
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); T(k,5); T(k,15); T(j,23); TL(i, [244, 3]); Seq(i)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 9]); U(k,1); T(k,16); Hoist_var(['C']); T(k,3); T(j,23); T(k,25); TL(i, [241, 4]); Seq(i)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 5]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [30, 176]); Seq(i); T(j,23); T(k,3); T(k,5); T(k,5)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [5, 7]); U(k,1); T(k,16); Hoist_var(['C']); T(k,25); T(k,3); TL(i, [123, 55]); Seq(i); T(j,23)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [5, 6]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [14, 5]); Seq(i); T(j,23); T(k,75); T(i,10)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 7]); U(k,1); T(k,16); Hoist_var(['C']); T(k,75); T(j,23); TL(i, [138, 64]); Seq(i)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); T(j,23); T(i,2); T(k,25); TL(i, [67, 29]); Seq(i); T(k,3)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 9]); U(k,1); T(k,16); Hoist_var(['C']); T(k,3); T(k,5); TL(i, [41, 4]); Seq(i); T(j,23); T(k,5); T(i,5)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 8]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [28, 11]); Seq(i); T(j,23); T(k,25); T(k,3); T(i,5)]
	# full_scheme = [V(j,16); U(j,3); UL(i, [4, 6]); U(k,1); T(k,16); Hoist_var(['C']); TL(i, [122, 2]); Seq(i); T(j,23); T(i,2); T(k,75)]

	return


# Test of the full ttile algorithm on Pinocchio
def test_full_ttile_algorithm_pinocchio_conv_RN02():
	# Deterministic test
	random.seed(42105105)

	# Problem specification
	filename_mickern_info = str(Path(__file__).parent / "mickern_xtctvm_pinocchio_conv_f32.csv")
	dprob_sizes = ddsizes_RN18["ResNet18_02"]
	machine = pinocchio_machine
	comp = Computation(Computation_spec.CONV, 4)
	nthread = None

	# Information about the scheme related to the computation       
	vector_dim = "f"
	lambda_dim = "x"
	reuse_dim = "c"
	lparallel_dim = ["f","x", "y"]
	loutput_array_name = [ "O" ]

	# Parameters of the Ttile algorithm
	unroll_order = ["f", "x", "y", "c", "w", "h"]
	reuse_loop_min = 32
	threshold_mickern_perf_ratio = 0.75

	expected_full_scheme = [
		"[V(f,16); U(f,2); UL(x, [3, 4]); U(y,2); U(c,2); U(w,1); U(h,1); T(c,32); Hoist_var(['O']); T(f,2); T(y,7); TL(x, [1, 1]); Seq(x); T(h,3); T(x,2); T(x,4); T(w,3); T(y,4)]",
		"[V(f,16); U(f,2); U(x,4); U(y,2); U(c,1); U(w,1); U(h,3); T(c,32); Hoist_var(['O']); T(f,2); T(x,2); T(w,3); T(x,7); T(c,2); T(y,2); T(y,14)]",
		"[V(f,16); U(f,4); UL(x, [2, 3]); U(y,2); U(c,2); U(w,1); U(h,1); T(c,32); Hoist_var(['O']); T(y,7); T(y,2); T(h,3); T(w,3); TL(x, [25, 2]); Seq(x); T(y,2)]",
		"[V(f,16); U(f,2); UL(x, [1, 2]); U(y,7); U(c,1); U(w,1); U(h,1); T(c,32); Hoist_var(['O']); T(y,2); T(f,2); T(h,3); TL(x, [18, 19]); Seq(x); T(y,2); T(y,2); T(w,3); T(c,2)]",
		"[V(f,16); U(f,2); UL(x, [6, 4]); U(y,2); U(c,1); U(w,1); U(h,3); T(c,32); Hoist_var(['O']); TL(x, [4, 8]); Seq(x); T(y,28); T(w,3); T(c,2); T(f,2)]",
		"[V(f,16); U(f,4); UL(x, [4, 5]); U(y,1); U(c,1); U(w,1); U(h,1); T(c,32); Hoist_var(['O']); T(y,2); T(c,2); T(h,3); T(w,3); TL(x, [9, 4]); Seq(x); T(y,4); T(y,7)]",
		"[V(f,16); U(f,2); UL(x, [3, 2]); U(y,4); U(c,1); U(w,1); U(h,3); T(c,32); Hoist_var(['O']); TL(x, [1, 2]); Seq(x); T(f,2); T(y,2); T(y,7); T(c,2); T(w,3); T(x,4); T(x,2)]",
		"[V(f,16); U(f,2); UL(x, [10, 8]); U(y,1); U(c,8); U(w,1); U(h,1); T(c,8); Hoist_var(['O']); T(y,28); T(w,3); TL(x, [4, 2]); Seq(x); T(f,2); T(y,2); T(h,3)]",
		"[V(f,16); U(f,2); UL(x, [5, 8]); U(y,1); U(c,2); U(w,1); U(h,3); T(c,32); Hoist_var(['O']); T(y,2); T(y,28); T(x,2); TL(x, [4, 1]); Seq(x); T(w,3); T(f,2)]",
		"[V(f,16); U(f,2); UL(x, [1, 2]); U(y,4); U(c,2); U(w,3); U(h,1); T(c,32); Hoist_var(['O']); TL(x, [10, 9]); Seq(x); T(y,2); T(x,2); T(f,2); T(y,7); T(h,3)]"]
	

	# Let's go
	for i in range(10):
		full_scheme = full_ttile_algorithm(filename_mickern_info, dprob_sizes, machine, comp,
							vector_dim, lambda_dim, reuse_dim, lparallel_dim, loutput_array_name,
							threshold_mickern_perf_ratio, unroll_order, reuse_loop_min,
							nthread=None)
		assert(full_scheme != None)

		# Check that the scheme is well-built
		assert(check_coherency_scheme(full_scheme))
		assert(check_dim_coherency_scheme(full_scheme, dprob_sizes, l_ignored_dims=['strx', 'stry'], verbose=True))

		# Result
		assert( convert_scheme_to_str(full_scheme) == expected_full_scheme[i])

	return

