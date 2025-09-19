from xtc.schedules.ttile.computation import Computation, Computation_spec, get_array_accesses, compute_footprint

# Test of the pretty-printer
def test_print_computation():
	comp = Computation(Computation_spec.CONV, 4)
	assert( str(comp) == "conv2d(f32)")

	comp = Computation(Computation_spec.MATMULT, 8)
	assert( str(comp) == "matmul(f64)")

	return



# Tests of the "compute_footprint" function
def test_compute_footprint_1():
	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = { "i": [8], "j": [3], "k": [4]}
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert( ddfootprint == {'': {'A': 32, 'B': 12, 'C': 24}} )

	return

def test_compute_footprint_2():
	comp = Computation(Computation_spec.MATMULT, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = { "i": [8], "j": [3,4], "k": [4,5]}
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert(ddfootprint == {'j*0,k*0': {'A': 32, 'B': 12, 'C': 24},
		'j*1,k*0': {'A': 32, 'B': 16, 'C': 32},
		'j*0,k*1': {'A': 40, 'B': 15, 'C': 24},
		'j*1,k*1': {'A': 40, 'B': 20, 'C': 32}})

	return

def test_compute_footprint_3():
	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = { "c": [8], "f": [4,8], "r": [4], "s": [3], "h" : [32,16], "w" : [7], "n": [1] }
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert(ddfootprint == {'h*0,f*0': {'O': 896, 'I': 2520, 'K': 384},
		'h*0,f*1': {'O': 1792, 'I': 2520, 'K': 768},
		'h*1,f*0': {'O': 448, 'I': 1368, 'K': 384},
		'h*1,f*1': {'O': 896, 'I': 1368, 'K': 768}})

	return

def test_compute_footprint_4():
	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = {'n': [1], 'h': [272], 'w': [272], 'f': [64], 'r': [3], 's': [3], 'c': [32], 'strx': [1], 'stry': [1]}
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert(ddfootprint == {'': {'O': 4734976, 'I': 2402432, 'K': 18432}})

	return

def test_compute_footprint_5():
	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = {'n': [1], 'h': [272], 'w': [272], 'f': [64], 'r': [1], 's': [1], 'c': [32], 'strx': [2], 'stry': [2]}
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert(ddfootprint == {'': {'O': 4734976, 'I': 2358792, 'K': 2048}})

	return

def test_compute_footprint_6():
	comp = Computation(Computation_spec.CONV, 4) # float32
	d_arrays_accs = get_array_accesses(comp)
	d_lsizes = {'n': [1], 'h': [68], 'w': [10], 'f': [64], 'r': [1], 's': [3], 'c': [32], 'strx': [1], 'stry': [1]}
	ddfootprint = compute_footprint(comp, d_arrays_accs, d_lsizes)

	assert(ddfootprint == {'': {'O': 43520, 'I': 26112, 'K': 6144}})

	return

