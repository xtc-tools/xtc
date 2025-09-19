from xtc.schedules.ttile.scheme import build_scheme_from_str, convert_scheme_to_str, check_coherency_scheme, get_sizes_scheme, check_dim_coherency_scheme

# Test of the parsing function (str to scheme) and the print function
def test_parse_print_1():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32) ]"
	
	scheme = build_scheme_from_str(str_scheme)
	str_scheme_out = convert_scheme_to_str(scheme)
	print(str_scheme_out)

	# Can we go back and forth?
	scheme_2 = build_scheme_from_str(str_scheme_out)
	str_scheme_out_2 = convert_scheme_to_str(scheme_2)
	
	assert(str_scheme_out == str_scheme_out_2)

	return

def test_parse_print_2():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32); Tpart(J,31); Hoist_var ['O']; Hoist_var ['A','B']; UL(I,[3,4]); TL(I,[48,50]); Seq(I); Tparal(J,4) ]"
	
	scheme = build_scheme_from_str(str_scheme)
	str_scheme_out = convert_scheme_to_str(scheme)
	print(str_scheme_out)

	# Can we go back and forth?
	scheme_2 = build_scheme_from_str(str_scheme_out)
	str_scheme_out_2 = convert_scheme_to_str(scheme_2)
	
	assert(str_scheme_out == str_scheme_out_2)

	return

def test_parse_print_3():
	# (Real scheme for ResNet04 on a AVX512 Intel machine)
	str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (T (H, 3))]"

	scheme = build_scheme_from_str(str_scheme)
	str_scheme_out = convert_scheme_to_str(scheme)
	print(str_scheme_out)

	# Can we go back and forth?
	scheme_2 = build_scheme_from_str(str_scheme_out)
	str_scheme_out_2 = convert_scheme_to_str(scheme_2)
	
	assert(str_scheme_out == str_scheme_out_2)

	return


# Test of the coherency of the scheme, in terms of organisation of the atoms
def test_check_coherency_scheme_1():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32) ]"					# True
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(b_coherency)

	return

def test_check_coherency_scheme_2():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32); U(J, 2) ]"			# False
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(not b_coherency)
	
	return

def test_check_coherency_scheme_3():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32) ]"				# False
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(not b_coherency)
	
	return

def test_check_coherency_scheme_4():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); Seq(I) ]"		# True
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(b_coherency)
	
	return

def test_check_coherency_scheme_5():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); Seq(J) ]"		# False
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(not b_coherency)
	
	return

def test_check_coherency_scheme_6():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10,12]); Seq(I) ]"		# True
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(b_coherency)
	
	return

def test_check_coherency_scheme_7():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(J,[10]); Seq(I); Seq(J) ]"	# True
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(b_coherency)
	
	return

def test_check_coherency_scheme_8():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10]); Seq(I) ]"			# False
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(not b_coherency)
	
	return

def test_check_coherency_scheme_9():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[1,2]); Seq(I); Seq(I) ]"	# False
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	assert(not b_coherency)
	
	return


# Test of the computation of the size of a scheme
def test_get_sizes_scheme_1():
	str_scheme = "[ V(J,16) ; U(J,2); U(I,6) ; T(K,32); T(I,10); T(J,4); T(K,4) ]"
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	d_sizes = get_sizes_scheme(scheme)

	assert(d_sizes == {'j': [128], 'i': [60], 'k': [128]})

	return

def test_get_sizes_scheme_2():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32); Tpart(K, 40) ]"
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	d_sizes = get_sizes_scheme(scheme)

	assert(d_sizes == {'j': [16], 'i': [3], 'k': [40]})

	return

def test_get_sizes_scheme_3():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10,12]); Seq(I) ]"
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	d_sizes = get_sizes_scheme(scheme)

	assert(d_sizes == {'j': [16], 'i': [78], 'k': [32]})

	return

def test_get_sizes_scheme_4():
	str_scheme = "[ V(J,16); U(J,2); UL(I,[12,13]); T(K,32);  T(J,7); TL(I, [1,2]); Seq(I); T(J,8); T(I,9); T(K,16); T(I,2)]"
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)

	d_sizes = get_sizes_scheme(scheme)

	assert(d_sizes == {'j': [1792], 'i': [684], 'k': [512]})

	return


# Test of the computation of the size of a scheme
def test_check_dim_coherency_scheme_1():
	str_scheme = "[ V(J,16) ; U(J,2); U(I,6) ; T(K,32); T(I,10); T(J,4); T(K,4) ]"
	d_prob_dims = { "j" : 128, "i" : 60, "k" : 128}
	
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)
	b_size_coherency = check_dim_coherency_scheme(scheme, d_prob_dims, verbose=True)

	assert(b_coherency)
	assert(b_size_coherency)

	return

def test_check_dim_coherency_scheme_2():
	str_scheme = "[ V(J,16) ; U(I,3) ; T(K,32); Tpart(K, 40) ]"
	d_prob_dims = { "j" : 16, "i" : 3, "k" : 40}
	
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)
	b_size_coherency = check_dim_coherency_scheme(scheme, d_prob_dims, verbose=True)

	assert(b_coherency)
	assert(b_size_coherency)

	return

def test_check_dim_coherency_scheme_3():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10,12]); Seq(I) ]"
	d_prob_dims = { "j" : 16, "i" : 78, "k" : 32}		# True
	
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)
	b_size_coherency = check_dim_coherency_scheme(scheme, d_prob_dims, verbose=True)

	assert(b_coherency)
	assert(b_size_coherency)

	return

def test_check_dim_coherency_scheme_4():
	str_scheme = "[ V(J,16) ; UL(I,[3,4]) ; T(K,32); TL(I,[10,12]); Seq(I) ]"
	d_prob_dims = { "j" : 16, "i" : 72, "k" : 32}		# False
	
	scheme = build_scheme_from_str(str_scheme)
	b_coherency = check_coherency_scheme(scheme, verbose=True)
	b_size_coherency = check_dim_coherency_scheme(scheme, d_prob_dims, verbose=False)

	assert(b_coherency)
	assert(not b_size_coherency)

	return
