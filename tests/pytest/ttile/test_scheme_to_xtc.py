
from xtc.schedules.ttile.scheme import build_scheme_from_str
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.archi import laptop_guillaume_machine, pinocchio_machine

from xtc.schedules.ttile.scheme_to_xtc import get_descr_sched, subst_dimname_xyhw_to_hwrs_conv2D_scheme
from xtc.schedules.ttile.scheme_to_xtc import build_xdsl_module_string_matmul, build_xdsl_module_string_conv
from xtc.schedules.ttile.scheme_to_xtc import launch_and_measure_scheme, launch_and_measure_scheme_graph_interf

from ttile_utils import requires_tvm, requires_pmu

# Test "get_descr_sched" (both modes)
def test_get_descr_sched_mlir_loops_1():
  comp = Computation(Computation_spec.MATMULT, 4)
  
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_loopschedule = get_descr_sched(scheme, comp, machine, False)
  expected_str = """{
"k",
"j",
"i",
"j#4096",
"k#1024",
"i#8"= {"unroll" = 8},
"j#32"= {"unroll" = 32},
"j#16"= { "vectorize" }
}"""
  assert(str_loopschedule == expected_str)
  return


def test_get_descr_sched_graph_1():
  comp = Computation(Computation_spec.CONV, 4)  # f32
  machine = pinocchio_machine

  str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (T (H, 3))]"
  scheme = build_scheme_from_str(str_scheme)
  scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
  
  #dsizes = {"n" : 1, "f" : 128, "c": 64, "h": 28, "w": 28, "r": 3, "s": 3, "strx": 1, "stry": 1}
  
  str_descriptor = get_descr_sched(scheme, comp, machine, True)
  expected_str = """{
"n": {},
"r": {},
"s": {},
"h": {},
"w": {},
"h#14": {},
"f": {},
"c": {},
"c#4": {"unroll" : 4},
"w#2": {"unroll" : 2},
"h#2": {"unroll" : 2},
"f#64": {"unroll" : 64},
"f#16": {"vectorize" : True}
}"""
  assert(str_descriptor == expected_str)
  return

def test_get_descr_sched_graph_2():
  comp = Computation(Computation_spec.CONV, 4)  # f32
  machine = pinocchio_machine

  str_scheme = "[V(F,16); U(F,4); UL(Y,[5,7]); U(C,2); T(C,8); T(X,34); T(X,2); T(Y,2); T(C,2); T(H,3); T(X,4); T(Y,2); T(W,3); T(Y,4); TL(Y,[2,1]); Seq(Y)]"
  scheme = build_scheme_from_str(str_scheme)
  scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
  
  #dsizes = {"n" : 1, "f" : 64, "c": 32, "h": 272, "w": 272, "r": 3, "s": 3, "strx": 1, "stry": 1}

  str_descriptor = get_descr_sched(scheme, comp, machine, True)
  d_desc = eval(str_descriptor)
  
  str_expected = "{'n': {}, 'w[0:160]': {'w#160': {}, 'w#80': {}, 's': {}, 'w#20': {}, 'h': {}, 'r': {}, 'c': {}, 'w#10': {}, 'h#68': {}, 'h#34': {}, 'c#16': {}, 'c#2': {'unroll': 2}, 'w#5': {'unroll': 5}, 'f': {'unroll': 64}, 'f#16': {'vectorize': True}}, 'w[160:272]': {'w#112': {}, 's': {}, 'w#28': {}, 'h': {}, 'r': {}, 'c': {}, 'w#14': {}, 'h#68': {}, 'h#34': {}, 'c#16': {}, 'c#2': {'unroll': 2}, 'w#7': {'unroll': 7}, 'f': {'unroll': 64}, 'f#16': {'vectorize': True}}}"
  
  assert(str(d_desc) == str_expected)
  return

# Test of "build_xdsl_module_string_matmul"
def test_build_xdsl_module_string_matmul():
  comp = Computation(Computation_spec.MATMULT, 4)
  
  dsizes = { "i" : 256, "j": 8192, "k": 2048 }
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_mlir_out = build_xdsl_module_string_matmul(comp, machine, scheme, dsizes)
  expected_output = """func.func @myfun(
  %A: memref<256x2048xf32>,
  %B: memref<2048x8192xf32>,
  %C: memref<256x8192xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.schedule = {
"k",
"j",
"i",
"j#4096",
"k#1024",
"i#8"= {"unroll" = 8},
"j#32"= {"unroll" = 32},
"j#16"= { "vectorize" }
}
    }
    ins(%A, %B : memref<256x2048xf32>, memref<2048x8192xf32>)
    outs(%C : memref<256x8192xf32>)
  return
}
"""
  assert(str_mlir_out==expected_output)
  return



# Test of "build_xdsl_module_string_conv"
def test_build_xdsl_module_string_conv():
  comp = Computation(Computation_spec.CONV, 4)
  
  dsizes = {"f" : 128, "c": 64, "h": 28, "w": 28, "r": 3, "s": 3, "strx": 1, "stry": 1}
  str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (T (H, 3))]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_mlir_out = build_xdsl_module_string_conv(comp, machine, scheme, dsizes)
  expected_output="""func.func @myfun(
  %I: memref<1x3x3x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x1x1x128xf32>
) {
  linalg.generic {
      indexing_maps = [
        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,
        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,
        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
          "reduction", "reduction", "reduction"]
    }
    ins(%I, %K : memref<1x3x3x64xf32>, memref<3x3x64x128xf32>)
    outs(%O : memref<1x1x1x128xf32>)
     attrs = {
      loop.dims = ["n","h","w","f","r","s","c"],
      loop.schedule = {
"n",
"r",
"s",
"h",
"w",
"h#14",
"f",
"c",
"c#4"= {"unroll" = 4},
"w#2"= {"unroll" = 2},
"h#2"= {"unroll" = 2},
"f#64"= {"unroll" = 64},
"f#16"= { "vectorize" }
}
     }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}
"""
  assert(str_mlir_out == expected_output)
  return


# Test of "launch_and_measure_scheme_matmul"
# WARNING - actual execution on a machine. This was done on "laptop_guillaume_machine"
def test_launch_and_measure_scheme_matmul():
  comp = Computation(Computation_spec.MATMULT, 4)
  dsizes = { "i" : 8, "j": 32, "k": 64 }
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 64); T(I, 1); T(J, 1)]"
  # Note: T(I, 1) + T(J, 1) due to a current limitation of xtc, that forbid a dimension to be fully unrolled
  #   (a tile which is not the full problem must be the one to be unrolled). Thus, this "hacky" workaround.
  scheme = build_scheme_from_str(str_scheme)
  machine = laptop_guillaume_machine

  res_measurement = launch_and_measure_scheme(comp, machine, scheme, dsizes)

  assert("peak_perf" in res_measurement.keys())

  return

def test_launch_and_measure_scheme_matmul_2():
  comp = Computation(Computation_spec.MATMULT, 4)
  dsizes = { "i" : 8, "j": 32, "k": 64 }
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 64); T(I, 1); T(J, 1)]"
  # Note: T(I, 1) + T(J, 1) due to a current limitation of xtc, that forbid a dimension to be fully unrolled
  #   (a tile which is not the full problem must be the one to be unrolled). Thus, this "hacky" workaround.
  scheme = build_scheme_from_str(str_scheme)
  machine = laptop_guillaume_machine

  res_measurement = launch_and_measure_scheme(comp, machine, scheme, dsizes)

  assert("peak_perf" in res_measurement.keys())
  return

# Test of "launch_and_measure_scheme_conv"
# WARNING - actual execution on a machine. This was done on "laptop_guillaume_machine"
def test_launch_and_measure_scheme_conv():
  comp = Computation(Computation_spec.CONV, 4)  # f32
  dsizes = {"n": 1, "f" : 64, "c": 32, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}
  str_scheme = "[V (F, 16); U (F, 2); U (C, 4); T (C, 16); (T (F, 2))]"
  scheme = build_scheme_from_str(str_scheme)
  machine = laptop_guillaume_machine

  res_measurement = launch_and_measure_scheme(comp, machine, scheme, dsizes)

  assert("peak_perf" in res_measurement.keys())

  return


# Test "launch_and_measure_scheme_graph_interf" on a matmult
def test_launch_and_measure_scheme_graph_interf_matmult_mlir():
	comp = Computation(Computation_spec.MATMULT, 4)
	machine = laptop_guillaume_machine

	str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
	scheme = build_scheme_from_str(str_scheme)

	dsizes = { "i" : 256, "j": 8192, "k": 2048 }

	backend = "mlir"

	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)

	assert("peak_perf" in res.keys())
	
	return


@requires_tvm
def test_launch_and_measure_scheme_graph_interf_matmult_tvm():
	comp = Computation(Computation_spec.MATMULT, 4)
	machine = laptop_guillaume_machine

	str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
	scheme = build_scheme_from_str(str_scheme)

	dsizes = { "i" : 256, "j": 8192, "k": 2048 }

	backend = "tvm"
	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)

	assert("peak_perf" in res.keys())
	
	return


# Test "launch_and_measure_scheme_graph_interf" on a conv
@requires_tvm
def test_launch_and_measure_scheme_graph_interf_conv_mlir():
	comp = Computation(Computation_spec.CONV, 4)  # f32
	machine = laptop_guillaume_machine

	str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); T (F, 2); T (X, 7); T (Y, 14); T (X, 2); T (W, 3); T (H, 3)]"
	scheme = build_scheme_from_str(str_scheme)
	
	dsizes = {"n" : 1, "f" : 128, "c": 64, "x": 28, "y": 28, "h": 3, "w": 3, "strx": 1, "stry": 1}
	
	backend = "mlir"
	#backend = "tvm"

	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)
	
	assert("peak_perf" in res.keys())

	return

@requires_tvm
def test_launch_and_measure_scheme_graph_interf_conv_tvm():
	comp = Computation(Computation_spec.CONV, 4)  # f32
	machine = laptop_guillaume_machine

	str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); T (F, 2); T (X, 7); T (Y, 14); T (X, 2); T (W, 3); T (H, 3)]"
	scheme = build_scheme_from_str(str_scheme)
	
	dsizes = {"n" : 1, "f" : 128, "c": 64, "x": 28, "y": 28, "h": 3, "w": 3, "strx": 1, "stry": 1}
	
	backend = "tvm"

	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)
	
	assert("peak_perf" in res.keys())

	return

#   ... With another data type
@requires_tvm
def test_launch_and_measure_scheme_graph_interf_conv_tvm_f64():
	comp = Computation(Computation_spec.CONV, 8)  # f64
	machine = laptop_guillaume_machine

	str_scheme = "[V (F, 8); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); T (F, 2); T (X, 7); T (Y, 14); T (X, 2); T (W, 3); T (H, 3)]"
	scheme = build_scheme_from_str(str_scheme)
	
	dsizes = {"n" : 1, "f" : 128, "c": 64, "x": 28, "y": 28, "h": 3, "w": 3, "strx": 1, "stry": 1}
	
	backend = "tvm"

	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)
	
	assert("peak_perf" in res.keys())

	return

#   ... With various atoms (including partial tiles and parallelism)
def test_launch_and_measure_scheme_graph_interf_matmul_partial_parall():
	comp = Computation(Computation_spec.MATMULT, 4)  # f32
	machine = laptop_guillaume_machine

	str_scheme = "[V(j,8); U(j,4); U(i,7); U(k,1); T(k,32); Hoist_var(['C']); Tpart(k,64); Tpart(i,21); Tpart(i,28); Tparal(i,2); Tparal(i,2)]"
	scheme = build_scheme_from_str(str_scheme)
	
	dsizes = {'i': 112, 'j': 32, 'k': 64 }
	
	backend = "mlir"

	res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend)
	
	assert("peak_perf" in res.keys())

	return

#   ... With pmu_counters
@requires_tvm
@requires_pmu
def test_launch_and_measure_scheme_graph_interf_pmu_counters():
  comp = Computation(Computation_spec.MATMULT, 4)
  machine = laptop_guillaume_machine

  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
  scheme = build_scheme_from_str(str_scheme)

  dsizes = { "i" : 256, "j": 8192, "k": 2048 }

  #backend = "mlir"
  backend = "tvm"

  res = launch_and_measure_scheme_graph_interf(comp, machine, scheme, dsizes, backend,
                                               pmu_counters=["cycles", "l1d.replacement"]) #, l_verbose=[False,False,True])

  assert("cycles" in res.keys())
  assert("l1d.replacement" in res.keys())

  return
