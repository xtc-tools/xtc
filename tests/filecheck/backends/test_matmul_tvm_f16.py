# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 4, 32, 256, "float16"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.strip_mine("i", {"i1": 2})
sch.strip_mine("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    print_source_ir=True,
    print_transformed_ir=True,
    dump_file="matmul_f16_tvm",
)
module = comp.compile(sched)

executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x256xfloat16
# CHECK-NEXT:    - %1 : 256x32xfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat16
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x256xfloat16, 256x32xfloat16] -> [4x32xfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((4, 256), "float16"), _1: T.Buffer((256, 32), "float16"), C: T.Buffer((4, 32), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k in T.grid(4, 32, 256):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float16(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  i, i1, = sch.split(i, factors=[None, 2])
# CHECK-NEXT:  j, j1, = sch.split(j, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(k, i, j, i1, j1)
# CHECK-NEXT:  sch.unroll(i1)
# CHECK-NEXT:  sch.vectorize(j1)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((4, 256), "float16"), _1: T.Buffer((256, 32), "float16"), C: T.Buffer((4, 32), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0_init, j_0_init in T.grid(2, 2):
# CHECK-NEXT:              for i_1_init in T.unroll(2):
# CHECK-NEXT:                  for j_1_init in T.vectorized(16):
# CHECK-NEXT:                      with T.sblock("C_init"):
# CHECK-NEXT:                          v_i = T.axis.spatial(4, i_0_init * 2 + i_1_init)
# CHECK-NEXT:                          v_j = T.axis.spatial(32, j_0_init * 16 + j_1_init)
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes(C[v_i, v_j])
# CHECK-NEXT:                          C[v_i, v_j] = T.float16(0.0)
# CHECK-NEXT:          for k, i_0, j_0 in T.grid(256, 2, 2):
# CHECK-NEXT:              for i_1 in T.unroll(2):
# CHECK-NEXT:                  for j_1 in T.vectorized(16):
# CHECK-NEXT:                      with T.sblock("C_update"):
# CHECK-NEXT:                          v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                          v_j = T.axis.spatial(32, j_0 * 16 + j_1)
# CHECK-NEXT:                          v_k = T.axis.reduce(256, k)
# CHECK-NEXT:                          T.reads(C[v_i, v_j], _0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                          T.writes(C[v_i, v_j])
# CHECK-NEXT:                          C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  CODE: 0
