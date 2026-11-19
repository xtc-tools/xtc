# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K = 4, 32, 256
a = O.tensor((I, K), "float16", name="A")
b = O.tensor((K, J), "float16", name="B")

with O.graph(name="matmul_f16_acc_f32") as gb:
    m = O.matmul(a, b, name="C", acc_dtype="float32")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="C")
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.tile("k", {"k1": 64})
sch.fuse_consumer_at("j")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_f16_fused_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: matmul_f16_acc_f32
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x256xfloat16
# CHECK-NEXT:    - %1 : 256x32xfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 4x32xfloat16
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1, acc_dtype=float32) {name = 'C'} : [4x256xfloat16, 256x32xfloat16] -> [4x32xfloat16]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [4x32xfloat16] -> [4x32xfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_f16_acc_f32(_0: T.Buffer((4, 256), "float16"), _1: T.Buffer((256, 32), "float16"), relu: T.Buffer((4, 32), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          C = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          C_cast = T.sblock_alloc_buffer((4, 32), "float16")
# CHECK-NEXT:          for i, j, k in T.grid(4, 32, 256):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + T.Cast("float32", _0[v_i, v_k]) * T.Cast("float32", _1[v_k, v_j])
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("C_cast"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(C[v_i, v_j])
# CHECK-NEXT:                  T.writes(C_cast[v_i, v_j])
# CHECK-NEXT:                  C_cast[v_i, v_j] = T.Cast("float16", C[v_i, v_j])
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("relu"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(C_cast[v_i, v_j])
# CHECK-NEXT:                  T.writes(relu[v_i, v_j])
# CHECK-NEXT:                  relu[v_i, v_j] = T.max(T.float16(0.0), C_cast[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  O_cast = sch.get_sblock("C_cast")
# CHECK-NEXT:  O_F0 = sch.get_consumers(O_cast)[0]
# CHECK-NEXT:  i, i1, = sch.split(i, factors=[None, 2])
# CHECK-NEXT:  j, j1, = sch.split(j, factors=[None, 16])
# CHECK-NEXT:  k, k1, = sch.split(k, factors=[None, 64])
# CHECK-NEXT:  sch.reorder(i, i1, j, j1, k, k1)
# CHECK-NEXT:  sch.reverse_compute_at(O_cast, j)
# CHECK-NEXT:  sch.reverse_compute_at(O_F0, j)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_f16_acc_f32(_0: T.Buffer((4, 256), "float16"), _1: T.Buffer((256, 32), "float16"), relu: T.Buffer((4, 32), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          C = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          C_cast = T.sblock_alloc_buffer((4, 32), "float16")
# CHECK-NEXT:          for i_0, i_1, j_0 in T.grid(2, 2, 2):
# CHECK-NEXT:              for j_1 in range(16):
# CHECK-NEXT:                  with T.sblock("C_init"):
# CHECK-NEXT:                      v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                      v_j = T.axis.spatial(32, j_0 * 16 + j_1)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[v_i, v_j])
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  for k_0, k_1 in T.grid(4, 64):
# CHECK-NEXT:                      with T.sblock("C_update"):
# CHECK-NEXT:                          v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                          v_j = T.axis.spatial(32, j_0 * 16 + j_1)
# CHECK-NEXT:                          v_k = T.axis.reduce(256, k_0 * 64 + k_1)
# CHECK-NEXT:                          T.reads(C[v_i, v_j], _0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                          T.writes(C[v_i, v_j])
# CHECK-NEXT:                          C[v_i, v_j] = C[v_i, v_j] + T.Cast("float32", _0[v_i, v_k]) * T.Cast("float32", _1[v_k, v_j])
# CHECK-NEXT:              for ax0 in range(16):
# CHECK-NEXT:                  with T.sblock("C_cast"):
# CHECK-NEXT:                      v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                      v_j = T.axis.spatial(32, j_0 * 16 + ax0)
# CHECK-NEXT:                      T.reads(C[v_i, v_j])
# CHECK-NEXT:                      T.writes(C_cast[v_i, v_j])
# CHECK-NEXT:                      C_cast[v_i, v_j] = T.Cast("float16", C[v_i, v_j])
# CHECK-NEXT:              for ax0 in range(16):
# CHECK-NEXT:                  with T.sblock("relu"):
# CHECK-NEXT:                      v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                      v_j = T.axis.spatial(32, j_0 * 16 + ax0)
# CHECK-NEXT:                      T.reads(C_cast[v_i, v_j])
# CHECK-NEXT:                      T.writes(relu[v_i, v_j])
# CHECK-NEXT:                      relu[v_i, v_j] = T.max(T.float16(0.0), C_cast[v_i, v_j])
# CHECK-NEXT:  CODE: 0
