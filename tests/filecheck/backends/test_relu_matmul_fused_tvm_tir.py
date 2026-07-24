# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 64, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    p = O.relu(a, name="relu")
    O.matmul(p, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, tir_schedule=True)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 8, "i2": 4})
sch.tile("j", {"j1": 32, "j2": 16})
sch.tile("k", {"k1": 16})
sch.interchange(["i", "j", "k", "i1", "j1", "k1", "i2", "j2"])
sch.buffer_at("j")
sch.pack_at("k", 1, pad=True)
sch.fuse_producer_at("k", 0)
sch.vectorize(["j2"])
sch.unroll({"i2": 4})
sch.parallelize(["i", "j"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="relu_matmul_fused_tvm_tir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")


# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 64x64xfloat32
# CHECK-NEXT:    - %1 : 64x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 64x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: relu(%0) {name = 'relu'} : [64x64xfloat32] -> [64x64xfloat32]
# CHECK-NEXT:    - %3: matmul(%2, %1) {name = 'C'} : [64x64xfloat32, 64x64xfloat32] -> [64x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def matmul(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"tir.noalias": T.bool(True)})
# CHECK-NEXT:          # with T.block("root"):
# CHECK-NEXT:          T_reshape = T.alloc_buffer((4096,))
# CHECK-NEXT:          relu = T.alloc_buffer((4096,))
# CHECK-NEXT:          T_reshape_1 = T.alloc_buffer((64, 64))
# CHECK-NEXT:          for ax0 in range(4096):
# CHECK-NEXT:              with T.block("T_reshape"):
# CHECK-NEXT:                  v_ax0 = T.axis.spatial(4096, ax0)
# CHECK-NEXT:                  T.reads(_0[v_ax0 % 4096 // 64, v_ax0 % 64])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0])
# CHECK-NEXT:                  T_reshape[v_ax0] = _0[v_ax0 % 4096 // 64, v_ax0 % 64]
# CHECK-NEXT:          for i in range(4096):
# CHECK-NEXT:              with T.block("relu"):
# CHECK-NEXT:                  v_i = T.axis.spatial(4096, i)
# CHECK-NEXT:                  T.reads(T_reshape[v_i])
# CHECK-NEXT:                  T.writes(relu[v_i])
# CHECK-NEXT:                  relu[v_i] = T.max(T.float32(0.0), T_reshape[v_i])
# CHECK-NEXT:          for ax0, ax1 in T.grid(64, 64):
# CHECK-NEXT:              with T.block("T_reshape_1"):
# CHECK-NEXT:                  v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
# CHECK-NEXT:                  T.reads(relu[(v_ax0 * 64 + v_ax1) % 4096])
# CHECK-NEXT:                  T.writes(T_reshape_1[v_ax0, v_ax1])
# CHECK-NEXT:                  T_reshape_1[v_ax0, v_ax1] = relu[(v_ax0 * 64 + v_ax1) % 4096]
# CHECK-NEXT:          for i, j, k in T.grid(64, 64, 64):
# CHECK-NEXT:              with T.block("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(T_reshape_1[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + T_reshape_1[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  O = sch.get_block("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  I_R1 = sch.cache_read(O, 1, "global")
# CHECK-NEXT:  O_W0 = sch.cache_write(O, 0, "global")
# CHECK-NEXT:  I_F0 = sch.get_producers(O)[0]
# CHECK-NEXT:  i, i1, i2, = sch.split(i, factors=[None, 2, 4])
# CHECK-NEXT:  j, j1, j2, = sch.split(j, factors=[None, 2, 16])
# CHECK-NEXT:  k, k1, = sch.split(k, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(i, j, k, i1, j1, k1, i2, j2)
# CHECK-NEXT:  sch.reverse_compute_at(O_W0, j)
# CHECK-NEXT:  sch.compute_at(I_R1, k)
# CHECK-NEXT:  sch.compute_at(I_F0, k)
# CHECK-NEXT:  sch.unroll(i2)
# CHECK-NEXT:  sch.vectorize(j2)
# CHECK-NEXT:  j = sch.fuse(i, j)
# CHECK-NEXT:  sch.parallel(j)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def matmul(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"tir.noalias": T.bool(True)})
# CHECK-NEXT:          # with T.block("root"):
# CHECK-NEXT:          T_reshape = T.alloc_buffer((4096,))
# CHECK-NEXT:          relu = T.alloc_buffer((4096,))
# CHECK-NEXT:          T_reshape_1 = T.alloc_buffer((64, 64))
# CHECK-NEXT:          _1_global = T.alloc_buffer((64, 64))
# CHECK-NEXT:          C_global = T.alloc_buffer((64, 64))
# CHECK-NEXT:          for ax0 in range(4096):
# CHECK-NEXT:              with T.block("T_reshape"):
# CHECK-NEXT:                  v_ax0 = T.axis.spatial(4096, ax0)
# CHECK-NEXT:                  T.reads(_0[v_ax0 % 4096 // 64, v_ax0 % 64])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0])
# CHECK-NEXT:                  T_reshape[v_ax0] = _0[v_ax0 % 4096 // 64, v_ax0 % 64]
# CHECK-NEXT:          for i in range(4096):
# CHECK-NEXT:              with T.block("relu"):
# CHECK-NEXT:                  v_i = T.axis.spatial(4096, i)
# CHECK-NEXT:                  T.reads(T_reshape[v_i])
# CHECK-NEXT:                  T.writes(relu[v_i])
# CHECK-NEXT:                  relu[v_i] = T.max(T.float32(0.0), T_reshape[v_i])
# CHECK-NEXT:          for i_0_j_0_fused in T.parallel(16):
# CHECK-NEXT:              for k_0 in range(4):
# CHECK-NEXT:                  for ax0, ax1 in T.grid(16, 32):
# CHECK-NEXT:                      with T.block("_1_global"):
# CHECK-NEXT:                          v0 = T.axis.spatial(64, k_0 * 16 + ax0)
# CHECK-NEXT:                          v1 = T.axis.spatial(64, i_0_j_0_fused % 2 * 32 + ax1)
# CHECK-NEXT:                          T.reads(_1[v0, v1])
# CHECK-NEXT:                          T.writes(_1_global[v0, v1])
# CHECK-NEXT:                          _1_global[v0, v1] = _1[v0, v1]
# CHECK-NEXT:                  for ax0, ax1 in T.grid(8, 16):
# CHECK-NEXT:                      with T.block("T_reshape_1"):
# CHECK-NEXT:                          v_ax0 = T.axis.spatial(64, i_0_j_0_fused // 2 * 8 + ax0)
# CHECK-NEXT:                          v_ax1 = T.axis.spatial(64, k_0 * 16 + ax1)
# CHECK-NEXT:                          T.reads(relu[(v_ax0 * 64 + v_ax1) % 4096])
# CHECK-NEXT:                          T.writes(T_reshape_1[v_ax0, v_ax1])
# CHECK-NEXT:                          T_reshape_1[v_ax0, v_ax1] = relu[(v_ax0 * 64 + v_ax1) % 4096]
# CHECK-NEXT:                  for i_1, j_1, k_1 in T.grid(2, 2, 16):
# CHECK-NEXT:                      for i_2 in T.unroll(4):
# CHECK-NEXT:                          for j_2 in T.vectorized(16):
# CHECK-NEXT:                              with T.block("C"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(64, i_0_j_0_fused // 2 * 8 + i_1 * 4 + i_2)
# CHECK-NEXT:                                  v_j = T.axis.spatial(64, i_0_j_0_fused % 2 * 32 + j_1 * 16 + j_2)
# CHECK-NEXT:                                  v_k = T.axis.reduce(64, k_0 * 16 + k_1)
# CHECK-NEXT:                                  T.reads(T_reshape_1[v_i, v_k], _1_global[v_k, v_j])
# CHECK-NEXT:                                  T.writes(C_global[v_i, v_j])
# CHECK-NEXT:                                  with T.init():
# CHECK-NEXT:                                      C_global[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                                  C_global[v_i, v_j] = C_global[v_i, v_j] + T_reshape_1[v_i, v_k] * _1_global[v_k, v_j]
# CHECK-NEXT:              for ax0, ax1 in T.grid(8, 32):
# CHECK-NEXT:                  with T.block("C_global"):
# CHECK-NEXT:                      v0 = T.axis.spatial(64, i_0_j_0_fused // 2 * 8 + ax0)
# CHECK-NEXT:                      v1 = T.axis.spatial(64, i_0_j_0_fused % 2 * 32 + ax1)
# CHECK-NEXT:                      T.reads(C_global[v0, v1])
# CHECK-NEXT:                      T.writes(C[v0, v1])
# CHECK-NEXT:                      C[v0, v1] = C_global[v0, v1]
# CHECK-NEXT:  CODE: 0
