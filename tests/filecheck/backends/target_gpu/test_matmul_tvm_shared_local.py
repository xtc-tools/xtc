# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm, tvm_cuda

import xtc.graphs.xtc.op as O
from xtc.backends.tvm.TVMBackend import TVMBackend as Backend

from xtc.runtimes.accelerator.gpu import GPUDevice

# Create device
gpu = GPUDevice()
I, J, K, dtype = 1024, 1024, 1024, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 64, "i2": 8})
sch.tile("j", {"j1": 128, "j2": 8})
sch.tile("k", {"k1": 16})
sch.pack_at("k", 0, "shared")
sch.pack_at("k", 1, "shared")
sch.buffer_at("k1", "local")
sch.gpu_block(["j", "i"])
sch.gpu_thread(["j1", "i1"])
sch.interchange(["i", "j","k", "i1", "j1", "k1", "i2", "j2"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="gpu_matmul_tvm_shared_local",
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
# CHECK-NEXT:    - %0 : 1024x1024xfloat32
# CHECK-NEXT:    - %1 : 1024x1024xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1024x1024xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [1024x1024xfloat32, 1024x1024xfloat32] -> [1024x1024xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((1024, 1024), "float32"), _1: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k in T.grid(1024, 1024, 1024):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  I_R0 = sch.cache_read(O, 0, "shared")
# CHECK-NEXT:  I_R1 = sch.cache_read(O, 1, "shared")
# CHECK-NEXT:  O_W0 = sch.cache_write(O, 0, "local")
# CHECK-NEXT:  i, i1, i2, = sch.split(i, factors=[None, 8, 8])
# CHECK-NEXT:  j, j1, j2, = sch.split(j, factors=[None, 16, 8])
# CHECK-NEXT:  k, k1, = sch.split(k, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(i, j, k, i1, j1, k1, i2, j2)
# CHECK-NEXT:  sch.bind(j, 'blockIdx.x')
# CHECK-NEXT:  sch.bind(i, 'blockIdx.y')
# CHECK-NEXT:  sch.bind(j1, 'threadIdx.x')
# CHECK-NEXT:  sch.bind(i1, 'threadIdx.y')
# CHECK-NEXT:  sch.reverse_compute_at(O_W0, k1)
# CHECK-NEXT:  sch.compute_at(I_R0, k)
# CHECK-NEXT:  sch.compute_at(I_R1, k)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((1024, 1024), "float32"), _1: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          _0_shared = T.sblock_alloc_buffer((1024, 1024), scope="shared")
# CHECK-NEXT:          _1_shared = T.sblock_alloc_buffer((1024, 1024), scope="shared")
# CHECK-NEXT:          C_local = T.sblock_alloc_buffer((1024, 1024), scope="local")
# CHECK-NEXT:          for i_0 in T.thread_binding(16, thread="blockIdx.y"):
# CHECK-NEXT:              for j_0 in T.thread_binding(8, thread="blockIdx.x"):
# CHECK-NEXT:                  for i_1_init in T.thread_binding(8, thread="threadIdx.y"):
# CHECK-NEXT:                      for j_1_init in T.thread_binding(16, thread="threadIdx.x"):
# CHECK-NEXT:                          for i_2_init, j_2_init in T.grid(8, 8):
# CHECK-NEXT:                              with T.sblock("C_init"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(1024, i_0 * 64 + i_1_init * 8 + i_2_init)
# CHECK-NEXT:                                  v_j = T.axis.spatial(1024, j_0 * 128 + j_1_init * 8 + j_2_init)
# CHECK-NEXT:                                  T.reads()
# CHECK-NEXT:                                  T.writes(C_local[v_i, v_j])
# CHECK-NEXT:                                  C_local[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  for k_0 in range(64):
# CHECK-NEXT:                      for ax0, ax1 in T.grid(64, 16):
# CHECK-NEXT:                          with T.sblock("_0_shared"):
# CHECK-NEXT:                              v0 = T.axis.spatial(1024, i_0 * 64 + ax0)
# CHECK-NEXT:                              v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
# CHECK-NEXT:                              T.reads(_0[v0, v1])
# CHECK-NEXT:                              T.writes(_0_shared[v0, v1])
# CHECK-NEXT:                              _0_shared[v0, v1] = _0[v0, v1]
# CHECK-NEXT:                      for ax0, ax1 in T.grid(16, 128):
# CHECK-NEXT:                          with T.sblock("_1_shared"):
# CHECK-NEXT:                              v0 = T.axis.spatial(1024, k_0 * 16 + ax0)
# CHECK-NEXT:                              v1 = T.axis.spatial(1024, j_0 * 128 + ax1)
# CHECK-NEXT:                              T.reads(_1[v0, v1])
# CHECK-NEXT:                              T.writes(_1_shared[v0, v1])
# CHECK-NEXT:                              _1_shared[v0, v1] = _1[v0, v1]
# CHECK-NEXT:                      for i_1 in T.thread_binding(8, thread="threadIdx.y"):
# CHECK-NEXT:                          for j_1 in T.thread_binding(16, thread="threadIdx.x"):
# CHECK-NEXT:                              for k_1 in range(16):
# CHECK-NEXT:                                  for i_2, j_2 in T.grid(8, 8):
# CHECK-NEXT:                                      with T.sblock("C_update"):
# CHECK-NEXT:                                          v_i = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
# CHECK-NEXT:                                          v_j = T.axis.spatial(1024, j_0 * 128 + j_1 * 8 + j_2)
# CHECK-NEXT:                                          v_k = T.axis.reduce(1024, k_0 * 16 + k_1)
# CHECK-NEXT:                                          T.reads(C_local[v_i, v_j], _0_shared[v_i, v_k], _1_shared[v_k, v_j])
# CHECK-NEXT:                                          T.writes(C_local[v_i, v_j])
# CHECK-NEXT:                                          C_local[v_i, v_j] = C_local[v_i, v_j] + _0_shared[v_i, v_k] * _1_shared[v_k, v_j]
# CHECK-NEXT:                                  for ax0, ax1 in T.grid(8, 8):
# CHECK-NEXT:                                      with T.sblock("C_local"):
# CHECK-NEXT:                                          v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
# CHECK-NEXT:                                          v1 = T.axis.spatial(1024, j_0 * 128 + j_1 * 8 + ax1)
# CHECK-NEXT:                                          T.reads(C_local[v0, v1])
# CHECK-NEXT:                                          T.writes(C[v0, v1])
# CHECK-NEXT:                                          C[v0, v1] = C_local[v0, v1]
# CHECK-NEXT:  CODE: 0
