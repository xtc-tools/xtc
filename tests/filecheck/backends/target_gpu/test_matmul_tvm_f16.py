# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
# REQUIRES: tvm_cuda

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 256, 256, 256, "float16"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="C")
sch.tile("i", {"i1": 64, "i2": 8})
sch.tile("j", {"j1": 128, "j2": 8})
sch.tile("k", {"k1": 32, "k2": 8})
sch.interchange(["i", "j", "i1", "j1", "k", "k1", "k2", "i2", "j2"])
sch.gpu_block(["i"])
sch.gpu_thread(["i1"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_gpu_f16_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
    target=gpu,
)
module = comp.compile(sched)

evaluator = module.get_evaluator(validate=True)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 256x256xfloat16
# CHECK-NEXT:    - %1 : 256x256xfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 256x256xfloat16
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [256x256xfloat16, 256x256xfloat16] -> [256x256xfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((256, 256), "float16"), _1: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k in T.grid(256, 256, 256):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float16(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  i, i1, i2, = sch.split(i, factors=[None, 8, 8])
# CHECK-NEXT:  j, j1, j2, = sch.split(j, factors=[None, 16, 8])
# CHECK-NEXT:  k, k1, k2, = sch.split(k, factors=[None, 4, 8])
# CHECK-NEXT:  sch.reorder(i, j, i1, j1, k, k1, k2, i2, j2)
# CHECK-NEXT:  sch.bind(i, 'blockIdx.x')
# CHECK-NEXT:  sch.bind(i1, 'threadIdx.x')
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((256, 256), "float16"), _1: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0 in T.thread_binding(4, thread="blockIdx.x"):
# CHECK-NEXT:              for j_0 in range(2):
# CHECK-NEXT:                  for i_1 in T.thread_binding(8, thread="threadIdx.x"):
# CHECK-NEXT:                      for j_1 in range(16):
# CHECK-NEXT:                          for i_2_init, j_2_init in T.grid(8, 8):
# CHECK-NEXT:                              with T.sblock("C_init"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(256, i_0 * 64 + i_1 * 8 + i_2_init)
# CHECK-NEXT:                                  v_j = T.axis.spatial(256, j_0 * 128 + j_1 * 8 + j_2_init)
# CHECK-NEXT:                                  T.reads()
# CHECK-NEXT:                                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                                  C[v_i, v_j] = T.float16(0.0)
# CHECK-NEXT:                          for k_0, k_1, k_2, i_2, j_2 in T.grid(8, 4, 8, 8, 8):
# CHECK-NEXT:                              with T.sblock("C_update"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(256, i_0 * 64 + i_1 * 8 + i_2)
# CHECK-NEXT:                                  v_j = T.axis.spatial(256, j_0 * 128 + j_1 * 8 + j_2)
# CHECK-NEXT:                                  v_k = T.axis.reduce(256, k_0 * 32 + k_1 * 8 + k_2)
# CHECK-NEXT:                                  T.reads(C[v_i, v_j], _0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  CODE: 0
