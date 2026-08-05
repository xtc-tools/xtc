# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul_relu") as gb:
    m = O.matmul(a, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, tir_schedule=True)

sch = impl.get_scheduler(default_node="matmul")
descript_scheduler(
    scheduler = sch,
    node_name = "matmul",
    abstract_dims = ["i","j","k"],
    spec = {
        "i": {},
        "j": {},
        "i#2": {},
        "j#16": {"fuse_consumer":True},
        "k": {},
    }
)
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_descript_fuse_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'matmul'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [4x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_relu(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), T_reshape: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          matmul = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          T_reshape_1 = T.sblock_alloc_buffer((128,))
# CHECK-NEXT:          relu = T.sblock_alloc_buffer((128,))
# CHECK-NEXT:          for i, j, k in T.grid(4, 32, 512):
# CHECK-NEXT:              with T.sblock("matmul"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(matmul[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      matmul[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  matmul[v_i, v_j] = matmul[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:          for ax0 in range(128):
# CHECK-NEXT:              with T.sblock("T_reshape"):
# CHECK-NEXT:                  v_ax0 = T.axis.spatial(128, ax0)
# CHECK-NEXT:                  T.reads(matmul[v_ax0 % 128 // 32, v_ax0 % 32])
# CHECK-NEXT:                  T.writes(T_reshape_1[v_ax0])
# CHECK-NEXT:                  T_reshape_1[v_ax0] = matmul[v_ax0 % 128 // 32, v_ax0 % 32]
# CHECK-NEXT:          for i in range(128):
# CHECK-NEXT:              with T.sblock("relu"):
# CHECK-NEXT:                  v_i = T.axis.spatial(128, i)
# CHECK-NEXT:                  T.reads(T_reshape_1[v_i])
# CHECK-NEXT:                  T.writes(relu[v_i])
# CHECK-NEXT:                  relu[v_i] = T.max(T.float32(0.0), T_reshape_1[v_i])
# CHECK-NEXT:          for ax0, ax1 in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("T_reshape_1"):
# CHECK-NEXT:                  v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
# CHECK-NEXT:                  T.reads(relu[(v_ax0 * 32 + v_ax1) % 128])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0, v_ax1])
# CHECK-NEXT:                  T_reshape[v_ax0, v_ax1] = relu[(v_ax0 * 32 + v_ax1) % 128]
# CHECK-NEXT:  O = sch.get_sblock("matmul")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  O_F0 = sch.get_consumers(O)[0]
# CHECK-NEXT:  i, i0, = sch.split(i, factors=[None, 2])
# CHECK-NEXT:  j, j0, = sch.split(j, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(i, j, i0, j0, k)
# CHECK-NEXT:  sch.reverse_compute_at(O_F0, j0)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_relu(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), T_reshape: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          matmul = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          T_reshape_1 = T.sblock_alloc_buffer((128,))
# CHECK-NEXT:          relu = T.sblock_alloc_buffer((128,))
# CHECK-NEXT:          for i_0, j_0, i_1, j_1 in T.grid(2, 2, 2, 16):
# CHECK-NEXT:              for k in range(512):
# CHECK-NEXT:                  with T.sblock("matmul"):
# CHECK-NEXT:                      v_i = T.axis.spatial(4, i_0 * 2 + i_1)
# CHECK-NEXT:                      v_j = T.axis.spatial(32, j_0 * 16 + j_1)
# CHECK-NEXT:                      v_k = T.axis.reduce(512, k)
# CHECK-NEXT:                      T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                      T.writes(matmul[v_i, v_j])
# CHECK-NEXT:                      with T.init():
# CHECK-NEXT:                          matmul[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                      matmul[v_i, v_j] = matmul[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:              with T.sblock("T_reshape"):
# CHECK-NEXT:                  v_ax0 = T.axis.spatial(128, i_0 * 64 + i_1 * 32 + j_0 * 16 + j_1)
# CHECK-NEXT:                  T.reads(matmul[v_ax0 % 128 // 32, v_ax0 % 32])
# CHECK-NEXT:                  T.writes(T_reshape_1[v_ax0])
# CHECK-NEXT:                  T_reshape_1[v_ax0] = matmul[v_ax0 % 128 // 32, v_ax0 % 32]
# CHECK-NEXT:          for i in range(128):
# CHECK-NEXT:              with T.sblock("relu"):
# CHECK-NEXT:                  v_i = T.axis.spatial(128, i)
# CHECK-NEXT:                  T.reads(T_reshape_1[v_i])
# CHECK-NEXT:                  T.writes(relu[v_i])
# CHECK-NEXT:                  relu[v_i] = T.max(T.float32(0.0), T_reshape_1[v_i])
# CHECK-NEXT:          for ax0, ax1 in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("T_reshape_1"):
# CHECK-NEXT:                  v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
# CHECK-NEXT:                  T.reads(relu[(v_ax0 * 32 + v_ax1) % 128])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0, v_ax1])
# CHECK-NEXT:                  T_reshape[v_ax0, v_ax1] = relu[(v_ax0 * 32 + v_ax1) % 128]
# CHECK-NEXT:  CODE: 0
