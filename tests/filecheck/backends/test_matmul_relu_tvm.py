# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul_relu") as gb:
    m = O.matmul(a, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler(nodes=["matmul"])
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_mlir",
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
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), T_reshape: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          matmul = T.allocate([128], "float32", "global")
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              matmul_1 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:              matmul_1[i * 32 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(512):
# CHECK-NEXT:                  cse_var_1: T.int32 = i * 32 + j
# CHECK-NEXT:                  _0_1 = T.Buffer((2048,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((16384,), data=_1.data)
# CHECK-NEXT:                  matmul_1[cse_var_1] = matmul_1[cse_var_1] + _0_1[i * 512 + k] * _1_1[k * 32 + j]
# CHECK-NEXT:          matmul_1 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:          for i in range(128):
# CHECK-NEXT:              matmul_2 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:              matmul_1[i] = T.max(T.float32(0.0), matmul_2[i])
# CHECK-NEXT:          for ax0, ax1 in T.grid(4, 32):
# CHECK-NEXT:              cse_var_2: T.int32 = ax0 * 32 + ax1
# CHECK-NEXT:              T_reshape_1 = T.Buffer((128,), data=T_reshape.data)
# CHECK-NEXT:              T_reshape_1[cse_var_2] = matmul_1[cse_var_2]
# CHECK-NEXT:  O = obj['matmul']
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  i, i1 = sch[O].split(i, factor=2)
# CHECK-NEXT:  j, j1 = sch[O].split(j, factor=16)
# CHECK-NEXT:  sch[O].reorder(k, i, j, i1, j1)
# CHECK-NEXT:  sch[O].unroll(i1)
# CHECK-NEXT:  sch[O].vectorize(j1)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), T_reshape: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          matmul = T.allocate([128], "float32", "global")
# CHECK-NEXT:          matmul_1 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:          for i_outer_init, j_outer_init in T.grid(2, 2):
# CHECK-NEXT:              cse_var_1: T.int32 = i_outer_init * 64 + j_outer_init * 16
# CHECK-NEXT:              matmul_1[cse_var_1:cse_var_1 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              matmul_1[cse_var_1 + 32:cse_var_1 + 32 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:          for k, i_outer, j_outer in T.grid(512, 2, 2):
# CHECK-NEXT:              cse_var_6: T.int32 = j_outer * 16
# CHECK-NEXT:              cse_var_5: T.int32 = i_outer * 1024 + k
# CHECK-NEXT:              cse_var_4: T.int32 = k * 32 + cse_var_6
# CHECK-NEXT:              cse_var_3: T.int32 = i_outer * 64 + cse_var_6
# CHECK-NEXT:              cse_var_2: T.int32 = cse_var_3 + 32
# CHECK-NEXT:              _0_1 = T.Buffer((2048,), data=_0.data)
# CHECK-NEXT:              _1_1 = T.Buffer((16384,), data=_1.data)
# CHECK-NEXT:              matmul_1[cse_var_3:cse_var_3 + 16] = matmul_1[cse_var_3:cse_var_3 + 16] + T.Broadcast(_0_1[cse_var_5], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:              matmul_1[cse_var_2:cse_var_2 + 16] = matmul_1[cse_var_2:cse_var_2 + 16] + T.Broadcast(_0_1[cse_var_5 + 512], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:          matmul_2 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:          for i in range(128):
# CHECK-NEXT:              matmul_3 = T.Buffer((128,), data=matmul)
# CHECK-NEXT:              matmul_2[i] = T.max(T.float32(0.0), matmul_3[i])
# CHECK-NEXT:          for ax0, ax1 in T.grid(4, 32):
# CHECK-NEXT:              cse_var_7: T.int32 = ax0 * 32 + ax1
# CHECK-NEXT:              T_reshape_1 = T.Buffer((128,), data=T_reshape.data)
# CHECK-NEXT:              T_reshape_1[cse_var_7] = matmul_2[cse_var_7]
# CHECK-NEXT:  CODE: 0
