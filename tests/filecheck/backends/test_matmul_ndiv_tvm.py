# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 3}) # non-divisible tile
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})    # non-full/non-divisible unrolled
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_ndiv_tvm",
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
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              C_1 = T.Buffer((128,), data=C.data)
# CHECK-NEXT:              C_1[i * 32 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(512):
# CHECK-NEXT:                  cse_var_1: T.int32 = i * 32 + j
# CHECK-NEXT:                  _0_1 = T.Buffer((2048,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((16384,), data=_1.data)
# CHECK-NEXT:                  C_1[cse_var_1] = C_1[cse_var_1] + _0_1[i * 512 + k] * _1_1[k * 32 + j]
# CHECK-NEXT:  O = obj['C']
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  i, i1 = sch[O].split(i, factor=3)
# CHECK-NEXT:  i1, __u_i1 = sch[O].split(i1, factor=2)
# CHECK-NEXT:  j, j1 = sch[O].split(j, factor=16)
# CHECK-NEXT:  sch[O].reorder(k, i, j, i1, __u_i1, j1)
# CHECK-NEXT:  sch[O].unroll(__u_i1)
# CHECK-NEXT:  sch[O].vectorize(j1)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          C_1 = T.Buffer((128,), data=C.data)
# CHECK-NEXT:          for i_outer_init, j_outer_init, i_inner_outer_init in T.grid(2, 2, 2):
# CHECK-NEXT:              if T.likely(i_outer_init * 3 + i_inner_outer_init * 2 < 4):
# CHECK-NEXT:                  C_1[i_outer_init * 96 + i_inner_outer_init * 64 + j_outer_init * 16:i_outer_init * 96 + i_inner_outer_init * 64 + j_outer_init * 16 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              if T.likely(i_outer_init * 3 + i_inner_outer_init * 2 < 3):
# CHECK-NEXT:                  if T.likely(i_inner_outer_init < 1):
# CHECK-NEXT:                      C_1[i_outer_init * 96 + i_inner_outer_init * 64 + j_outer_init * 16 + 32:i_outer_init * 96 + i_inner_outer_init * 64 + j_outer_init * 16 + 32 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:          for k, i_outer, j_outer, i_inner_outer in T.grid(512, 2, 2, 2):
# CHECK-NEXT:              _0_1 = T.Buffer((2048,), data=_0.data)
# CHECK-NEXT:              _1_1 = T.Buffer((16384,), data=_1.data)
# CHECK-NEXT:              if T.likely(i_outer * 3 + i_inner_outer * 2 < 4):
# CHECK-NEXT:                  cse_var_2: T.int32 = j_outer * 16
# CHECK-NEXT:                  cse_var_1: T.int32 = i_outer * 96 + i_inner_outer * 64 + cse_var_2
# CHECK-NEXT:                  C_1[cse_var_1:cse_var_1 + 16] = C_1[cse_var_1:cse_var_1 + 16] + T.Broadcast(_0_1[i_outer * 1536 + i_inner_outer * 1024 + k], 16) * _1_1[k * 32 + cse_var_2:k * 32 + cse_var_2 + 16]
# CHECK-NEXT:              if T.likely(i_outer * 3 + i_inner_outer * 2 < 3):
# CHECK-NEXT:                  if T.likely(i_inner_outer < 1):
# CHECK-NEXT:                      cse_var_4: T.int32 = j_outer * 16
# CHECK-NEXT:                      cse_var_3: T.int32 = i_outer * 96 + i_inner_outer * 64 + cse_var_4 + 32
# CHECK-NEXT:                      C_1[cse_var_3:cse_var_3 + 16] = C_1[cse_var_3:cse_var_3 + 16] + T.Broadcast(_0_1[i_outer * 1536 + i_inner_outer * 1024 + k + 512], 16) * _1_1[k * 32 + cse_var_4:k * 32 + cse_var_4 + 16]
# CHECK-NEXT:  CODE: 0
