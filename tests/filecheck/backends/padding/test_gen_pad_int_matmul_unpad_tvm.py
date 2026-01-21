# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad(a, padding=2, name="A_pad")
    p2 = O.pad(b, padding=2, name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad(m_pad, padding=2, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_int_matmul_unpad_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: pad_matmul_unpad
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 14x14xfloat32
# CHECK-NEXT:    - %1 : 14x14xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %5 : 14x14xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad(%0, padding=(2, 2), constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %3: pad(%1, padding=(2, 2), constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %4: matmul(%2, %3) {name = 'matmul_padded'} : [18x18xfloat32, 18x18xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %5: unpad(%4, padding=(2, 2)) {name = 'C'} : [18x18xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((14, 14), "float32"), _1: T.Buffer((14, 14), "float32"), C: T.Buffer((14, 14), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          A_pad = T.allocate([252], "float32", "global")
# CHECK-NEXT:          B_pad = T.allocate([252], "float32", "global")
# CHECK-NEXT:          matmul_padded = T.allocate([196], "float32", "global")
# CHECK-NEXT:          A_pad_1 = T.Buffer((252,), data=A_pad)
# CHECK-NEXT:          for i0, i1 in T.grid(14, 18):
# CHECK-NEXT:              _0_1 = T.Buffer((196,), data=_0.data)
# CHECK-NEXT:              A_pad_1[i0 * 18 + i1] = T.if_then_else(2 <= i1 and i1 < 16, _0_1[i0 * 14 + i1 - 2], T.float32(0.0))
# CHECK-NEXT:          B_pad_1 = T.Buffer((252,), data=B_pad)
# CHECK-NEXT:          for i0, i1 in T.grid(18, 14):
# CHECK-NEXT:              cse_var_1: T.int32 = i0 * 14 + i1
# CHECK-NEXT:              _1_1 = T.Buffer((196,), data=_1.data)
# CHECK-NEXT:              B_pad_1[cse_var_1] = T.if_then_else(2 <= i0 and i0 < 16, _1_1[cse_var_1 - 28], T.float32(0.0))
# CHECK-NEXT:          matmul_padded_1 = T.Buffer((196,), data=matmul_padded)
# CHECK-NEXT:          for i, j in T.grid(14, 14):
# CHECK-NEXT:              matmul_padded_1[i * 14 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(18):
# CHECK-NEXT:                  cse_var_2: T.int32 = i * 14 + j
# CHECK-NEXT:                  matmul_padded_1[cse_var_2] = matmul_padded_1[cse_var_2] + A_pad_1[i * 18 + k] * B_pad_1[k * 14 + j]
# CHECK-NEXT:          for i0, i1 in T.grid(14, 14):
# CHECK-NEXT:              cse_var_3: T.int32 = i0 * 14 + i1
# CHECK-NEXT:              C_1 = T.Buffer((196,), data=C.data)
# CHECK-NEXT:              C_1[cse_var_3] = matmul_padded_1[cse_var_3]
# CHECK-NEXT:  O = obj['matmul_padded']
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  sch[O].reorder(i, j, k)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((14, 14), "float32"), _1: T.Buffer((14, 14), "float32"), C: T.Buffer((14, 14), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          A_pad = T.allocate([252], "float32", "global")
# CHECK-NEXT:          B_pad = T.allocate([252], "float32", "global")
# CHECK-NEXT:          matmul_padded = T.allocate([196], "float32", "global")
# CHECK-NEXT:          A_pad_1 = T.Buffer((252,), data=A_pad)
# CHECK-NEXT:          for i0, i1 in T.grid(14, 18):
# CHECK-NEXT:              _0_1 = T.Buffer((196,), data=_0.data)
# CHECK-NEXT:              A_pad_1[i0 * 18 + i1] = T.if_then_else(2 <= i1 and i1 < 16, _0_1[i0 * 14 + i1 - 2], T.float32(0.0))
# CHECK-NEXT:          B_pad_1 = T.Buffer((252,), data=B_pad)
# CHECK-NEXT:          for i0, i1 in T.grid(18, 14):
# CHECK-NEXT:              cse_var_1: T.int32 = i0 * 14 + i1
# CHECK-NEXT:              _1_1 = T.Buffer((196,), data=_1.data)
# CHECK-NEXT:              B_pad_1[cse_var_1] = T.if_then_else(2 <= i0 and i0 < 16, _1_1[cse_var_1 - 28], T.float32(0.0))
# CHECK-NEXT:          matmul_padded_1 = T.Buffer((196,), data=matmul_padded)
# CHECK-NEXT:          for i, j in T.grid(14, 14):
# CHECK-NEXT:              matmul_padded_1[i * 14 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(18):
# CHECK-NEXT:                  cse_var_2: T.int32 = i * 14 + j
# CHECK-NEXT:                  matmul_padded_1[cse_var_2] = matmul_padded_1[cse_var_2] + A_pad_1[i * 18 + k] * B_pad_1[k * 14 + j]
# CHECK-NEXT:          for i0, i1 in T.grid(14, 14):
# CHECK-NEXT:              cse_var_3: T.int32 = i0 * 14 + i1
# CHECK-NEXT:              C_1 = T.Buffer((196,), data=C.data)
# CHECK-NEXT:              C_1[cse_var_3] = matmul_padded_1[cse_var_3]
# CHECK-NEXT:  CODE: 0
