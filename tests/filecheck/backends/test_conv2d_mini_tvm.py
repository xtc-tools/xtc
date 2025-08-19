# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="conv2d_nhwc_mini") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_mini_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'O'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for h, w, f in T.grid(8, 8, 16):
# CHECK-NEXT:              O_1 = T.Buffer((1024,), data=O.data)
# CHECK-NEXT:              O_1[h * 128 + w * 16 + f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c in T.grid(3, 3, 3):
# CHECK-NEXT:                  cse_var_1: T.int32 = h * 128 + w * 16 + f
# CHECK-NEXT:                  _0_1 = T.Buffer((300,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((432,), data=_1.data)
# CHECK-NEXT:                  O_1[cse_var_1] = O_1[cse_var_1] + _0_1[h * 30 + r * 30 + w * 3 + s * 3 + c] * _1_1[r * 144 + s * 48 + c * 16 + f]
# CHECK-NEXT:  O = obj['O']
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, r, s, c)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for h, w, f in T.grid(8, 8, 16):
# CHECK-NEXT:              O_1 = T.Buffer((1024,), data=O.data)
# CHECK-NEXT:              O_1[h * 128 + w * 16 + f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c in T.grid(3, 3, 3):
# CHECK-NEXT:                  cse_var_1: T.int32 = h * 128 + w * 16 + f
# CHECK-NEXT:                  _0_1 = T.Buffer((300,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((432,), data=_1.data)
# CHECK-NEXT:                  O_1[cse_var_1] = O_1[cse_var_1] + _0_1[h * 30 + r * 30 + w * 3 + s * 3 + c] * _1_1[r * 144 + s * 48 + c * 16 + f]
# CHECK-NEXT:  CODE: 0
