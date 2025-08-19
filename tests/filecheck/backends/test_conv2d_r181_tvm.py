# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend

# Resnet18_01 size
N, H, W, F, R, S, C, SH, SW, dtype = 1, 224, 224, 64, 7, 7, 3, 2, 2, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_nhwc_r181
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:    - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x112x112x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'O'} : [1x230x230x3xfloat32, 7x7x3x64xfloat32] -> [1x112x112x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((1, 230, 230, 3), "float32"), _1: T.Buffer((7, 7, 3, 64), "float32"), O: T.Buffer((1, 112, 112, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for h, w, f in T.grid(112, 112, 64):
# CHECK-NEXT:              O_1 = T.Buffer((802816,), data=O.data)
# CHECK-NEXT:              O_1[h * 7168 + w * 64 + f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c in T.grid(7, 7, 3):
# CHECK-NEXT:                  cse_var_1: T.int32 = h * 7168 + w * 64 + f
# CHECK-NEXT:                  _0_1 = T.Buffer((158700,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((9408,), data=_1.data)
# CHECK-NEXT:                  O_1[cse_var_1] = O_1[cse_var_1] + _0_1[h * 1380 + r * 690 + w * 6 + s * 3 + c] * _1_1[r * 1344 + s * 192 + c * 64 + f]
# CHECK-NEXT:  O = obj['O']
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=4)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=16)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, r, s, c, w1, f1)
# CHECK-NEXT:  sch[O].unroll(w1)
# CHECK-NEXT:  sch[O].unroll(c)
# CHECK-NEXT:  sch[O].vectorize(f1)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((1, 230, 230, 3), "float32"), _1: T.Buffer((7, 7, 3, 64), "float32"), O: T.Buffer((1, 112, 112, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for h, w_outer, f_outer in T.grid(112, 28, 4):
# CHECK-NEXT:              cse_var_1: T.int32 = h * 7168 + w_outer * 256 + f_outer * 16
# CHECK-NEXT:              O_1 = T.Buffer((802816,), data=O.data)
# CHECK-NEXT:              O_1[cse_var_1:cse_var_1 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              O_1[cse_var_1 + 64:cse_var_1 + 64 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              O_1[cse_var_1 + 128:cse_var_1 + 128 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              O_1[cse_var_1 + 192:cse_var_1 + 192 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              for r, s in T.grid(7, 7):
# CHECK-NEXT:                  cse_var_8: T.int32 = cse_var_1 + 64
# CHECK-NEXT:                  cse_var_7: T.int32 = cse_var_1 + 192
# CHECK-NEXT:                  cse_var_6: T.int32 = cse_var_1 + 128
# CHECK-NEXT:                  cse_var_5: T.int32 = r * 1344 + s * 192 + f_outer * 16
# CHECK-NEXT:                  cse_var_4: T.int32 = cse_var_5 + 64
# CHECK-NEXT:                  cse_var_3: T.int32 = cse_var_5 + 128
# CHECK-NEXT:                  cse_var_2: T.int32 = h * 1380 + r * 690 + w_outer * 24 + s * 3
# CHECK-NEXT:                  _0_1 = T.Buffer((158700,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((9408,), data=_1.data)
# CHECK-NEXT:                  O_1[cse_var_1:cse_var_1 + 16] = O_1[cse_var_1:cse_var_1 + 16] + T.Broadcast(_0_1[cse_var_2], 16) * _1_1[cse_var_5:cse_var_5 + 16]
# CHECK-NEXT:                  O_1[cse_var_8:cse_var_8 + 16] = O_1[cse_var_8:cse_var_8 + 16] + T.Broadcast(_0_1[cse_var_2 + 6], 16) * _1_1[cse_var_5:cse_var_5 + 16]
# CHECK-NEXT:                  O_1[cse_var_6:cse_var_6 + 16] = O_1[cse_var_6:cse_var_6 + 16] + T.Broadcast(_0_1[cse_var_2 + 12], 16) * _1_1[cse_var_5:cse_var_5 + 16]
# CHECK-NEXT:                  O_1[cse_var_7:cse_var_7 + 16] = O_1[cse_var_7:cse_var_7 + 16] + T.Broadcast(_0_1[cse_var_2 + 18], 16) * _1_1[cse_var_5:cse_var_5 + 16]
# CHECK-NEXT:                  O_1[cse_var_1:cse_var_1 + 16] = O_1[cse_var_1:cse_var_1 + 16] + T.Broadcast(_0_1[cse_var_2 + 1], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:                  O_1[cse_var_8:cse_var_8 + 16] = O_1[cse_var_8:cse_var_8 + 16] + T.Broadcast(_0_1[cse_var_2 + 7], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:                  O_1[cse_var_6:cse_var_6 + 16] = O_1[cse_var_6:cse_var_6 + 16] + T.Broadcast(_0_1[cse_var_2 + 13], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:                  O_1[cse_var_7:cse_var_7 + 16] = O_1[cse_var_7:cse_var_7 + 16] + T.Broadcast(_0_1[cse_var_2 + 19], 16) * _1_1[cse_var_4:cse_var_4 + 16]
# CHECK-NEXT:                  O_1[cse_var_1:cse_var_1 + 16] = O_1[cse_var_1:cse_var_1 + 16] + T.Broadcast(_0_1[cse_var_2 + 2], 16) * _1_1[cse_var_3:cse_var_3 + 16]
# CHECK-NEXT:                  O_1[cse_var_8:cse_var_8 + 16] = O_1[cse_var_8:cse_var_8 + 16] + T.Broadcast(_0_1[cse_var_2 + 8], 16) * _1_1[cse_var_3:cse_var_3 + 16]
# CHECK-NEXT:                  O_1[cse_var_6:cse_var_6 + 16] = O_1[cse_var_6:cse_var_6 + 16] + T.Broadcast(_0_1[cse_var_2 + 14], 16) * _1_1[cse_var_3:cse_var_3 + 16]
# CHECK-NEXT:                  O_1[cse_var_7:cse_var_7 + 16] = O_1[cse_var_7:cse_var_7 + 16] + T.Broadcast(_0_1[cse_var_2 + 20], 16) * _1_1[cse_var_3:cse_var_3 + 16]
# CHECK-NEXT:  CODE: 0
