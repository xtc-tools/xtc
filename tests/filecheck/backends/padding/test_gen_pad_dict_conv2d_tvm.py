# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad(a, padding={1: 2, 2: (2, 2)}, name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_dict_conv2d_nhwc_mini_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:    - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32] 
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:     @T.prim_func
# CHECK-NEXT:     def main(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((5, 5, 3, 16), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:         T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:         pad = T.allocate([363], "float32", "global")
# CHECK-NEXT:         pad_1 = T.Buffer((363,), data=pad)
# CHECK-NEXT:         for i1, i2, i3 in T.grid(11, 11, 3):
# CHECK-NEXT:             cse_var_1: T.int32 = i2 * 3
# CHECK-NEXT:             _0_1 = T.Buffer((192,), data=_0.data)
# CHECK-NEXT:             pad_1[i1 * 33 + cse_var_1 + i3] = T.if_then_else(2 <= i1 and i1 < 10 and 2 <= i2 and i2 < 10, _0_1[i1 * 24 + cse_var_1 + i3 - 54], T.float32(0.0))
# CHECK-NEXT:         for h, w, f in T.grid(4, 4, 16):
# CHECK-NEXT:             conv_1 = T.Buffer((256,), data=conv.data)
# CHECK-NEXT:             conv_1[h * 64 + w * 16 + f] = T.float32(0.0)
# CHECK-NEXT:             for r, s, c in T.grid(5, 5, 3):
# CHECK-NEXT:                 cse_var_2: T.int32 = h * 64 + w * 16 + f
# CHECK-NEXT:                 _1_1 = T.Buffer((1200,), data=_1.data)
# CHECK-NEXT:                 conv_1[cse_var_2] = conv_1[cse_var_2] + pad_1[h * 66 + r * 33 + w * 6 + s * 3 + c] * _1_1[r * 240 + s * 48 + c * 16 + f]
# CHECK-NEXT:  O = obj['conv']
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
# CHECK-NEXT:      def main(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((5, 5, 3, 16), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          pad = T.allocate([363], "float32", "global")
# CHECK-NEXT:          pad_1 = T.Buffer((363,), data=pad)
# CHECK-NEXT:          for i1, i2, i3 in T.grid(11, 11, 3):
# CHECK-NEXT:              cse_var_1: T.int32 = i2 * 3
# CHECK-NEXT:              _0_1 = T.Buffer((192,), data=_0.data)
# CHECK-NEXT:              pad_1[i1 * 33 + cse_var_1 + i3] = T.if_then_else(2 <= i1 and i1 < 10 and 2 <= i2 and i2 < 10, _0_1[i1 * 24 + cse_var_1 + i3 - 54], T.float32(0.0))
# CHECK-NEXT:          for h, w, f in T.grid(4, 4, 16):
# CHECK-NEXT:              conv_1 = T.Buffer((256,), data=conv.data)
# CHECK-NEXT:              conv_1[h * 64 + w * 16 + f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c in T.grid(5, 5, 3):
# CHECK-NEXT:                  cse_var_2: T.int32 = h * 64 + w * 16 + f
# CHECK-NEXT:                  _1_1 = T.Buffer((1200,), data=_1.data)
# CHECK-NEXT:                  conv_1[cse_var_2] = conv_1[cse_var_2] + pad_1[h * 66 + r * 33 + w * 6 + s * 3 + c] * _1_1[r * 240 + s * 48 + c * 16 + f]
# CHECK-NEXT:  CODE: 0
