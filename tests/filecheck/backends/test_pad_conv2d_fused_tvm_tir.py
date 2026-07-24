# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph, tir_schedule=True)

sch = impl.get_scheduler()
sch.interchange(["b", "h", "w", "r", "s", "c", "f"])
sch.fuse_producer_at("w", 0)
sch.vectorize(["f"])
sched = sch.schedule()
comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_fused_tvm_tir",
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
# CHECK-NEXT:    - %2: pad2d(%0, padding={-3: (2, 2), -2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def pad_conv2d_nhwc_mini(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((5, 5, 3, 16), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tir.noalias": T.bool(True)})
# CHECK-NEXT:          # with T.block("root"):
# CHECK-NEXT:          pad = T.alloc_buffer((1, 12, 12, 3))
# CHECK-NEXT:          for i0, i1, i2, i3 in T.grid(1, 12, 12, 3):
# CHECK-NEXT:              with T.block("pad"):
# CHECK-NEXT:                  v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
# CHECK-NEXT:                  T.reads(_0[v_i0, v_i1 - 2, v_i2 - 2, v_i3])
# CHECK-NEXT:                  T.writes(pad[v_i0, v_i1, v_i2, v_i3])
# CHECK-NEXT:                  pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(2 <= v_i1 and v_i1 < 10 and 2 <= v_i2 and v_i2 < 10, _0[v_i0, v_i1 - 2, v_i2 - 2, v_i3], T.float32(0.0))
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 4, 4, 16, 5, 5, 3):
# CHECK-NEXT:              with T.block("conv"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_block("conv")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  I_F0 = sch.get_producers(O)[0]
# CHECK-NEXT:  sch.reorder(b, h, w, r, s, c, f)
# CHECK-NEXT:  sch.compute_at(I_F0, w)
# CHECK-NEXT:  sch.vectorize(f)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def pad_conv2d_nhwc_mini(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((5, 5, 3, 16), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tir.noalias": T.bool(True)})
# CHECK-NEXT:          # with T.block("root"):
# CHECK-NEXT:          pad = T.alloc_buffer((1, 12, 12, 3))
# CHECK-NEXT:          for b, h, w in T.grid(1, 4, 4):
# CHECK-NEXT:              for ax0, ax1, ax2 in T.grid(5, 5, 3):
# CHECK-NEXT:                  with T.block("pad"):
# CHECK-NEXT:                      v_i0 = T.axis.spatial(1, 0)
# CHECK-NEXT:                      v_i1 = T.axis.spatial(12, h * 2 + ax0)
# CHECK-NEXT:                      v_i2 = T.axis.spatial(12, w * 2 + ax1)
# CHECK-NEXT:                      v_i3 = T.axis.spatial(3, ax2)
# CHECK-NEXT:                      T.reads(_0[v_i0, v_i1 - 2, v_i2 - 2, v_i3])
# CHECK-NEXT:                      T.writes(pad[v_i0, v_i1, v_i2, v_i3])
# CHECK-NEXT:                      pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(2 <= v_i1 and v_i1 < 10 and 2 <= v_i2 and v_i2 < 10, _0[v_i0, v_i1 - 2, v_i2 - 2, v_i3], T.float32(0.0))
# CHECK-NEXT:              for r, s, c in T.grid(5, 5, 3):
# CHECK-NEXT:                  for f in T.vectorized(16):
# CHECK-NEXT:                      with T.block("conv"):
# CHECK-NEXT:                          v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                          T.reads(pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                          T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                          with T.init():
# CHECK-NEXT:                              conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                          conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
