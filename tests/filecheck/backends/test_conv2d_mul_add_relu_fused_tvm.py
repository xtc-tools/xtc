# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")
scale = O.tensor((F,), dtype, name="scale")
bias = O.tensor((F,), dtype, name="bias")

with O.graph(name="conv2d_mul_add_relu") as gb:
    c = O.conv2d(a, b, stride=(SH, SW), name="conv")
    d = O.mul(c, scale, name="mul")
    e = O.add(d, bias, name="add")
    f = O.relu(e, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="conv")
sch.fuse_consumer_at("f", chain=True)
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_mul_add_relu_fused_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_mul_add_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    - %2 : 16xfloat32
# CHECK-NEXT:    - %3 : 16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %7 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %4: conv2d(%0, %1, stride=(1, 1)) {name = 'conv'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %5: mul(%4, %2) {name = 'mul'} : [1x8x8x16xfloat32, 16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %6: add(%5, %3) {name = 'add'} : [1x8x8x16xfloat32, 16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %7: relu(%6) {name = 'relu'} : [1x8x8x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_mul_add_relu(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), _2: T.Buffer((16,), "float32"), _3: T.Buffer((16,), "float32"), relu: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          conv = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          mul = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          add = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 8, 8, 16, 3, 3, 3):
# CHECK-NEXT:              with T.sblock("conv"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:          for i, j, k, l in T.grid(1, 8, 8, 16):
# CHECK-NEXT:              with T.sblock("mul"):
# CHECK-NEXT:                  v_i, v_j, v_k, v_l = T.axis.remap("SSSS", [i, j, k, l])
# CHECK-NEXT:                  T.reads(conv[v_i, v_j, v_k, v_l], _2[v_l])
# CHECK-NEXT:                  T.writes(mul[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  mul[v_i, v_j, v_k, v_l] = conv[v_i, v_j, v_k, v_l] * _2[v_l]
# CHECK-NEXT:          for i, j, k, l in T.grid(1, 8, 8, 16):
# CHECK-NEXT:              with T.sblock("add"):
# CHECK-NEXT:                  v_i, v_j, v_k, v_l = T.axis.remap("SSSS", [i, j, k, l])
# CHECK-NEXT:                  T.reads(mul[v_i, v_j, v_k, v_l], _3[v_l])
# CHECK-NEXT:                  T.writes(add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  add[v_i, v_j, v_k, v_l] = mul[v_i, v_j, v_k, v_l] + _3[v_l]
# CHECK-NEXT:          for i, j, k, l in T.grid(1, 8, 8, 16):
# CHECK-NEXT:              with T.sblock("relu"):
# CHECK-NEXT:                  v_i, v_j, v_k, v_l = T.axis.remap("SSSS", [i, j, k, l])
# CHECK-NEXT:                  T.reads(add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  T.writes(relu[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  relu[v_i, v_j, v_k, v_l] = T.max(T.float32(0.0), add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:  O = sch.get_sblock("conv")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  O_F0 = sch.get_consumers(O)[0]
# CHECK-NEXT:  O_F1 = sch.get_sblock("add")
# CHECK-NEXT:  O_F2 = sch.get_sblock("relu")
# CHECK-NEXT:  sch.reorder(b, h, w, f, r, s, c)
# CHECK-NEXT:  sch.reverse_compute_at(O_F0, f)
# CHECK-NEXT:  sch.reverse_compute_at(O_F1, f)
# CHECK-NEXT:  sch.reverse_compute_at(O_F2, f)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_mul_add_relu(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), _2: T.Buffer((16,), "float32"), _3: T.Buffer((16,), "float32"), relu: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          conv = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          mul = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          add = T.sblock_alloc_buffer((1, 8, 8, 16))
# CHECK-NEXT:          for b, h, w, f in T.grid(1, 8, 8, 16):
# CHECK-NEXT:              with T.sblock("conv_init"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f = T.axis.remap("SSSS", [b, h, w, f])
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c in T.grid(3, 3, 3):
# CHECK-NEXT:                  with T.sblock("conv_update"):
# CHECK-NEXT:                      v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                      T.reads(conv[v_b, v_h, v_w, v_f], _0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                      T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                      conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:              with T.sblock("mul"):
# CHECK-NEXT:                  v_i = T.axis.spatial(1, 0)
# CHECK-NEXT:                  v_j, v_k, v_l = T.axis.remap("SSS", [h, w, f])
# CHECK-NEXT:                  T.reads(conv[v_i, v_j, v_k, v_l], _2[v_l])
# CHECK-NEXT:                  T.writes(mul[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  mul[v_i, v_j, v_k, v_l] = conv[v_i, v_j, v_k, v_l] * _2[v_l]
# CHECK-NEXT:              with T.sblock("add"):
# CHECK-NEXT:                  v_i = T.axis.spatial(1, 0)
# CHECK-NEXT:                  v_j, v_k, v_l = T.axis.remap("SSS", [h, w, f])
# CHECK-NEXT:                  T.reads(mul[v_i, v_j, v_k, v_l], _3[v_l])
# CHECK-NEXT:                  T.writes(add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  add[v_i, v_j, v_k, v_l] = mul[v_i, v_j, v_k, v_l] + _3[v_l]
# CHECK-NEXT:              with T.sblock("relu"):
# CHECK-NEXT:                  v_i = T.axis.spatial(1, 0)
# CHECK-NEXT:                  v_j, v_k, v_l = T.axis.remap("SSS", [h, w, f])
# CHECK-NEXT:                  T.reads(add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  T.writes(relu[v_i, v_j, v_k, v_l])
# CHECK-NEXT:                  relu[v_i, v_j, v_k, v_l] = T.max(T.float32(0.0), add[v_i, v_j, v_k, v_l])
# CHECK-NEXT:  CODE: 0
