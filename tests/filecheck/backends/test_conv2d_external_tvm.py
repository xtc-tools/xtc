# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

from pathlib import Path

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

# A reduced version of Yolo9000_12
N, H, W, F, R, S, C = 1, 34, 34, 128, 3, 3, 32
SH, SW = 1, 1
dtype = "float32"
abstract_dims = ["b", "h", "w", "f", "r", "s", "c"]
abstract_sizes = dict(zip(abstract_dims, [N, H, W, F, R, S, C]))

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_yolo9k2") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="C")

conv2d_uk = "external_conv2d_uk_wxcxf"
impl = Backend(gb.graph)
sch = impl.get_scheduler()
sch.strip_mine("w", {"w0": 2})
sch.strip_mine("c", {"c0": 4})
sch.strip_mine("f", {"f0": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w0", "c0", "f0"])
sch.external_at("c", conv2d_uk)

print(sch.get_loop_nest().root_node.pretty_print())

schedule = sch.schedule()

microkernel_csrc = Path(__file__).parent / f"{conv2d_uk}.c"
res = impl.evaluate(
    schedule,
    compiler_args=dict(
        save_temps=True,
        print_transformed_ir=True,
        csrcs=[str(microkernel_csrc)],
    ),
)

print("VALID:", isinstance(res, float))

# CHECK:       loop b
# CHECK-NEXT:    loop h
# CHECK-NEXT:      loop w
# CHECK-NEXT:        loop f
# CHECK-NEXT:          loop r
# CHECK-NEXT:            loop s
# CHECK-NEXT:              loop c  // external(external_conv2d_uk_wxcxf)
# CHECK-NEXT:                tile(w, 2)
# CHECK-NEXT:                  tile(c, 4)
# CHECK-NEXT:                    tile(f, 16)
# CHECK-NEXT:                      ...
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  w, w0, = sch.split(w, factors=[None, 2])
# CHECK-NEXT:  f, f0, = sch.split(f, factors=[None, 16])
# CHECK-NEXT:  c, c0, = sch.split(c, factors=[None, 4])
# CHECK-NEXT:  sch.reorder(b, h, w, f, r, s, c, w0, c0, f0)
# CHECK-NEXT:  sch = externalize_tile_below(sch, O, c, 'external_conv2d_uk_wxcxf')
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_yolo9k2(_0: T.Buffer((1, 36, 36, 32), "float32"), _1: T.Buffer((3, 3, 32, 128), "float32"), C: T.Buffer((1, 34, 34, 128), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w_0, f_0 in T.grid(1, 34, 17, 8):
# CHECK-NEXT:              for w_1_init, f_1_init in T.grid(2, 16):
# CHECK-NEXT:                  with T.sblock("C_init"):
# CHECK-NEXT:                      v_b, v_h = T.axis.remap("SS", [b, h])
# CHECK-NEXT:                      v_w = T.axis.spatial(34, w_0 * 2 + w_1_init)
# CHECK-NEXT:                      v_f = T.axis.spatial(128, f_0 * 16 + f_1_init)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                      C[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:              for r, s, c_0 in T.grid(3, 3, 8):
# CHECK-NEXT:                  T.call_extern("int32", "external_conv2d_uk_wxcxf", T.tvm_access_ptr(T.type_annotation("float32"), C.data, b * 147968 + h * 4352 + T.Add(w_0 * 2, 0) * 128 + T.Add(f_0 * 16, 0), 147968 - (b * 147968 + h * 4352 + T.Add(w_0 * 2, 0) * 128 + T.Add(f_0 * 16, 0)), 3), T.tvm_access_ptr(T.type_annotation("float32"), _0.data, b * 41472 + (h + r) * 1152 + (T.Add(w_0 * 2, 0) + s) * 32 + T.Add(c_0 * 4, 0), 41472 - (b * 41472 + (h + r) * 1152 + (T.Add(w_0 * 2, 0) + s) * 32 + T.Add(c_0 * 4, 0)), 1), T.tvm_access_ptr(T.type_annotation("float32"), _1.data, r * 12288 + s * 4096 + T.Add(c_0 * 4, 0) * 128 + T.Add(f_0 * 16, 0), 36864 - (r * 12288 + s * 4096 + T.Add(c_0 * 4, 0) * 128 + T.Add(f_0 * 16, 0)), 1), T.int64(2), T.int64(4), T.int64(16), T.int64(128), T.int64(0), T.int64(1), T.int64(32), T.int64(1), T.int64(0), T.int64(0), T.int64(128), T.int64(1))
# CHECK-NEXT:  VALID: True
