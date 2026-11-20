# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

# Odd extents exercise spatial and reduction tails in each split branch.
A = O.tensor((1, 9, 6, 5), "float32", name="A")
W = O.tensor((3, 2, 5, 5), "float32", name="W")
with O.graph(name="conv2d_split") as gb:
    O.conv2d(A, W, name="C")

graph = gb.graph
print(graph)
impl = Backend(graph)
sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["b", "h", "w", "f", "r", "s", "c"],
    spec={
        "b": {},
        "w": {},
        "f": {},
        "r": {},
        "s": {},
        "c": {},
        "h[:3]": {
            "h#2": {"unroll": True},
            "w#2": {},
            "r#2": {},
            "c#2": {},
        },
        "h[3:]": {
            "h#3": {"unroll": True},
            "w#2": {},
            "r#2": {},
            "c#2": {},
        },
    },
)
sched = sch.schedule()
with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/conv2d_split_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sched)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_split
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x9x6x5xfloat32
# CHECK-NEXT:    - %1 : 3x2x5x5xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x7x5x5xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1) {name = 'C'} : [1x9x6x5xfloat32, 3x2x5x5xfloat32] -> [1x7x5x5xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_split(_0: T.Buffer((1, 9, 6, 5), "float32"), _1: T.Buffer((3, 2, 5, 5), "float32"), C: T.Buffer((1, 7, 5, 5), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 7, 5, 5, 3, 2, 5):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(C[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  C[v_b, v_h, v_w, v_f] = C[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  w, w0, = sch.split(w, factors=[None, 2])
# CHECK-NEXT:  r, r0, = sch.split(r, factors=[None, 2])
# CHECK-NEXT:  c, c0, = sch.split(c, factors=[None, 2])
# CHECK-NEXT:  sch.reorder(b, w, f, r, s, c, h, w0, r0, c0)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, h, [3, None])
# CHECK-NEXT:  O_h_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_h_0)
# CHECK-NEXT:  assert len(__loops) == 4, 'unexpected physical loop chain for O_h_0'
# CHECK-NEXT:  h, w0, r0, c0, = __loops
# CHECK-NEXT:  h, h0, = sch.split(h, factors=[None, 2])
# CHECK-NEXT:  sch.reorder(h, h0, w0, r0, c0)
# CHECK-NEXT:  sch.unroll(h0)
# CHECK-NEXT:  O_h_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_h_1)
# CHECK-NEXT:  assert len(__loops) == 4, 'unexpected physical loop chain for O_h_1'
# CHECK-NEXT:  h, w0, r0, c0, = __loops
# CHECK-NEXT:  h, h0, = sch.split(h, factors=[None, 3])
# CHECK-NEXT:  sch.reorder(h, h0, w0, r0, c0)
# CHECK-NEXT:  sch.unroll(h0)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_split(_0: T.Buffer((1, 9, 6, 5), "float32"), _1: T.Buffer((3, 2, 5, 5), "float32"), C: T.Buffer((1, 7, 5, 5), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, w_0, f in T.grid(1, 3, 5):
# CHECK-NEXT:              for h0_0_init in range(2):
# CHECK-NEXT:                  for h0_1_init in T.unroll(2):
# CHECK-NEXT:                      for w_1_init in range(2):
# CHECK-NEXT:                          with T.sblock("C_h0_init"):
# CHECK-NEXT:                              v_b_init = T.axis.spatial(1, b)
# CHECK-NEXT:                              v_h_init = T.axis.spatial(3, h0_0_init * 2 + h0_1_init)
# CHECK-NEXT:                              v_w_init = T.axis.spatial(5, w_1_init + w_0 * 2)
# CHECK-NEXT:                              v_f_init = T.axis.spatial(5, f)
# CHECK-NEXT:                              T.where(w_0 * 2 + w_1_init < 5 and T.Add(T.Mul(0, 2), 0) < 3 and T.Add(T.Mul(0, 2), 0) < 5 and h0_0_init * 2 + h0_1_init < 3)
# CHECK-NEXT:                              T.reads()
# CHECK-NEXT:                              T.writes(C[0, 0:3, 0:5, 0:5])
# CHECK-NEXT:                              C[v_b_init, v_h_init, v_w_init, v_f_init] = T.float32(0.0)
# CHECK-NEXT:              for h1_0_init in range(2):
# CHECK-NEXT:                  for h1_1_init in T.unroll(3):
# CHECK-NEXT:                      for w_1_init in range(2):
# CHECK-NEXT:                          with T.sblock("C_h1_init"):
# CHECK-NEXT:                              v_b_init = T.axis.spatial(1, b)
# CHECK-NEXT:                              v_h_init = T.axis.spatial((3, 4), h1_0_init * 3 + h1_1_init + 3)
# CHECK-NEXT:                              v_w_init = T.axis.spatial(5, w_1_init + w_0 * 2)
# CHECK-NEXT:                              v_f_init = T.axis.spatial(5, f)
# CHECK-NEXT:                              T.where(w_0 * 2 + w_1_init < 5 and T.Add(T.Mul(0, 2), 0) < 3 and T.Add(T.Mul(0, 2), 0) < 5 and h1_0_init * 3 + h1_1_init < 4)
# CHECK-NEXT:                              T.reads()
# CHECK-NEXT:                              T.writes(C[0, 3, 0:5, 0:5])
# CHECK-NEXT:                              C[v_b_init, v_h_init, v_w_init, v_f_init] = T.float32(0.0)
# CHECK-NEXT:              for r_0, s, c_0 in T.grid(2, 2, 3):
# CHECK-NEXT:                  with T.sblock("C_h_common"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      with T.sblock("C_h0_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for h0_0 in range(2):
# CHECK-NEXT:                              for h0_1 in T.unroll(2):
# CHECK-NEXT:                                  for w_1, r_1, c_1 in T.grid(2, 2, 2):
# CHECK-NEXT:                                      with T.sblock("C_h0_update"):
# CHECK-NEXT:                                          v_b = T.axis.spatial(1, b)
# CHECK-NEXT:                                          v_h = T.axis.spatial(3, h0_0 * 2 + h0_1)
# CHECK-NEXT:                                          v_w = T.axis.spatial(5, w_1 + w_0 * 2)
# CHECK-NEXT:                                          v_f = T.axis.spatial(5, f)
# CHECK-NEXT:                                          v_r = T.axis.reduce(3, r_1 + r_0 * 2)
# CHECK-NEXT:                                          v_s = T.axis.reduce(2, s)
# CHECK-NEXT:                                          v_c = T.axis.reduce(5, c_1 + c_0 * 2)
# CHECK-NEXT:                                          T.where(w_0 * 2 + w_1 < 5 and r_0 * 2 + r_1 < 3 and c_0 * 2 + c_1 < 5 and h0_0 * 2 + h0_1 < 3)
# CHECK-NEXT:                                          T.reads(_0[v_b:v_b + (1 - v_b), v_h + v_r:v_h + v_r + (1 - (v_h + v_r)), v_w + v_s:v_w + v_s + (1 - (v_w + v_s)), v_c:v_c + (1 - v_c)], _1[0:3, 0:2, 0:5, 0:5], C[0, 0:3, 0:5, 0:5])
# CHECK-NEXT:                                          T.writes(C[0, 0:3, 0:5, 0:5])
# CHECK-NEXT:                                          C[v_b, v_h, v_w, v_f] = C[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:                      with T.sblock("C_h1_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for h1_0 in range(2):
# CHECK-NEXT:                              for h1_1 in T.unroll(3):
# CHECK-NEXT:                                  for w_1, r_1, c_1 in T.grid(2, 2, 2):
# CHECK-NEXT:                                      with T.sblock("C_h1_update"):
# CHECK-NEXT:                                          v_b = T.axis.spatial(1, b)
# CHECK-NEXT:                                          v_h = T.axis.spatial((3, 4), h1_0 * 3 + h1_1 + 3)
# CHECK-NEXT:                                          v_w = T.axis.spatial(5, w_1 + w_0 * 2)
# CHECK-NEXT:                                          v_f = T.axis.spatial(5, f)
# CHECK-NEXT:                                          v_r = T.axis.reduce(3, r_1 + r_0 * 2)
# CHECK-NEXT:                                          v_s = T.axis.reduce(2, s)
# CHECK-NEXT:                                          v_c = T.axis.reduce(5, c_1 + c_0 * 2)
# CHECK-NEXT:                                          T.where(w_0 * 2 + w_1 < 5 and r_0 * 2 + r_1 < 3 and c_0 * 2 + c_1 < 5 and h1_0 * 3 + h1_1 < 4)
# CHECK-NEXT:                                          T.reads(_0[v_b:v_b + (1 - v_b), v_h + v_r:v_h + v_r + (1 - (v_h + v_r)), v_w + v_s:v_w + v_s + (1 - (v_w + v_s)), v_c:v_c + (1 - v_c)], _1[0:3, 0:2, 0:5, 0:5], C[0, 3, 0:5, 0:5])
# CHECK-NEXT:                                          T.writes(C[0, 3, 0:5, 0:5])
# CHECK-NEXT:                                          C[v_b, v_h, v_w, v_f] = C[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
