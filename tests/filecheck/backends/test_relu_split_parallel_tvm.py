# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

A = O.tensor((4, 5, 3, 7, 11), "float32", name="A")
with O.graph(name="relu_split_parallel") as gb:
    O.relu(A, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler()
sch.set_dims(["i", "j", "k", "l", "m"])
sch.split("m", {"left": 0, "right": 4})
sch.interchange(["i", "j", "k", "l", "m", "left", "right"])
# Fuse i and j in loop order. Skip l after the gap and m inside the branch.
sch.parallelize(["l", "j", "i"])
sch.interchange(["m"], root="./left")
sch.parallelize(["m"], root="./left")
sch.interchange(["m"], root="./right")
schedule = sch.schedule()

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_parallel_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(schedule)
    res = module.get_executor(validate=True).execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_split_parallel
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x5x3x7x11xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 4x5x3x7x11xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [4x5x3x7x11xfloat32] -> [4x5x3x7x11xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_parallel(_0: T.Buffer((4, 5, 3, 7, 11), "float32"), C: T.Buffer((4, 5, 3, 7, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k, l, m in T.grid(4, 5, 3, 7, 11):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k, v_l, v_m = T.axis.remap("SSSSS", [i, j, k, l, m])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j, v_k, v_l, v_m])
# CHECK-NEXT:                  T.writes(C[v_i, v_j, v_k, v_l, v_m])
# CHECK-NEXT:                  C[v_i, v_j, v_k, v_l, v_m] = T.max(T.float32(0.0), _0[v_i, v_j, v_k, v_l, v_m])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, l, m, = sch.get_loops(O)
# CHECK-NEXT:  sch.reorder(i, j, k, l, m)
# CHECK-NEXT:  j = sch.fuse(i, j)
# CHECK-NEXT:  sch.parallel(j)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, m, [4, None])
# CHECK-NEXT:  O_m_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_m_0)
# CHECK-NEXT:  assert len(__loops) == 1, 'unexpected physical loop chain for O_m_0'
# CHECK-NEXT:  m, = __loops
# CHECK-NEXT:  sch.reorder(m)
# CHECK-NEXT:  O_m_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_m_1)
# CHECK-NEXT:  assert len(__loops) == 1, 'unexpected physical loop chain for O_m_1'
# CHECK-NEXT:  m, = __loops
# CHECK-NEXT:  sch.reorder(m)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_parallel(_0: T.Buffer((4, 5, 3, 7, 11), "float32"), C: T.Buffer((4, 5, 3, 7, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_j_fused in T.parallel(20):
# CHECK-NEXT:              for k, l in T.grid(3, 7):
# CHECK-NEXT:                  with T.sblock("C_m_common"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      with T.sblock("C_m0_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for m0 in range(4):
# CHECK-NEXT:                              with T.sblock("C_m0"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(4, i_j_fused // 5)
# CHECK-NEXT:                                  v_j = T.axis.spatial(5, i_j_fused % 5)
# CHECK-NEXT:                                  v_k, v_l, v_m = T.axis.remap("SSS", [k, l, m0])
# CHECK-NEXT:                                  T.reads(_0[0:4, 0:5, 0:3, 0:7, 0:4])
# CHECK-NEXT:                                  T.writes(C[0:4, 0:5, 0:3, 0:7, 0:4])
# CHECK-NEXT:                                  C[v_i, v_j, v_k, v_l, v_m] = T.max(T.float32(0.0), _0[v_i, v_j, v_k, v_l, v_m])
# CHECK-NEXT:                      with T.sblock("C_m1_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for m1 in range(7):
# CHECK-NEXT:                              with T.sblock("C_m1"):
# CHECK-NEXT:                                  v_i = T.axis.spatial(4, i_j_fused // 5)
# CHECK-NEXT:                                  v_j = T.axis.spatial(5, i_j_fused % 5)
# CHECK-NEXT:                                  v_k, v_l = T.axis.remap("SS", [k, l])
# CHECK-NEXT:                                  v_m = T.axis.spatial((4, 7), m1 + 4)
# CHECK-NEXT:                                  T.reads(_0[0:4, 0:5, 0:3, 0:7, 4:7])
# CHECK-NEXT:                                  T.writes(C[0:4, 0:5, 0:3, 0:7, 4:7])
# CHECK-NEXT:                                  C[v_i, v_j, v_k, v_l, v_m] = T.max(T.float32(0.0), _0[v_i, v_j, v_k, v_l, v_m])
# CHECK-NEXT:  CODE: 0
