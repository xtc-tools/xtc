# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

A = O.tensor((19, 23, 7), "float32", name="A")
with O.graph(name="relu_split_shared_tiles") as gb:
    O.relu(A, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i", "j", "k"],
    spec={
        "k": {"unroll": 2},  # The common k#3 refinement gives this a span of 6.
        "i[:10]": {
            "j[:5]": {"k#3": {"unroll": 10}, "k#2": {}},  # Fully unroll 2 trips.
            "j[5:]": {"k#3": {}},
        },
        "i[10:]": {"j": {}, "k#3": {}, "i#4": {}},
    },
)
schedule = sch.schedule()

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_shared_tiles_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(schedule)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0

# CHECK:       graph:
# CHECK-NEXT:    name: relu_split_shared_tiles
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 19x23x7xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 19x23x7xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [19x23x7xfloat32] -> [19x23x7xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_shared_tiles(_0: T.Buffer((19, 23, 7), "float32"), C: T.Buffer((19, 23, 7), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k in T.grid(19, 23, 7):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j, v_k])
# CHECK-NEXT:                  T.writes(C[v_i, v_j, v_k])
# CHECK-NEXT:                  C[v_i, v_j, v_k] = T.max(T.float32(0.0), _0[v_i, v_j, v_k])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  k, __u_k, k0, = sch.split(k, factors=[None, 2, 3])
# CHECK-NEXT:  sch.reorder(k, __u_k, i, j, k0)
# CHECK-NEXT:  sch.unroll(__u_k)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [10, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 3, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, j, k0, = __loops
# CHECK-NEXT:  sch.reorder(i, j, k0)
# CHECK-NEXT:  sch, O_i_0_SPLITS = loop_partition_rebased(sch, j, [5, None])
# CHECK-NEXT:  O_i_0_j_0 = sch.get_sblock(O_i_0_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0_j_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0_j_0'
# CHECK-NEXT:  j, k0, = __loops
# CHECK-NEXT:  k0, k1, = sch.split(k0, factors=[None, 2])
# CHECK-NEXT:  sch.reorder(j, k0, k1)
# CHECK-NEXT:  sch.unroll(k0)
# CHECK-NEXT:  O_i_0_j_1 = sch.get_sblock(O_i_0_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0_j_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0_j_1'
# CHECK-NEXT:  j, k0, = __loops
# CHECK-NEXT:  sch.reorder(j, k0)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 3, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, j, k0, = __loops
# CHECK-NEXT:  i, i0, = sch.split(i, factors=[None, 4])
# CHECK-NEXT:  sch.reorder(i, j, k0, i0)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_shared_tiles(_0: T.Buffer((19, 23, 7), "float32"), C: T.Buffer((19, 23, 7), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for k_0 in range(2):
# CHECK-NEXT:              for k_1 in T.unroll(2):
# CHECK-NEXT:                  with T.sblock("C_i_common"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      with T.sblock("C_i0_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for i0 in range(10):
# CHECK-NEXT:                              with T.sblock("C_i0_j_common"):
# CHECK-NEXT:                                  T.reads()
# CHECK-NEXT:                                  T.writes()
# CHECK-NEXT:                                  with T.sblock("C_i0_j0_partition"):
# CHECK-NEXT:                                      T.reads()
# CHECK-NEXT:                                      T.writes()
# CHECK-NEXT:                                      for j0 in range(5):
# CHECK-NEXT:                                          for k_2_0 in T.unroll(2):
# CHECK-NEXT:                                              for k_2_1 in range(2):
# CHECK-NEXT:                                                  with T.sblock("C_i0_j0"):
# CHECK-NEXT:                                                      v_i, v_j = T.axis.remap("SS", [i0, j0])
# CHECK-NEXT:                                                      v_k = T.axis.spatial(7, k_2_0 * 2 + k_2_1 + (k_0 * 6 + k_1 * 3))
# CHECK-NEXT:                                                      T.where((k_0 * 2 + k_1) * 3 + (k_2_0 * 2 + k_2_1) < 7 and k_2_0 * 2 + k_2_1 < 3)
# CHECK-NEXT:                                                      T.reads(_0[0:10, 0:5, 0:7])
# CHECK-NEXT:                                                      T.writes(C[0:10, 0:5, 0:7])
# CHECK-NEXT:                                                      C[v_i, v_j, v_k] = T.max(T.float32(0.0), _0[v_i, v_j, v_k])
# CHECK-NEXT:                                  with T.sblock("C_i0_j1_partition"):
# CHECK-NEXT:                                      T.reads()
# CHECK-NEXT:                                      T.writes()
# CHECK-NEXT:                                      for j1, k_2 in T.grid(18, 3):
# CHECK-NEXT:                                          with T.sblock("C_i0_j1"):
# CHECK-NEXT:                                              v_i = T.axis.spatial(10, i0)
# CHECK-NEXT:                                              v_j = T.axis.spatial((5, 18), j1 + 5)
# CHECK-NEXT:                                              v_k = T.axis.spatial(7, k_0 * 6 + k_1 * 3 + k_2)
# CHECK-NEXT:                                              T.where((k_0 * 2 + k_1) * 3 + k_2 < 7)
# CHECK-NEXT:                                              T.reads(_0[0:10, 5:18, 0:7])
# CHECK-NEXT:                                              T.writes(C[0:10, 5:18, 0:7])
# CHECK-NEXT:                                              C[v_i, v_j, v_k] = T.max(T.float32(0.0), _0[v_i, v_j, v_k])
# CHECK-NEXT:                      with T.sblock("C_i1_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for i1_0, j, k_2, i1_1 in T.grid(3, 23, 3, 4):
# CHECK-NEXT:                              with T.sblock("C_i1"):
# CHECK-NEXT:                                  v_i = T.axis.spatial((10, -1), i1_0 * 4 + i1_1 + 10)
# CHECK-NEXT:                                  v_j = T.axis.spatial(23, j)
# CHECK-NEXT:                                  v_k = T.axis.spatial(7, k_2 + (k_0 * 6 + k_1 * 3))
# CHECK-NEXT:                                  T.where((k_0 * 2 + k_1) * 3 + k_2 < 7 and i1_0 * 4 + i1_1 < 9)
# CHECK-NEXT:                                  T.reads(_0[10:-1, 0:23, 0:7])
# CHECK-NEXT:                                  T.writes(C[10:-1, 0:23, 0:7])
# CHECK-NEXT:                                  C[v_i, v_j, v_k] = T.max(T.float32(0.0), _0[v_i, v_j, v_k])
# CHECK-NEXT:  CODE: 0
