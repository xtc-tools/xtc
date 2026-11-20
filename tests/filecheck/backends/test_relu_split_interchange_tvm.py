# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

A = O.tensor((7, 11), "float32", name="A")
with O.graph(name="relu_split_interchange") as gb:
    O.relu(A, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i", "j"],
    spec={
        "j": {},
        "j#4": {},
        "i[:3]": {"j#2": {}},
        "i[3:]": {"j#2": {}, "i": {"parallelize": True}},
    },
)
schedule = sch.schedule()

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_interchange_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(schedule)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_split_interchange
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 7x11xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 7x11xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [7x11xfloat32] -> [7x11xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_interchange(_0: T.Buffer((7, 11), "float32"), C: T.Buffer((7, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j in T.grid(7, 11):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  j, j0, __t_j_1, = sch.split(j, factors=[None, 2, 2])
# CHECK-NEXT:  sch.reorder(j, j0, i, __t_j_1)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [3, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, __t_j_1, = __loops
# CHECK-NEXT:  sch.reorder(i, __t_j_1)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, __t_j_1, = __loops
# CHECK-NEXT:  sch.reorder(__t_j_1, i)
# CHECK-NEXT:  sch.parallel(i)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_interchange(_0: T.Buffer((7, 11), "float32"), C: T.Buffer((7, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for j_0, j_1 in T.grid(3, 2):
# CHECK-NEXT:              with T.sblock("C_i_common"):
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes()
# CHECK-NEXT:                  with T.sblock("C_i0_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i0, j_2 in T.grid(3, 2):
# CHECK-NEXT:                          with T.sblock("C_i0"):
# CHECK-NEXT:                              v_i = T.axis.spatial(3, i0)
# CHECK-NEXT:                              v_j = T.axis.spatial(11, j_0 * 4 + j_1 * 2 + j_2)
# CHECK-NEXT:                              T.where((j_0 * 2 + j_1) * 2 + j_2 < 11)
# CHECK-NEXT:                              T.reads(_0[0:3, 0:11])
# CHECK-NEXT:                              T.writes(C[0:3, 0:11])
# CHECK-NEXT:                              C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i1_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for j_2 in range(2):
# CHECK-NEXT:                          for i1 in T.parallel(4):
# CHECK-NEXT:                              with T.sblock("C_i1"):
# CHECK-NEXT:                                  v_i = T.axis.spatial((3, 4), i1 + 3)
# CHECK-NEXT:                                  v_j = T.axis.spatial(11, j_0 * 4 + j_1 * 2 + j_2)
# CHECK-NEXT:                                  T.where((j_0 * 2 + j_1) * 2 + j_2 < 11)
# CHECK-NEXT:                                  T.reads(_0[3, 0:11])
# CHECK-NEXT:                                  T.writes(C[3, 0:11])
# CHECK-NEXT:                                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  CODE: 0
