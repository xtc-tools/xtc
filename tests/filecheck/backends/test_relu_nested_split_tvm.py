# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

A = O.tensor((19, 5), "float32", name="A")
with O.graph(name="relu_nested_split") as gb:
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
        "i": {},
        "i#16": {},
        "i#8": {},
        "i[:5]": {
            "i[:2]": {"i#2": {}, "j": {}},
            "i[2:]": {"i#3": {}, "j": {}},
        },
        "i[5:7]": {"i#2": {}, "j": {}},
        "i[7:]": {"j": {}},
    },
)
schedule = sch.schedule()

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_nested_split_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(schedule)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0

# CHECK:       graph:
# CHECK-NEXT:    name: relu_nested_split
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 19x5xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 19x5xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [19x5xfloat32] -> [19x5xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_nested_split(_0: T.Buffer((19, 5), "float32"), C: T.Buffer((19, 5), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j in T.grid(19, 5):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  i, i0, i1, = sch.split(i, factors=[None, 2, 8])
# CHECK-NEXT:  sch.reorder(i, i0, i1, j)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i1, [5, 2, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i1, j, = __loops
# CHECK-NEXT:  sch.reorder(i1, j)
# CHECK-NEXT:  sch, O_i_0_SPLITS = loop_partition_rebased(sch, i1, [2, None])
# CHECK-NEXT:  O_i_0_i_0 = sch.get_sblock(O_i_0_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0_i_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0_i_0'
# CHECK-NEXT:  i1, j, = __loops
# CHECK-NEXT:  i1, i0, = sch.split(i1, factors=[None, 2])
# CHECK-NEXT:  sch.reorder(i1, i0, j)
# CHECK-NEXT:  O_i_0_i_1 = sch.get_sblock(O_i_0_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0_i_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0_i_1'
# CHECK-NEXT:  i1, j, = __loops
# CHECK-NEXT:  i1, i0, = sch.split(i1, factors=[None, 3])
# CHECK-NEXT:  sch.reorder(i1, i0, j)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i1, j, = __loops
# CHECK-NEXT:  i1, i0, = sch.split(i1, factors=[None, 2])
# CHECK-NEXT:  sch.reorder(i1, i0, j)
# CHECK-NEXT:  O_i_2 = sch.get_sblock(O_SPLITS[2])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_2)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_2'
# CHECK-NEXT:  i1, j, = __loops
# CHECK-NEXT:  sch.reorder(i1, j)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_nested_split(_0: T.Buffer((19, 5), "float32"), C: T.Buffer((19, 5), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0, i_1 in T.grid(2, 2):
# CHECK-NEXT:              with T.sblock("C_i_2_common"):
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes()
# CHECK-NEXT:                  with T.sblock("C_i_20_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      with T.sblock("C_i_20_i_20_common"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          with T.sblock("C_i_20_i_200_partition"):
# CHECK-NEXT:                              T.reads()
# CHECK-NEXT:                              T.writes()
# CHECK-NEXT:                              for i_200_0, i_200_1, j in T.grid(1, 2, 5):
# CHECK-NEXT:                                  with T.sblock("C_i_20_i_200"):
# CHECK-NEXT:                                      v_i = T.axis.spatial(19, i_200_0 * 2 + i_200_1 + (i_0 * 16 + i_1 * 8))
# CHECK-NEXT:                                      v_j = T.axis.spatial(5, j)
# CHECK-NEXT:                                      T.where((i_0 * 2 + i_1) * 8 + (i_200_0 * 2 + i_200_1) < 19)
# CHECK-NEXT:                                      T.reads(_0[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                                      T.writes(C[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                                      C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                          with T.sblock("C_i_20_i_201_partition"):
# CHECK-NEXT:                              T.reads()
# CHECK-NEXT:                              T.writes()
# CHECK-NEXT:                              for i_201_0, i_201_1, j in T.grid(1, 3, 5):
# CHECK-NEXT:                                  with T.sblock("C_i_20_i_201"):
# CHECK-NEXT:                                      v_i = T.axis.spatial(19, i_201_0 * 3 + i_201_1 + (i_0 * 16 + i_1 * 8 + 2))
# CHECK-NEXT:                                      v_j = T.axis.spatial(5, j)
# CHECK-NEXT:                                      T.where((i_0 * 2 + i_1) * 8 + (i_201_0 * 3 + i_201_1 + 2) < 19)
# CHECK-NEXT:                                      T.reads(_0[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                                      T.writes(C[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                                      C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i_21_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i_21_0, i_21_1, j in T.grid(1, 2, 5):
# CHECK-NEXT:                          with T.sblock("C_i_21"):
# CHECK-NEXT:                              v_i = T.axis.spatial(19, i_21_0 * 2 + i_21_1 + (i_0 * 16 + i_1 * 8 + 5))
# CHECK-NEXT:                              v_j = T.axis.spatial(5, j)
# CHECK-NEXT:                              T.where((i_0 * 2 + i_1) * 8 + (i_21_0 * 2 + i_21_1 + 5) < 19)
# CHECK-NEXT:                              T.reads(_0[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                              T.writes(C[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                              C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i_22_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i_22, j in T.grid(1, 5):
# CHECK-NEXT:                          with T.sblock("C_i_22"):
# CHECK-NEXT:                              v_i = T.axis.spatial(19, i_0 * 16 + i_1 * 8 + (i_22 + 7))
# CHECK-NEXT:                              v_j = T.axis.spatial(5, j)
# CHECK-NEXT:                              T.where((i_0 * 2 + i_1) * 8 + (i_22 + 7) < 19)
# CHECK-NEXT:                              T.reads(_0[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                              T.writes(C[v_i:v_i + (1 - v_i - v_i), v_j:v_j + (1 - v_j - v_j)])
# CHECK-NEXT:                              C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  CODE: 0
