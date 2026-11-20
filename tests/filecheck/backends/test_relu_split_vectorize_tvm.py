# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

inp = O.tensor((19, 53), "float32", name="A")
with O.graph(name="relu_normalized_shared_vector_tile") as gb:
    O.relu(inp, name="C")

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
        "i[:10]": {"j#48": {"vectorize": True}},
        "i[10:]": {"j#48": {"vectorize": True}},
    },
)
sched = sch.schedule()
with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_vectorize_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sched)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_normalized_shared_vector_tile
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 19x53xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 19x53xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [19x53xfloat32] -> [19x53xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_normalized_shared_vector_tile(_0: T.Buffer((19, 53), "float32"), C: T.Buffer((19, 53), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j in T.grid(19, 53):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  j, j0, __v_j0, = sch.split(j, factors=[None, 3, 16])
# CHECK-NEXT:  sch.reorder(j, i, j0, __v_j0)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [10, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 3, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, j0, __v_j0, = __loops
# CHECK-NEXT:  sch.reorder(i, j0, __v_j0)
# CHECK-NEXT:  sch.unroll(j0)
# CHECK-NEXT:  sch.vectorize(__v_j0)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 3, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, j0, __v_j0, = __loops
# CHECK-NEXT:  sch.reorder(i, j0, __v_j0)
# CHECK-NEXT:  sch.unroll(j0)
# CHECK-NEXT:  sch.vectorize(__v_j0)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_normalized_shared_vector_tile(_0: T.Buffer((19, 53), "float32"), C: T.Buffer((19, 53), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for j_0 in range(2):
# CHECK-NEXT:              with T.sblock("C_i_common"):
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes()
# CHECK-NEXT:                  with T.sblock("C_i0_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i0 in range(10):
# CHECK-NEXT:                          for j_1 in T.unroll(3):
# CHECK-NEXT:                              for j_2 in T.vectorized(16):
# CHECK-NEXT:                                  with T.sblock("C_i0"):
# CHECK-NEXT:                                      v_i = T.axis.spatial(10, i0)
# CHECK-NEXT:                                      v_j = T.axis.spatial(53, j_0 * 48 + j_1 * 16 + j_2)
# CHECK-NEXT:                                      T.where((j_0 * 3 + j_1) * 16 + j_2 < 53)
# CHECK-NEXT:                                      T.reads(_0[0:10, 0:53])
# CHECK-NEXT:                                      T.writes(C[0:10, 0:53])
# CHECK-NEXT:                                      C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i1_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i1 in range(9):
# CHECK-NEXT:                          for j_1 in T.unroll(3):
# CHECK-NEXT:                              for j_2 in T.vectorized(16):
# CHECK-NEXT:                                  with T.sblock("C_i1"):
# CHECK-NEXT:                                      v_i = T.axis.spatial((10, 9), i1 + 10)
# CHECK-NEXT:                                      v_j = T.axis.spatial(53, j_0 * 48 + j_1 * 16 + j_2)
# CHECK-NEXT:                                      T.where((j_0 * 3 + j_1) * 16 + j_2 < 53)
# CHECK-NEXT:                                      T.reads(_0[10:9, 0:53])
# CHECK-NEXT:                                      T.writes(C[10:9, 0:53])
# CHECK-NEXT:                                      C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  CODE: 0
