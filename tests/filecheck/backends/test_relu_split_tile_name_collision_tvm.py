# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

A = O.tensor((17, 63), "float32", name="A")
with O.graph(name="relu_split_tile_name_collision") as gb:
    O.relu(A, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler()
sch.split("i", {"left": 0, "right": 6})
sch.tile("j", {"__t_j_2": 16, "c": 8})
sch.interchange(["j", "__t_j_2", "c", "i", "left", "right"])
for branch in ("left", "right"):
    # Request only the shared inner tile, not the enclosing tiles above i.
    sch.tile("j", {"c": 4}, root=f"./{branch}")
    sch.interchange(["c", "i"], root=f"./{branch}")
schedule = sch.schedule()

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_tile_name_collision_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(schedule)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_split_tile_name_collision
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 17x63xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 17x63xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [17x63xfloat32] -> [17x63xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_tile_name_collision(_0: T.Buffer((17, 63), "float32"), C: T.Buffer((17, 63), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j in T.grid(17, 63):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  j, __t_j_2, c, __t_j_2_1, = sch.split(j, factors=[None, 2, 2, 4])
# CHECK-NEXT:  sch.reorder(j, __t_j_2, c, i, __t_j_2_1)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [6, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, __t_j_2_1, = __loops
# CHECK-NEXT:  sch.reorder(__t_j_2_1, i)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, __t_j_2_1, = __loops
# CHECK-NEXT:  sch.reorder(__t_j_2_1, i)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_tile_name_collision(_0: T.Buffer((17, 63), "float32"), C: T.Buffer((17, 63), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for j_0, j_1, j_2 in T.grid(4, 2, 2):
# CHECK-NEXT:              with T.sblock("C_i_common"):
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes()
# CHECK-NEXT:                  with T.sblock("C_i0_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for j_3, i0 in T.grid(4, 6):
# CHECK-NEXT:                          with T.sblock("C_i0"):
# CHECK-NEXT:                              v_i = T.axis.spatial(6, i0)
# CHECK-NEXT:                              v_j = T.axis.spatial(63, j_0 * 16 + j_1 * 8 + j_2 * 4 + j_3)
# CHECK-NEXT:                              T.where(((j_0 * 2 + j_1) * 2 + j_2) * 4 + j_3 < 63)
# CHECK-NEXT:                              T.reads(_0[0:6, 0:63])
# CHECK-NEXT:                              T.writes(C[0:6, 0:63])
# CHECK-NEXT:                              C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i1_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for j_3, i1 in T.grid(4, 11):
# CHECK-NEXT:                          with T.sblock("C_i1"):
# CHECK-NEXT:                              v_i = T.axis.spatial((6, 11), i1 + 6)
# CHECK-NEXT:                              v_j = T.axis.spatial(63, j_0 * 16 + j_1 * 8 + j_2 * 4 + j_3)
# CHECK-NEXT:                              T.where(((j_0 * 2 + j_1) * 2 + j_2) * 4 + j_3 < 63)
# CHECK-NEXT:                              T.reads(_0[6:11, 0:63])
# CHECK-NEXT:                              T.writes(C[6:11, 0:63])
# CHECK-NEXT:                              C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  CODE: 0
