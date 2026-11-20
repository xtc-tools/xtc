# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

A = O.tensor((19, 11), "float32", name="A")
with O.graph(name="relu_split_cache") as gb:
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
        "j": {"buffer": "global", "pack": (0, "global", False)},
        "i[:10]": {},
        "i[10:]": {},
    },
)
sched = sch.schedule()
with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_split_cache_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sched)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_split_cache
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 19x11xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 19x11xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [19x11xfloat32] -> [19x11xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_cache(_0: T.Buffer((19, 11), "float32"), C: T.Buffer((19, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j in T.grid(19, 11):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  C[v_i, v_j] = T.max(T.float32(0.0), _0[v_i, v_j])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  I_R0 = sch.cache_read(O, 0, "global")
# CHECK-NEXT:  O_W0 = sch.cache_write(O, 0, "global")
# CHECK-NEXT:  sch.reorder(j, i)
# CHECK-NEXT:  sch.reverse_compute_at(O_W0, j)
# CHECK-NEXT:  sch.compute_at(I_R0, j)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [10, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 1, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, = __loops
# CHECK-NEXT:  sch.reorder(i)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 1, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, = __loops
# CHECK-NEXT:  sch.reorder(i)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_split_cache(_0: T.Buffer((19, 11), "float32"), C: T.Buffer((19, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          _0_global = T.sblock_alloc_buffer((19, 11))
# CHECK-NEXT:          C_global = T.sblock_alloc_buffer((19, 11))
# CHECK-NEXT:          for j in range(11):
# CHECK-NEXT:              for ax0 in range(19):
# CHECK-NEXT:                  with T.sblock("_0_global"):
# CHECK-NEXT:                      v0, v1 = T.axis.remap("SS", [ax0, j])
# CHECK-NEXT:                      T.reads(_0[v0:v0 + (1 - v0), v1:v1 + (1 - v1)])
# CHECK-NEXT:                      T.writes(_0_global[v0:v0 + (1 - v0), v1:v1 + (1 - v1)])
# CHECK-NEXT:                      _0_global[v0, v1] = _0[v0, v1]
# CHECK-NEXT:              with T.sblock("C_i_common"):
# CHECK-NEXT:                  T.reads()
# CHECK-NEXT:                  T.writes()
# CHECK-NEXT:                  with T.sblock("C_i0_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i0 in range(10):
# CHECK-NEXT:                          with T.sblock("C_i0"):
# CHECK-NEXT:                              v_i, v_j = T.axis.remap("SS", [i0, j])
# CHECK-NEXT:                              T.reads(_0_global[0:10, 0:11])
# CHECK-NEXT:                              T.writes(C_global[0:10, 0:11])
# CHECK-NEXT:                              C_global[v_i, v_j] = T.max(T.float32(0.0), _0_global[v_i, v_j])
# CHECK-NEXT:                  with T.sblock("C_i1_partition"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      for i1 in range(9):
# CHECK-NEXT:                          with T.sblock("C_i1"):
# CHECK-NEXT:                              v_i = T.axis.spatial((10, 9), i1 + 10)
# CHECK-NEXT:                              v_j = T.axis.spatial(11, j)
# CHECK-NEXT:                              T.reads(_0_global[10:9, 0:11])
# CHECK-NEXT:                              T.writes(C_global[10:9, 0:11])
# CHECK-NEXT:                              C_global[v_i, v_j] = T.max(T.float32(0.0), _0_global[v_i, v_j])
# CHECK-NEXT:              for ax0 in range(19):
# CHECK-NEXT:                  with T.sblock("C_global"):
# CHECK-NEXT:                      v0, v1 = T.axis.remap("SS", [ax0, j])
# CHECK-NEXT:                      T.reads(C_global[v0:v0 + (1 - v0), v1:v1 + (1 - v1)])
# CHECK-NEXT:                      T.writes(C[v0:v0 + (1 - v0), v1:v1 + (1 - v1)])
# CHECK-NEXT:                      C[v0, v1] = C_global[v0, v1]
# CHECK-NEXT:  CODE: 0
