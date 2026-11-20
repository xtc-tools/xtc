# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

inp = O.tensor((120,), "float32", name="A")
with O.graph(name="relu_non_dividing_tiles") as gb:
    O.relu(inp, name="C")

graph = gb.graph
print(graph)
impl = Backend(graph)
sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i"],
    spec={"i": {}, "i#50": {"unroll": 32}},
)
sched = sch.schedule()
with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/relu_nondiv_tiles_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sched)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: relu_non_dividing_tiles
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 120xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 120xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'C'} : [120xfloat32] -> [120xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_non_dividing_tiles(_0: T.Buffer((120,), "float32"), C: T.Buffer((120,), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i in range(120):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i = T.axis.spatial(120, i)
# CHECK-NEXT:                  T.reads(_0[v_i])
# CHECK-NEXT:                  T.writes(C[v_i])
# CHECK-NEXT:                  C[v_i] = T.max(T.float32(0.0), _0[v_i])
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, = sch.get_loops(O)
# CHECK-NEXT:  i, i0, = sch.split(i, factors=[None, 50])
# CHECK-NEXT:  i0, __u_i0, = sch.split(i0, factors=[None, 32])
# CHECK-NEXT:  sch.reorder(i, i0, __u_i0)
# CHECK-NEXT:  sch.unroll(__u_i0)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def relu_non_dividing_tiles(_0: T.Buffer((120,), "float32"), C: T.Buffer((120,), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0, i_1_0 in T.grid(3, 2):
# CHECK-NEXT:              for i_1_1 in T.unroll(32):
# CHECK-NEXT:                  with T.sblock("C"):
# CHECK-NEXT:                      v_i = T.axis.spatial(120, i_0 * 50 + (i_1_0 * 32 + i_1_1))
# CHECK-NEXT:                      T.where(i_0 * 50 + (i_1_0 * 32 + i_1_1) < 120 and i_1_0 * 32 + i_1_1 < 50)
# CHECK-NEXT:                      T.reads(_0[v_i])
# CHECK-NEXT:                      T.writes(C[v_i])
# CHECK-NEXT:                      C[v_i] = T.max(T.float32(0.0), _0[v_i])
# CHECK-NEXT:  CODE: 0
