# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A")
b = O.tensor((J,), dtype, name="B")
c = O.tensor((I, 1), dtype, name="C")

with O.graph(name="add_mul") as gb:
    x = O.add(a, b, name="add")
    O.mul(x, c, name="mul")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="add")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="binary_elementwise_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: add_mul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x32xfloat32
# CHECK-NEXT:    - %1 : 32xfloat32
# CHECK-NEXT:    - %2 : 4x1xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %3: add(%0, %1) {name = 'add'} : [4x32xfloat32, 32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:    - %4: mul(%3, %2) {name = 'mul'} : [4x32xfloat32, 4x1xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def add_mul(_0: T.Buffer((4, 32), "float32"), _1: T.Buffer((32,), "float32"), _2: T.Buffer((4, 1), "float32"), mul: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          add = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("add"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j], _1[v_j])
# CHECK-NEXT:                  T.writes(add[v_i, v_j])
# CHECK-NEXT:                  add[v_i, v_j] = _0[v_i, v_j] + _1[v_j]
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("mul"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(add[v_i, v_j], _2[v_i, 0])
# CHECK-NEXT:                  T.writes(mul[v_i, v_j])
# CHECK-NEXT:                  mul[v_i, v_j] = add[v_i, v_j] * _2[v_i, 0]
# CHECK-NEXT:  O = sch.get_sblock("add")
# CHECK-NEXT:  i, j, = sch.get_loops(O)
# CHECK-NEXT:  sch.reorder(i, j)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def add_mul(_0: T.Buffer((4, 32), "float32"), _1: T.Buffer((32,), "float32"), _2: T.Buffer((4, 1), "float32"), mul: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          add = T.sblock_alloc_buffer((4, 32))
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("add"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(_0[v_i, v_j], _1[v_j])
# CHECK-NEXT:                  T.writes(add[v_i, v_j])
# CHECK-NEXT:                  add[v_i, v_j] = _0[v_i, v_j] + _1[v_j]
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              with T.sblock("mul"):
# CHECK-NEXT:                  v_i, v_j = T.axis.remap("SS", [i, j])
# CHECK-NEXT:                  T.reads(add[v_i, v_j], _2[v_i, 0])
# CHECK-NEXT:                  T.writes(mul[v_i, v_j])
# CHECK-NEXT:                  mul[v_i, v_j] = add[v_i, v_j] * _2[v_i, 0]
# CHECK-NEXT:  CODE: 0
