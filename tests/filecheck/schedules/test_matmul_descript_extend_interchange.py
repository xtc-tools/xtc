# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
# REQUIRES: module_xvs

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.search.strategies import Strategy_Descript as Strategy

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()

spec = """
k: interchange
i: interchange
j: interchange
i#2:
j#16:
"""

sample = {"interchange": 2}

strategy = Strategy(graph, spec)

strategy.generate(sch, sample)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_interchange",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

#CHECK: graph:
#CHECK-NEXT:   name: matmul
#CHECK-NEXT:   inputs:
#CHECK-NEXT:   - %0 : 4x512xfloat32
#CHECK-NEXT:   - %1 : 512x32xfloat32
#CHECK-NEXT:   outputs:
#CHECK-NEXT:   - %2 : 4x32xfloat32
#CHECK-NEXT:   nodes:
#CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
#CHECK-EMPTY:
#CHECK-NEXT: # from tvm.script import ir as I
#CHECK-NEXT: # from tvm.script import tirx as T
#CHECK-NEXT: # from tvm.tirx.layout import Axis
#CHECK-EMPTY:
#CHECK-NEXT: @I.ir_module
#CHECK-NEXT: class Module:
#CHECK-NEXT:     @T.prim_func(s_tir=True)
#CHECK-NEXT:     def matmul(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:         T.func_attr({"tirx.noalias": True})
#CHECK-NEXT:         # with T.sblock("root"):
#CHECK-NEXT:         for i, j, k in T.grid(4, 32, 512):
#CHECK-NEXT:             with T.sblock("C"):
#CHECK-NEXT:                 v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
#CHECK-NEXT:                 T.reads(_0[v_i, v_k], _1[v_k, v_j])
#CHECK-NEXT:                 T.writes(C[v_i, v_j])
#CHECK-NEXT:                 with T.init():
#CHECK-NEXT:                     C[v_i, v_j] = T.float32(0.0)
#CHECK-NEXT:                 C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
#CHECK-NEXT: O = sch.get_sblock("C")
#CHECK-NEXT: i, j, k, = sch.get_loops(O)
#CHECK-NEXT: i, i0, = sch.split(i, factors=[None, 2])
#CHECK-NEXT: j, j0, = sch.split(j, factors=[None, 16])
#CHECK-NEXT: sch.reorder(k, j, i, i0, j0)
#CHECK-EMPTY:
#CHECK-NEXT: # from tvm.script import ir as I
#CHECK-NEXT: # from tvm.script import tirx as T
#CHECK-NEXT: # from tvm.tirx.layout import Axis
#CHECK-EMPTY:
#CHECK-NEXT: @I.ir_module
#CHECK-NEXT: class Module:
#CHECK-NEXT:     @T.prim_func(s_tir=True)
#CHECK-NEXT:     def matmul(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:         T.func_attr({"tirx.noalias": True})
#CHECK-NEXT:         # with T.sblock("root"):
#CHECK-NEXT:         for k, j_0, i_0, i_1, j_1 in T.grid(512, 2, 2, 2, 16):
#CHECK-NEXT:             with T.sblock("C"):
#CHECK-NEXT:                 v_i = T.axis.spatial(4, i_0 * 2 + i_1)
#CHECK-NEXT:                 v_j = T.axis.spatial(32, j_0 * 16 + j_1)
#CHECK-NEXT:                 v_k = T.axis.reduce(512, k)
#CHECK-NEXT:                 T.reads(_0[v_i, v_k], _1[v_k, v_j])
#CHECK-NEXT:                 T.writes(C[v_i, v_j])
#CHECK-NEXT:                 with T.init():
#CHECK-NEXT:                     C[v_i, v_j] = T.float32(0.0)
#CHECK-NEXT:                 C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
#CHECK-NEXT: CODE: 0
