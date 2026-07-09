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
#CHECK-NEXT: # from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT: @I.ir_module
#CHECK-NEXT: class Module:
#CHECK-NEXT:     @T.prim_func
#CHECK-NEXT:     def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:         T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:         for i, j in T.grid(4, 32):
#CHECK-NEXT:             C_1 = T.Buffer((128,), data=C.data)
#CHECK-NEXT:             C_1[i * 32 + j] = T.float32(0.0)
#CHECK-NEXT:             for k in range(512):
#CHECK-NEXT:                 cse_var_1: T.int32 = i * 32 + j
#CHECK-NEXT:                 _0_1 = T.Buffer((2048,), data=_0.data)
#CHECK-NEXT:                 _1_1 = T.Buffer((16384,), data=_1.data)
#CHECK-NEXT:                 C_1[cse_var_1] = C_1[cse_var_1] + _0_1[i * 512 + k] * _1_1[k * 32 + j]
#CHECK-NEXT: O = obj['C']
#CHECK-NEXT: i, j, = O.op.axis
#CHECK-NEXT: k, = O.op.reduce_axis
#CHECK-NEXT: i, i0 = sch[O].split(i, factor=2)
#CHECK-NEXT: j, j0 = sch[O].split(j, factor=16)
#CHECK-NEXT: sch[O].reorder(k, j, i, i0, j0)
#CHECK-EMPTY:
#CHECK-NEXT: # from tvm.script import ir as I
#CHECK-NEXT: # from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT: @I.ir_module
#CHECK-NEXT: class Module:
#CHECK-NEXT:     @T.prim_func
#CHECK-NEXT:     def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:         T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:         C_1 = T.Buffer((128,), data=C.data)
#CHECK-NEXT:         for j_outer_init, i_outer_init, i_inner_init, j_inner_init in T.grid(2, 2, 2, 16):
#CHECK-NEXT:             C_1[i_outer_init * 64 + i_inner_init * 32 + j_outer_init * 16 + j_inner_init] = T.float32(0.0)
#CHECK-NEXT:         for k, j_outer, i_outer, i_inner, j_inner in T.grid(512, 2, 2, 2, 16):
#CHECK-NEXT:             cse_var_2: T.int32 = j_outer * 16
#CHECK-NEXT:             cse_var_1: T.int32 = i_outer * 64 + i_inner * 32 + cse_var_2 + j_inner
#CHECK-NEXT:             _0_1 = T.Buffer((2048,), data=_0.data)
#CHECK-NEXT:             _1_1 = T.Buffer((16384,), data=_1.data)
#CHECK-NEXT:             C_1[cse_var_1] = C_1[cse_var_1] + _0_1[i_outer * 1024 + i_inner * 512 + k] * _1_1[k * 32 + cse_var_2 + j_inner]
#CHECK-NEXT: CODE: 0
