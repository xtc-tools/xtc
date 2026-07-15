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
- P: parallelize
- R: unroll=u_k
- P: unroll vectorize"""

strategy = Strategy(graph, spec)

sample = {'prt_i_0': 1, 'u_k_prt_k': 8, 'prt_j_0': 2}

strategy.generate(sch, sample)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_prp",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

#CHECK:graph:
#CHECK-NEXT:  name: matmul
#CHECK-NEXT:  inputs:
#CHECK-NEXT:  - %0 : 4x512xfloat32
#CHECK-NEXT:  - %1 : 512x32xfloat32
#CHECK-NEXT:  outputs:
#CHECK-NEXT:  - %2 : 4x32xfloat32
#CHECK-NEXT:  nodes:
#CHECK-NEXT:  - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
#CHECK-EMPTY:
#CHECK-NEXT:# from tvm.script import ir as I
#CHECK-NEXT:# from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT:@I.ir_module
#CHECK-NEXT:class Module:
#CHECK-NEXT:    @T.prim_func
#CHECK-NEXT:    def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:        for i, j in T.grid(4, 32):
#CHECK-NEXT:            C_1 = T.Buffer((128,), data=C.data)
#CHECK-NEXT:            C_1[i * 32 + j] = T.float32(0.0)
#CHECK-NEXT:            for k in range(512):
#CHECK-NEXT:                cse_var_1: T.int32 = i * 32 + j
#CHECK-NEXT:                _0_1 = T.Buffer((2048,), data=_0.data)
#CHECK-NEXT:                _1_1 = T.Buffer((16384,), data=_1.data)
#CHECK-NEXT:                C_1[cse_var_1] = C_1[cse_var_1] + _0_1[i * 512 + k] * _1_1[k * 32 + j]
#CHECK-NEXT:O = obj['C']
#CHECK-NEXT:i, j, = O.op.axis
#CHECK-NEXT:k, = O.op.reduce_axis
#CHECK-NEXT:k, __u_k = sch[O].split(k, factor=8)
#CHECK-NEXT:i, i0 = sch[O].split(i, factor=1)
#CHECK-NEXT:j, j0 = sch[O].split(j, factor=2)
#CHECK-NEXT:sch[O].reorder(i, j, k, __u_k, i0, j0)
#CHECK-NEXT:sch[O].unroll(__u_k)
#CHECK-NEXT:sch[O].unroll(i0)
#CHECK-NEXT:sch[O].vectorize(j0)
#CHECK-NEXT:sch[O].parallel(i)
#CHECK-EMPTY:
#CHECK-NEXT:# from tvm.script import ir as I
#CHECK-NEXT:# from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT:@I.ir_module
#CHECK-NEXT:class Module:
#CHECK-NEXT:    @T.prim_func
#CHECK-NEXT:    def main(_0: T.Buffer((4, 512), "float32"), _1: T.Buffer((512, 32), "float32"), C: T.Buffer((4, 32), "float32")):
#CHECK-NEXT:        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:        for i_outer in T.parallel(4):
#CHECK-NEXT:            for j_outer in range(16):
#CHECK-NEXT:                C_1 = T.Buffer((128,), data=C.data)
#CHECK-NEXT:                C_1[i_outer * 32 + j_outer * 2:i_outer * 32 + j_outer * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                for k_outer in range(64):
#CHECK-NEXT:                    cse_var_4: T.int32 = j_outer * 2
#CHECK-NEXT:                    cse_var_3: T.int32 = k_outer * 256 + cse_var_4
#CHECK-NEXT:                    cse_var_2: T.int32 = i_outer * 512 + k_outer * 8
#CHECK-NEXT:                    cse_var_1: T.int32 = i_outer * 32 + cse_var_4
#CHECK-NEXT:                    _0_1 = T.Buffer((2048,), data=_0.data)
#CHECK-NEXT:                    _1_1 = T.Buffer((16384,), data=_1.data)
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2], 2) * _1_1[cse_var_3:cse_var_3 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 1], 2) * _1_1[cse_var_3 + 32:cse_var_3 + 32 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 2], 2) * _1_1[cse_var_3 + 64:cse_var_3 + 64 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 3], 2) * _1_1[cse_var_3 + 96:cse_var_3 + 96 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 4], 2) * _1_1[cse_var_3 + 128:cse_var_3 + 128 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 5], 2) * _1_1[cse_var_3 + 160:cse_var_3 + 160 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 6], 2) * _1_1[cse_var_3 + 192:cse_var_3 + 192 + 2]
#CHECK-NEXT:                    C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_1[cse_var_2 + 7], 2) * _1_1[cse_var_3 + 224:cse_var_3 + 224 + 2]
#CHECK-NEXT:CODE: 0
