# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
# REQUIRES: module_xvs

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 512, 512, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()
axes_sizes = {"i": I, "j": J, "k": K}
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i", "j", "k"],
    abstract_dim_sizes=axes_sizes,
    abstract_matrix=["A", "B", "C"],
    spec="""
j: parallelize=par
k:
i:
$: pack(B) pack(A)
j#jL3:
i#iL2:
k#kL1: unroll=k_unroll
i#iR: unroll
j#jR: vectorize
    """,
    sample={
        "par": 1,
        "jL3": 36,
        "iL2": 128,
        "kL1": 16,
        "k_unroll": 2,
        "iR": 2,
        "jR": 6,
    },
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_tvm_yaml_goto",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: graph:
# CHECK-NEXT:  name: matmul
# CHECK-NEXT:  inputs:
# CHECK-NEXT:  - %0 : 512x512xfloat32
# CHECK-NEXT:  - %1 : 512x512xfloat32
# CHECK-NEXT:  outputs:
# CHECK-NEXT:  - %2 : 512x512xfloat32
# CHECK-NEXT:  nodes:
# CHECK-NEXT:  - %2: matmul(%0, %1) {name = 'C'} : [512x512xfloat32, 512x512xfloat32] -> [512x512xfloat32]
# CHECK-EMPTY:
# CHECK-NEXT:# from tvm.script import ir as I
# CHECK-NEXT:# from tvm.script import tirx as T
# CHECK-NEXT:# from tvm.tirx.layout import Axis
# CHECK-EMPTY:
# CHECK-NEXT:@I.ir_module
# CHECK-NEXT:class Module:
# CHECK-NEXT:    @T.prim_func(s_tir=True)
# CHECK-NEXT:    def matmul(_0: T.Buffer((512, 512), "float32"), _1: T.Buffer((512, 512), "float32"), C: T.Buffer((512, 512), "float32")):
# CHECK-NEXT:        T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:        # with T.sblock("root"):
# CHECK-NEXT:        for i, j, k in T.grid(512, 512, 512):
# CHECK-NEXT:            with T.sblock("C"):
# CHECK-NEXT:                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                T.writes(C[v_i, v_j])
# CHECK-NEXT:                with T.init():
# CHECK-NEXT:                    C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:O = sch.get_sblock("C")
# CHECK-NEXT:i, j, k, = sch.get_loops(O)
# CHECK-NEXT:I_R0 = sch.cache_read(O, 0, "global")
# CHECK-NEXT:i, i0, i1, = sch.split(i, factors=[None, 64, 2])
# CHECK-NEXT:j, j0, j1, __v_j1, = sch.split(j, factors=[None, 12, 3, 2])
# CHECK-NEXT:k, k0, __u_k0, = sch.split(k, factors=[None, 8, 2])
# CHECK-NEXT:sch.reorder(j, k, i, j0, i0, k0, __u_k0, i1, j1, __v_j1)
# CHECK-NEXT:sch.compute_at(I_R0, i)
# CHECK-NEXT:sch.unroll(__u_k0)
# CHECK-NEXT:sch.unroll(i1)
# CHECK-NEXT:sch.unroll(j1)
# CHECK-NEXT:sch.vectorize(__v_j1)
# CHECK-NEXT:sch.parallel(j)
# CHECK-EMPTY:
# CHECK-NEXT:# from tvm.script import ir as I
# CHECK-NEXT:# from tvm.script import tirx as T
# CHECK-NEXT:# from tvm.tirx.layout import Axis
# CHECK-EMPTY:
# CHECK-NEXT:@I.ir_module
# CHECK-NEXT:class Module:
# CHECK-NEXT:    @T.prim_func(s_tir=True)
# CHECK-NEXT:    def matmul(_0: T.Buffer((512, 512), "float32"), _1: T.Buffer((512, 512), "float32"), C: T.Buffer((512, 512), "float32")):
# CHECK-NEXT:        T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:        # with T.sblock("root"):
# CHECK-NEXT:        _0_global = T.sblock_alloc_buffer((512, 512))
# CHECK-NEXT:        for j_0 in T.parallel(8):
# CHECK-NEXT:            for k_0, i_0 in T.grid(32, 4):
# CHECK-NEXT:                for ax0, ax1 in T.grid(128, 16):
# CHECK-NEXT:                    with T.sblock("_0_global"):
# CHECK-NEXT:                        v0 = T.axis.spatial(512, i_0 * 128 + ax0)
# CHECK-NEXT:                        v1 = T.axis.spatial(512, k_0 * 16 + ax1)
# CHECK-NEXT:                        T.reads(_0[v0, v1])
# CHECK-NEXT:                        T.writes(_0_global[v0, v1])
# CHECK-NEXT:                        _0_global[v0, v1] = _0[v0, v1]
# CHECK-NEXT:                for j_1, i_1, k_1 in T.grid(12, 64, 8):
# CHECK-NEXT:                    for k_2 in T.unroll(2):
# CHECK-NEXT:                        for i_2 in T.unroll(2):
# CHECK-NEXT:                            for j_2 in T.unroll(3):
# CHECK-NEXT:                                for j_3 in T.vectorized(2):
# CHECK-NEXT:                                    with T.sblock("C"):
# CHECK-NEXT:                                        v_i = T.axis.spatial(512, i_0 * 128 + i_1 * 2 + i_2)
# CHECK-NEXT:                                        v_j = T.axis.spatial(512, j_0 * 72 + j_1 * 6 + j_2 * 2 + j_3)
# CHECK-NEXT:                                        v_k = T.axis.reduce(512, k_0 * 16 + k_1 * 2 + k_2)
# CHECK-NEXT:                                        T.where(((j_0 * 12 + j_1) * 3 + j_2) * 2 + j_3 < 512)
# CHECK-NEXT:                                        T.reads(_0_global[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                                        T.writes(C[v_i, v_j])
# CHECK-NEXT:                                        with T.init():
# CHECK-NEXT:                                            C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                                        C[v_i, v_j] = C[v_i, v_j] + _0_global[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:CODE: 0
