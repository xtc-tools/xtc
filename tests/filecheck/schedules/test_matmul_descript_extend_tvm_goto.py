# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript_extend import descript_extend_scheduler

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
descript_extend_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_axis=["i", "j", "k"],
    abstract_axis_sizes=axes_sizes,
    spec={
        "DDR": {
            "j": {"parallelize": "par"},
            "k": {},
            "i": {},
            # "explore_axis_order": True,
            "pack": [("pack_B", 1, True), ("pack_A", 0, True)],
        },
        # "DDRk": {
        # },
        # "DDRi": {
        # },
        "L3": {
            "j#jL3": {},
        },
        "L2": {
            "i#iL2": {},
        },
        "L1": {
            "k#kL1": {"unroll": "k_unroll"},
        },
        "R": {"i#iR": {"unroll": None}, "j#jR": {"vectorize": None}},
    },
    sample={
        "par": None,
        "jL3": 36,
        "iL2": 128,
        "kL1": 16,
        "k_unroll": 2,
        "iR": 2,
        "jR": 6,
        "pack_B": None,
        "pack_A": None,
    },
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_tvm_goto",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

#CHECK: graph:
#CHECK-NEXT:  name: matmul
#CHECK-NEXT:  inputs:
#CHECK-NEXT:  - %0 : 512x512xfloat32
#CHECK-NEXT:  - %1 : 512x512xfloat32
#CHECK-NEXT:  outputs:
#CHECK-NEXT:  - %2 : 512x512xfloat32
#CHECK-NEXT:  nodes:
#CHECK-NEXT:  - %2: matmul(%0, %1) {name = 'C'} : [512x512xfloat32, 512x512xfloat32] -> [512x512xfloat32]
#CHECK-EMPTY:
#CHECK-NEXT:# from tvm.script import ir as I
#CHECK-NEXT:# from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT:@I.ir_module
#CHECK-NEXT:class Module:
#CHECK-NEXT:    @T.prim_func
#CHECK-NEXT:    def main(_0: T.Buffer((512, 512), "float32"), _1: T.Buffer((512, 512), "float32"), C: T.Buffer((512, 512), "float32")):
#CHECK-NEXT:        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:        for i, j in T.grid(512, 512):
#CHECK-NEXT:            C_1 = T.Buffer((262144,), data=C.data)
#CHECK-NEXT:            C_1[i * 512 + j] = T.float32(0.0)
#CHECK-NEXT:            for k in range(512):
#CHECK-NEXT:                cse_var_2: T.int32 = i * 512
#CHECK-NEXT:                cse_var_1: T.int32 = cse_var_2 + j
#CHECK-NEXT:                _0_1 = T.Buffer((262144,), data=_0.data)
#CHECK-NEXT:                _1_1 = T.Buffer((262144,), data=_1.data)
#CHECK-NEXT:                C_1[cse_var_1] = C_1[cse_var_1] + _0_1[cse_var_2 + k] * _1_1[k * 512 + j]
#CHECK-NEXT:INPS = list(obj.values())[:-1]
#CHECK-NEXT:O = obj['C']
#CHECK-NEXT:I_R0 = sch.cache_read(INPS[0], "local", [O])
#CHECK-NEXT:i, j, = O.op.axis
#CHECK-NEXT:k, = O.op.reduce_axis
#CHECK-NEXT:j, j0 = sch[O].split(j, factor=36)
#CHECK-NEXT:i, i0 = sch[O].split(i, factor=128)
#CHECK-NEXT:k, k0 = sch[O].split(k, factor=16)
#CHECK-NEXT:k0, __u_k0 = sch[O].split(k0, factor=2)
#CHECK-NEXT:i0, i1 = sch[O].split(i0, factor=2)
#CHECK-NEXT:j0, j1 = sch[O].split(j0, factor=6)
#CHECK-NEXT:j1, __v_j1 = sch[O].split(j1, factor=2)
#CHECK-NEXT:sch[O].reorder(j, k, i, j0, i0, k0, __u_k0, i1, j1, __v_j1)
#CHECK-NEXT:sch[I_R0].compute_at(sch[O], i)
#CHECK-NEXT:sch[I_R0].storage_align(I_R0.op.axis[-2], factor=1024, offset=16)
#CHECK-NEXT:sch[O].unroll(__u_k0)
#CHECK-NEXT:sch[O].unroll(i1)
#CHECK-NEXT:sch[O].unroll(j1)
#CHECK-NEXT:sch[O].vectorize(__v_j1)
#CHECK-NEXT:sch[O].parallel(j)
#CHECK-EMPTY:
#CHECK-NEXT:# from tvm.script import ir as I
#CHECK-NEXT:# from tvm.script import tir as T
#CHECK-EMPTY:
#CHECK-NEXT:@I.ir_module
#CHECK-NEXT:class Module:
#CHECK-NEXT:    @T.prim_func
#CHECK-NEXT:    def main(_0: T.Buffer((512, 512), "float32"), _1: T.Buffer((512, 512), "float32"), C: T.Buffer((512, 512), "float32")):
#CHECK-NEXT:        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
#CHECK-NEXT:        for j_outer in T.parallel(15):
#CHECK-NEXT:            _0_local = T.allocate([2048], "float32", "local")
#CHECK-NEXT:            C_1 = T.Buffer((262144,), data=C.data)
#CHECK-NEXT:            for i_outer_init, j_inner_outer_init, i_inner_outer_init in T.grid(4, 6, 64):
#CHECK-NEXT:                if T.likely(j_outer * 9 + j_inner_outer_init * 3 // 2 < 128):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                if T.likely(j_outer * 9 + (j_inner_outer_init * 3 + 1) // 2 < 128):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 2:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 2 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                if T.likely(j_outer * 9 + j_inner_outer_init * 3 // 2 < 127):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 4:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 4 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                if T.likely(j_outer * 9 + j_inner_outer_init * 3 // 2 < 128):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 512:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 512 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                if T.likely(j_outer * 9 + (j_inner_outer_init * 3 + 1) // 2 < 128):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 514:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 514 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:                if T.likely(j_outer * 9 + j_inner_outer_init * 3 // 2 < 127):
#CHECK-NEXT:                    C_1[i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 516:i_outer_init * 65536 + i_inner_outer_init * 1024 + j_outer * 36 + j_inner_outer_init * 6 + 516 + 2] = T.Broadcast(T.float32(0.0), 2)
#CHECK-NEXT:            for k_outer, i_outer in T.grid(32, 4):
#CHECK-NEXT:                _0_local_1 = T.Buffer((2048,), data=_0_local, scope="local")
#CHECK-NEXT:                for ax0, ax1 in T.grid(128, 16):
#CHECK-NEXT:                    _0_1 = T.Buffer((262144,), data=_0.data)
#CHECK-NEXT:                    _0_local_1[ax0 * 16 + ax1] = _0_1[i_outer * 65536 + ax0 * 512 + k_outer * 16 + ax1]
#CHECK-NEXT:                for j_inner_outer, i_inner_outer, k_inner_outer in T.grid(6, 64, 8):
#CHECK-NEXT:                    _1_1 = T.Buffer((262144,), data=_1.data)
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 128):
#CHECK-NEXT:                        cse_var_3: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_2: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_1: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_3 + cse_var_2
#CHECK-NEXT:                        C_1[cse_var_1:cse_var_1 + 2] = C_1[cse_var_1:cse_var_1 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_3 + cse_var_2:k_outer * 8192 + k_inner_outer * 1024 + cse_var_3 + cse_var_2 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + (j_inner_outer * 3 + 1) // 2 < 128):
#CHECK-NEXT:                        cse_var_6: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_5: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_4: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_6 + cse_var_5 + 2
#CHECK-NEXT:                        C_1[cse_var_4:cse_var_4 + 2] = C_1[cse_var_4:cse_var_4 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_6 + cse_var_5 + 2:k_outer * 8192 + k_inner_outer * 1024 + cse_var_6 + cse_var_5 + 2 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 127):
#CHECK-NEXT:                        cse_var_9: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_8: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_7: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_9 + cse_var_8 + 4
#CHECK-NEXT:                        C_1[cse_var_7:cse_var_7 + 2] = C_1[cse_var_7:cse_var_7 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_9 + cse_var_8 + 4:k_outer * 8192 + k_inner_outer * 1024 + cse_var_9 + cse_var_8 + 4 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 128):
#CHECK-NEXT:                        cse_var_12: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_11: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_10: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_12 + cse_var_11 + 512
#CHECK-NEXT:                        C_1[cse_var_10:cse_var_10 + 2] = C_1[cse_var_10:cse_var_10 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 16], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_12 + cse_var_11:k_outer * 8192 + k_inner_outer * 1024 + cse_var_12 + cse_var_11 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + (j_inner_outer * 3 + 1) // 2 < 128):
#CHECK-NEXT:                        cse_var_15: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_14: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_13: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_15 + cse_var_14 + 514
#CHECK-NEXT:                        C_1[cse_var_13:cse_var_13 + 2] = C_1[cse_var_13:cse_var_13 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 16], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_15 + cse_var_14 + 2:k_outer * 8192 + k_inner_outer * 1024 + cse_var_15 + cse_var_14 + 2 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 127):
#CHECK-NEXT:                        cse_var_18: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_17: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_16: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_18 + cse_var_17 + 516
#CHECK-NEXT:                        C_1[cse_var_16:cse_var_16 + 2] = C_1[cse_var_16:cse_var_16 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 16], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_18 + cse_var_17 + 4:k_outer * 8192 + k_inner_outer * 1024 + cse_var_18 + cse_var_17 + 4 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 128):
#CHECK-NEXT:                        cse_var_21: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_20: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_19: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_21 + cse_var_20
#CHECK-NEXT:                        C_1[cse_var_19:cse_var_19 + 2] = C_1[cse_var_19:cse_var_19 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 1], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_21 + cse_var_20 + 512:k_outer * 8192 + k_inner_outer * 1024 + cse_var_21 + cse_var_20 + 512 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + (j_inner_outer * 3 + 1) // 2 < 128):
#CHECK-NEXT:                        cse_var_24: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_23: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_22: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_24 + cse_var_23 + 2
#CHECK-NEXT:                        C_1[cse_var_22:cse_var_22 + 2] = C_1[cse_var_22:cse_var_22 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 1], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_24 + cse_var_23 + 514:k_outer * 8192 + k_inner_outer * 1024 + cse_var_24 + cse_var_23 + 514 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 127):
#CHECK-NEXT:                        cse_var_27: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_26: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_25: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_27 + cse_var_26 + 4
#CHECK-NEXT:                        C_1[cse_var_25:cse_var_25 + 2] = C_1[cse_var_25:cse_var_25 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 1], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_27 + cse_var_26 + 516:k_outer * 8192 + k_inner_outer * 1024 + cse_var_27 + cse_var_26 + 516 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 128):
#CHECK-NEXT:                        cse_var_30: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_29: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_28: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_30 + cse_var_29 + 512
#CHECK-NEXT:                        C_1[cse_var_28:cse_var_28 + 2] = C_1[cse_var_28:cse_var_28 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 17], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_30 + cse_var_29 + 512:k_outer * 8192 + k_inner_outer * 1024 + cse_var_30 + cse_var_29 + 512 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + (j_inner_outer * 3 + 1) // 2 < 128):
#CHECK-NEXT:                        cse_var_33: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_32: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_31: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_33 + cse_var_32 + 514
#CHECK-NEXT:                        C_1[cse_var_31:cse_var_31 + 2] = C_1[cse_var_31:cse_var_31 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 17], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_33 + cse_var_32 + 514:k_outer * 8192 + k_inner_outer * 1024 + cse_var_33 + cse_var_32 + 514 + 2]
#CHECK-NEXT:                    if T.likely(j_outer * 9 + j_inner_outer * 3 // 2 < 127):
#CHECK-NEXT:                        cse_var_36: T.int32 = j_outer * 36
#CHECK-NEXT:                        cse_var_35: T.int32 = j_inner_outer * 6
#CHECK-NEXT:                        cse_var_34: T.int32 = i_outer * 65536 + i_inner_outer * 1024 + cse_var_36 + cse_var_35 + 516
#CHECK-NEXT:                        C_1[cse_var_34:cse_var_34 + 2] = C_1[cse_var_34:cse_var_34 + 2] + T.Broadcast(_0_local_1[i_inner_outer * 32 + k_inner_outer * 2 + 17], 2) * _1_1[k_outer * 8192 + k_inner_outer * 1024 + cse_var_36 + cse_var_35 + 516:k_outer * 8192 + k_inner_outer * 1024 + cse_var_36 + cse_var_35 + 516 + 2]
#CHECK:CODE: 0

