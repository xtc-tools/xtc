# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 64, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    p = O.relu(a, name="relu")
    O.matmul(p, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 8, "i2": 4})
sch.tile("j", {"j1": 32, "j2": 16})
sch.tile("k", {"k1": 16})
sch.interchange(["i", "j", "k", "i1", "j1", "k1", "i2", "j2"])
sch.buffer_at("j")
sch.pack_at("k", 1, pad=True)
sch.fuse_producer_at("k", 0)
sch.vectorize(["j2"])
sch.unroll({"i2": 4})
sch.parallelize(["i", "j"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="relu_matmul_tvm_fused",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")


# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 64x64xfloat32
# CHECK-NEXT:    - %1 : 64x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 64x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: relu(%0) {name = 'relu'} : [64x64xfloat32] -> [64x64xfloat32]
# CHECK-NEXT:    - %3: matmul(%2, %1) {name = 'C'} : [64x64xfloat32, 64x64xfloat32] -> [64x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          T_reshape = T.allocate([4096], "float32", "global")
# CHECK-NEXT:          T_reshape_1 = T.Buffer((4096,), data=T_reshape)
# CHECK-NEXT:          for ax0 in range(4096):
# CHECK-NEXT:              _0_1 = T.Buffer((4096,), data=_0.data)
# CHECK-NEXT:              T_reshape_1[ax0] = _0_1[ax0]
# CHECK-NEXT:          for i in range(4096):
# CHECK-NEXT:              T_reshape_2 = T.Buffer((4096,), data=T_reshape)
# CHECK-NEXT:              T_reshape_2[i] = T.max(T.float32(0.0), T_reshape_1[i])
# CHECK-NEXT:          for i, j in T.grid(64, 64):
# CHECK-NEXT:              C_1 = T.Buffer((4096,), data=C.data)
# CHECK-NEXT:              C_1[i * 64 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(64):
# CHECK-NEXT:                  cse_var_2: T.int32 = i * 64
# CHECK-NEXT:                  cse_var_1: T.int32 = cse_var_2 + j
# CHECK-NEXT:                  T_reshape_2 = T.Buffer((4096,), data=T_reshape)
# CHECK-NEXT:                  _1_1 = T.Buffer((4096,), data=_1.data)
# CHECK-NEXT:                  C_1[cse_var_1] = C_1[cse_var_1] + T_reshape_2[cse_var_2 + k] * _1_1[k * 64 + j]
# CHECK-NEXT:  INPS = list(obj.values())[:-1]
# CHECK-NEXT:  O = obj['C']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "global")
# CHECK-NEXT:  I_R1 = sch.cache_read(INPS[1], "global", [O_W0])
# CHECK-NEXT:  I_F0 = O_W0.op.input_tensors[0]
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  i, i_ = sch[O].split(i, factor=8)
# CHECK-NEXT:  j, j_ = sch[O].split(j, factor=32)
# CHECK-NEXT:  sch[O].reorder(i, j, i_, j_)
# CHECK-NEXT:  j = sch[O].fuse(i, j)
# CHECK-NEXT:  sch[O].parallel(j)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], j)
# CHECK-NEXT:  i, j, = O_W0.op.axis
# CHECK-NEXT:  k, = O_W0.op.reduce_axis
# CHECK-NEXT:  i1 = i
# CHECK-NEXT:  j1 = j
# CHECK-NEXT:  k, k1 = sch[O_W0].split(k, factor=16)
# CHECK-NEXT:  i1, i2 = sch[O_W0].split(i1, factor=4)
# CHECK-NEXT:  j1, j2 = sch[O_W0].split(j1, factor=16)
# CHECK-NEXT:  sch[O_W0].reorder(k, i1, j1, k1, i2, j2)
# CHECK-NEXT:  sch[I_R1].compute_at(sch[O_W0], k)
# CHECK-NEXT:  sch[I_R1].storage_align(I_R1.op.axis[-2], factor=1024, offset=16)
# CHECK-NEXT:  sch[I_F0].compute_at(sch[O_W0], k)
# CHECK-NEXT:  sch[O_W0].unroll(i2)
# CHECK-NEXT:  sch[O_W0].vectorize(j2)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          T_reshape = T.allocate([4096], "float32", "global")
# CHECK-NEXT:          T_reshape_1 = T.Buffer((4096,), data=T_reshape)
# CHECK-NEXT:          for ax0 in range(4096):
# CHECK-NEXT:              _0_1 = T.Buffer((4096,), data=_0.data)
# CHECK-NEXT:              T_reshape_1[ax0] = _0_1[ax0]
# CHECK-NEXT:          T_reshape_2 = T.Buffer((4096,), data=T_reshape)
# CHECK-NEXT:          for i in range(4096):
# CHECK-NEXT:              T_reshape_2[i] = T.max(T.float32(0.0), T_reshape_1[i])
# CHECK-NEXT:          for i_outer_j_outer_fused in T.parallel(16):
# CHECK-NEXT:              C_global = T.allocate([256], "float32", "global")
# CHECK-NEXT:              T_reshape_3 = T.allocate([128], "float32", "global")
# CHECK-NEXT:              _1_global = T.allocate([16640], "float32", "global")
# CHECK-NEXT:              C_global_1 = T.Buffer((256,), data=C_global)
# CHECK-NEXT:              for i_c_outer_init, j_c_outer_init in T.grid(2, 2):
# CHECK-NEXT:                  cse_var_1: T.int32 = i_c_outer_init * 128 + j_c_outer_init * 16
# CHECK-NEXT:                  C_global_1[cse_var_1:cse_var_1 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:                  C_global_1[cse_var_1 + 32:cse_var_1 + 32 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:                  C_global_1[cse_var_1 + 64:cse_var_1 + 64 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:                  C_global_1[cse_var_1 + 96:cse_var_1 + 96 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              for k_outer in range(4):
# CHECK-NEXT:                  T_reshape_4 = T.Buffer((128,), data=T_reshape_3)
# CHECK-NEXT:                  for ax0, ax1 in T.grid(8, 16):
# CHECK-NEXT:                      T_reshape_4[ax0 * 16 + ax1] = T_reshape_2[i_outer_j_outer_fused // 2 * 512 + ax0 * 64 + k_outer * 16 + ax1]
# CHECK-NEXT:                  _1_global_1 = T.Buffer((16640,), data=_1_global)
# CHECK-NEXT:                  for ax0, ax1 in T.grid(16, 32):
# CHECK-NEXT:                      _1_1 = T.Buffer((4096,), data=_1.data)
# CHECK-NEXT:                      _1_global_1[ax0 * 1040 + ax1] = _1_1[k_outer * 1024 + ax0 * 64 + i_outer_j_outer_fused % 2 * 32 + ax1]
# CHECK-NEXT:                  for i_c_outer, j_c_outer, k_inner in T.grid(2, 2, 16):
# CHECK-NEXT:                      cse_var_8: T.int32 = j_c_outer * 16
# CHECK-NEXT:                      cse_var_7: T.int32 = i_c_outer * 64 + k_inner
# CHECK-NEXT:                      cse_var_6: T.int32 = k_inner * 1040 + cse_var_8
# CHECK-NEXT:                      cse_var_5: T.int32 = i_c_outer * 128 + cse_var_8
# CHECK-NEXT:                      cse_var_4: T.int32 = cse_var_5 + 96
# CHECK-NEXT:                      cse_var_3: T.int32 = cse_var_5 + 64
# CHECK-NEXT:                      cse_var_2: T.int32 = cse_var_5 + 32
# CHECK-NEXT:                      C_global_1[cse_var_5:cse_var_5 + 16] = C_global_1[cse_var_5:cse_var_5 + 16] + T.Broadcast(T_reshape_4[cse_var_7], 16) * _1_global_1[cse_var_6:cse_var_6 + 16]
# CHECK-NEXT:                      C_global_1[cse_var_2:cse_var_2 + 16] = C_global_1[cse_var_2:cse_var_2 + 16] + T.Broadcast(T_reshape_4[cse_var_7 + 16], 16) * _1_global_1[cse_var_6:cse_var_6 + 16]
# CHECK-NEXT:                      C_global_1[cse_var_3:cse_var_3 + 16] = C_global_1[cse_var_3:cse_var_3 + 16] + T.Broadcast(T_reshape_4[cse_var_7 + 32], 16) * _1_global_1[cse_var_6:cse_var_6 + 16]
# CHECK-NEXT:                      C_global_1[cse_var_4:cse_var_4 + 16] = C_global_1[cse_var_4:cse_var_4 + 16] + T.Broadcast(T_reshape_4[cse_var_7 + 48], 16) * _1_global_1[cse_var_6:cse_var_6 + 16]
# CHECK-NEXT:              for i_inner, j_inner in T.grid(8, 32):
# CHECK-NEXT:                  C_1 = T.Buffer((4096,), data=C.data)
# CHECK-NEXT:                  C_1[i_outer_j_outer_fused // 2 * 512 + i_inner * 64 + i_outer_j_outer_fused % 2 * 32 + j_inner] = C_global_1[i_inner * 32 + j_inner]
# CHECK-NEXT:  CODE: 0
