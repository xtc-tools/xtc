# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

from pathlib import Path

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

M, N, K, dtype = 32, 32, 8, "float32"
a = O.tensor((M, K), dtype, name="A")
b = O.tensor((K, N), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

matmul_uk = "external_matmul_uk_ixj"
impl = Backend(gb.graph)

sch = impl.get_scheduler()
sch.strip_mine("i", {"i1": 16})
sch.strip_mine("j", {"j1": 16})
sch.strip_mine("k", {"k1": 8})
sch.interchange(["i", "j", "k", "k1", "i1", "j1"])
sch.external_at("k1", matmul_uk)

print(sch.get_loop_nest().root_node.pretty_print())

schedule = sch.schedule()

microkernel_csrc = Path(__file__).parent / f"{matmul_uk}.c"
res = impl.evaluate(
    schedule,
    compiler_args=dict(
        save_temps=True,
        print_transformed_ir=True,
        csrcs=[str(microkernel_csrc)],
    ),
)

print("VALID:", isinstance(res, float))

# CHECK:       loop i
# CHECK-NEXT:    loop j
# CHECK-NEXT:      loop k
# CHECK-NEXT:        tile(k, 8)  // external(external_matmul_uk_ixj)
# CHECK-NEXT:          tile(i, 16)
# CHECK-NEXT:            tile(j, 16)
# CHECK-NEXT:              ...
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  i, i1, = sch.split(i, factors=[None, 16])
# CHECK-NEXT:  j, j1, = sch.split(j, factors=[None, 16])
# CHECK-NEXT:  k, k1, = sch.split(k, factors=[None, 8])
# CHECK-NEXT:  sch.reorder(i, j, k, k1, i1, j1)
# CHECK-NEXT:  sch = externalize_tile_below(sch, O, k1, 'external_matmul_uk_ixj')
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((32, 8), "float32"), _1: T.Buffer((8, 32), "float32"), C: T.Buffer((32, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0, j_0 in T.grid(2, 2):
# CHECK-NEXT:              for i_1_init, j_1_init in T.grid(16, 16):
# CHECK-NEXT:                  with T.sblock("C_init"):
# CHECK-NEXT:                      v_i = T.axis.spatial(32, i_0 * 16 + i_1_init)
# CHECK-NEXT:                      v_j = T.axis.spatial(32, j_0 * 16 + j_1_init)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[v_i, v_j])
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:              for k_0, k_1 in T.grid(1, 8):
# CHECK-NEXT:                  T.call_extern("int32", "external_matmul_uk_ixj", T.tvm_access_ptr(T.type_annotation("float32"), C.data, T.Add(i_0 * 16, 0) * 32 + T.Add(j_0 * 16, 0), 1024 - (T.Add(i_0 * 16, 0) * 32 + T.Add(j_0 * 16, 0)), 3), T.tvm_access_ptr(T.type_annotation("float32"), _0.data, T.Add(i_0 * 16, 0) * 8 + (k_0 * 8 + k_1), 256 - (T.Add(i_0 * 16, 0) * 8 + (k_0 * 8 + k_1)), 1), T.tvm_access_ptr(T.type_annotation("float32"), _1.data, (k_0 * 8 + k_1) * 32 + T.Add(j_0 * 16, 0), 256 - ((k_0 * 8 + k_1) * 32 + T.Add(j_0 * 16, 0)), 1), T.int64(16), T.int64(16), T.int64(32), T.int64(1), T.int64(8), T.int64(0), T.int64(0), T.int64(1))
# CHECK-NEXT:  VALID: True
