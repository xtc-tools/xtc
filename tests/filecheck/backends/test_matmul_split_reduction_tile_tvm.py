# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 19, 11, 23, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")
with O.graph(name="matmul_split_reduction_tile") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)
impl = Backend(graph)
sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i", "j", "k"],
    spec={
        "j": {},
        "k": {},
        "i[:12]": {"k#6": {}},
        "i[12:]": {"k#6": {}},
    },
)
sched = sch.schedule()
with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=f"{tmp}/matmul_split_reduction_tile_tvm",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sched)
    executor = module.get_executor(validate=True)
    res = executor.execute()
    print(f"CODE: {res}")
    assert res == 0
# CHECK:       graph:
# CHECK-NEXT:    name: matmul_split_reduction_tile
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 19x23xfloat32
# CHECK-NEXT:    - %1 : 23x11xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 19x11xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [19x23xfloat32, 23x11xfloat32] -> [19x11xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_split_reduction_tile(_0: T.Buffer((19, 23), "float32"), _1: T.Buffer((23, 11), "float32"), C: T.Buffer((19, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i, j, k in T.grid(19, 11, 23):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(_0[v_i, v_k], _1[v_k, v_j])
# CHECK-NEXT:                  T.writes(C[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  k, k0, = sch.split(k, factors=[None, 6])
# CHECK-NEXT:  sch.reorder(j, k, i, k0)
# CHECK-NEXT:  sch, O_SPLITS = loop_partition_rebased(sch, i, [12, None])
# CHECK-NEXT:  O_i_0 = sch.get_sblock(O_SPLITS[0])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_0)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_0'
# CHECK-NEXT:  i, k0, = __loops
# CHECK-NEXT:  sch.reorder(i, k0)
# CHECK-NEXT:  O_i_1 = sch.get_sblock(O_SPLITS[1])
# CHECK-NEXT:  __loops = sch.get_loops(O_i_1)
# CHECK-NEXT:  assert len(__loops) == 2, 'unexpected physical loop chain for O_i_1'
# CHECK-NEXT:  i, k0, = __loops
# CHECK-NEXT:  sch.reorder(i, k0)
# CHECK-NEXT:  sch = decompose_reduction_initializers(sch)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul_split_reduction_tile(_0: T.Buffer((19, 23), "float32"), _1: T.Buffer((23, 11), "float32"), C: T.Buffer((19, 11), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for j in range(11):
# CHECK-NEXT:              for i0_init in range(12):
# CHECK-NEXT:                  with T.sblock("C_i0_init"):
# CHECK-NEXT:                      v_i_init, v_j_init = T.axis.remap("SS", [i0_init, j])
# CHECK-NEXT:                      T.where(T.Add(T.Mul(0, 6), 0) < 23)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[0:12, 0:11])
# CHECK-NEXT:                      C[v_i_init, v_j_init] = T.float32(0.0)
# CHECK-NEXT:              for i1_init in range(7):
# CHECK-NEXT:                  with T.sblock("C_i1_init"):
# CHECK-NEXT:                      v_i_init = T.axis.spatial((12, 7), i1_init + 12)
# CHECK-NEXT:                      v_j_init = T.axis.spatial(11, j)
# CHECK-NEXT:                      T.where(T.Add(T.Mul(0, 6), 0) < 23)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[12:7, 0:11])
# CHECK-NEXT:                      C[v_i_init, v_j_init] = T.float32(0.0)
# CHECK-NEXT:              for k_0 in range(4):
# CHECK-NEXT:                  with T.sblock("C_i_common"):
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes()
# CHECK-NEXT:                      with T.sblock("C_i0_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for i0, k_1 in T.grid(12, 6):
# CHECK-NEXT:                              with T.sblock("C_i0_update"):
# CHECK-NEXT:                                  v_i, v_j = T.axis.remap("SS", [i0, j])
# CHECK-NEXT:                                  v_k = T.axis.reduce(23, k_0 * 6 + k_1)
# CHECK-NEXT:                                  T.where(k_0 * 6 + k_1 < 23)
# CHECK-NEXT:                                  T.reads(_0[0:12, 0:23], _1[0:23, 0:11], C[0:12, 0:11])
# CHECK-NEXT:                                  T.writes(C[0:12, 0:11])
# CHECK-NEXT:                                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:                      with T.sblock("C_i1_partition"):
# CHECK-NEXT:                          T.reads()
# CHECK-NEXT:                          T.writes()
# CHECK-NEXT:                          for i1, k_1 in T.grid(7, 6):
# CHECK-NEXT:                              with T.sblock("C_i1_update"):
# CHECK-NEXT:                                  v_i = T.axis.spatial((12, 7), i1 + 12)
# CHECK-NEXT:                                  v_j = T.axis.spatial(11, j)
# CHECK-NEXT:                                  v_k = T.axis.reduce(23, k_0 * 6 + k_1)
# CHECK-NEXT:                                  T.where(k_0 * 6 + k_1 < 23)
# CHECK-NEXT:                                  T.reads(_0[12:7, 0:23], _1[0:23, 0:11], C[12:7, 0:11])
# CHECK-NEXT:                                  T.writes(C[12:7, 0:11])
# CHECK-NEXT:                                  C[v_i, v_j] = C[v_i, v_j] + _0[v_i, v_k] * _1[v_k, v_j]
# CHECK-NEXT:  CODE: 0
