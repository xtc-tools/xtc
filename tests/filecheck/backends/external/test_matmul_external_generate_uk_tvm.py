# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import argparse
from pathlib import Path
from importlib import import_module
import tempfile
import shutil

import xtc.graphs.xtc.op as O
from xtc.runtimes.host import HostRuntime
from utils.generate_uk import generate_uk_ixj

parser = argparse.ArgumentParser("external uk test")
parser.add_argument("--backend", default="tvm")
parser.add_argument("--uk", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--perf", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

backend = import_module(f"xtc.backends.{args.backend}")
        
M, N, K, dtype = 512, 1024, 128, "float32"
a = O.tensor((M, K), dtype, name="A")
b = O.tensor((K, N), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

I0, I1, I2 = 256, 16, 8
J0, J1, J2 = 1024, 64, 16
K0 = 16

uk_csrc_template = str(Path(__file__).parent / "external_matmul_uk_ixjxk.c.jinja")
tdir = tempfile.mkdtemp(dir=".")
uk_csrc_fmt = str(Path(tdir) / "external_matmul_uk_ixjxk_{i}x{j}.c")
uk_csrc = generate_uk_ixj(uk_csrc_fmt, uk_csrc_template, i=I2, j=J2)
uk_symbol = Path(uk_csrc).stem

impl = backend.Backend(gb.graph)
sch = impl.get_scheduler()
sch.strip_mine("i", {"i0": I0, "i1": I1, "i2": I2})
sch.strip_mine("j", {"j0": J0, "j1": J1, "j2": J2})
sch.strip_mine("k", {"k0": K0})
#sch.buffer_at("j0")
if args.uk:
    sch.interchange(["i", "j", "i0", "j0", "k", "i1", "j1", "i2", "j2", "k0"])
    sch.external_at("j1", uk_symbol)
else:
    sch.interchange(["i", "j", "i0", "j0", "k", "i1", "j1", "k0", "i2", "j2"])
    sch.vectorize(["j2"])
    sch.unroll({"k0": K0, "i2": I2})
print(sch.get_loop_nest().root_node.pretty_print())

res = impl.evaluate(
    sch.schedule(),
    compiler_args=dict(
        emit_c=True,
        save_temps=True,
        print_transformed_ir=True,
        **(dict(csrcs=[str(uk_csrc)]) if args.uk else {}),
    ),
)

assert isinstance(res, float), f"{res}"

if args.perf:
    flops = HostRuntime.get().evaluate_flops(dtype)
    peak_time = (M*N*K)/flops
    print(f"Flops {flops/1e9:.2f} GHz")
    print(f"Time {res*1000:.3f} ms")
    print(f"Peak time {peak_time*1000:.3f} ms")
    print(f"Peak perf {peak_time/res*100:.2f} %")

shutil.rmtree(tdir)

# CHECK:       loop i
# CHECK-NEXT:    loop j
# CHECK-NEXT:      tile(i, 256)
# CHECK-NEXT:        tile(j, 1024)
# CHECK-NEXT:          loop k
# CHECK-NEXT:            tile(i, 16)
# CHECK-NEXT:              tile(j, 64)  // external(external_matmul_uk_ixjxk_8x16)
# CHECK-NEXT:                tile(i, 8)
# CHECK-NEXT:                  tile(j, 16)
# CHECK-NEXT:                    tile(k, 16)
# CHECK-NEXT:                      ...
# CHECK-NEXT:  O = sch.get_sblock("C")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  i, i0, i1, i2, = sch.split(i, factors=[None, 16, 2, 8])
# CHECK-NEXT:  j, j0, j1, j2, = sch.split(j, factors=[None, 16, 4, 16])
# CHECK-NEXT:  k, k0, = sch.split(k, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(i, j, i0, j0, k, i1, j1, i2, j2, k0)
# CHECK-NEXT:  sch = externalize_tile_below(sch, O, j1, 'external_matmul_uk_ixjxk_8x16')
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def matmul(_0: T.Buffer((512, 128), "float32"), _1: T.Buffer((128, 1024), "float32"), C: T.Buffer((512, 1024), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for i_0, j_0, i_1, j_1 in T.grid(2, 1, 16, 16):
# CHECK-NEXT:              for i_2_init, j_2_init, i_3_init, j_3_init in T.grid(2, 4, 8, 16):
# CHECK-NEXT:                  with T.sblock("C_init"):
# CHECK-NEXT:                      v_i = T.axis.spatial(512, i_0 * 256 + i_1 * 16 + i_2_init * 8 + i_3_init)
# CHECK-NEXT:                      v_j = T.axis.spatial(1024, j_0 * 1024 + j_1 * 64 + j_2_init * 16 + j_3_init)
# CHECK-NEXT:                      T.reads()
# CHECK-NEXT:                      T.writes(C[v_i, v_j])
# CHECK-NEXT:                      C[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:              for k_0, i_2, j_2 in T.grid(8, 2, 4):
# CHECK-NEXT:                  T.call_extern("int32", "external_matmul_uk_ixjxk_8x16", T.tvm_access_ptr(T.type_annotation("float32"), C.data, T.Add(i_0 * 256 + i_1 * 16 + i_2 * 8, 0) * 1024 + T.Add(j_0 * 1024 + j_1 * 64 + j_2 * 16, 0), 524288 - (T.Add(i_0 * 256 + i_1 * 16 + i_2 * 8, 0) * 1024 + T.Add(j_0 * 1024 + j_1 * 64 + j_2 * 16, 0)), 3), T.tvm_access_ptr(T.type_annotation("float32"), _0.data, T.Add(i_0 * 256 + i_1 * 16 + i_2 * 8, 0) * 128 + T.Add(k_0 * 16, 0), 65536 - (T.Add(i_0 * 256 + i_1 * 16 + i_2 * 8, 0) * 128 + T.Add(k_0 * 16, 0)), 1), T.tvm_access_ptr(T.type_annotation("float32"), _1.data, T.Add(k_0 * 16, 0) * 1024 + T.Add(j_0 * 1024 + j_1 * 64 + j_2 * 16, 0), 131072 - (T.Add(k_0 * 16, 0) * 1024 + T.Add(j_0 * 1024 + j_1 * 64 + j_2 * 16, 0)), 1), T.int64(8), T.int64(16), T.int64(16), T.int64(1024), T.int64(1), T.int64(0), T.int64(128), T.int64(0), T.int64(1), T.int64(0), T.int64(1), T.int64(1024))
