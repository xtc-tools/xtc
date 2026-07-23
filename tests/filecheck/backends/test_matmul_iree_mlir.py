# RUN: python %s 2>&1 | filecheck %s

# generate_mlir() does not invoke IREE, so this test needs no IREE install: it
# checks the linalg-on-tensors MLIR handed to IREE, with the schedule carried as
# an iree_codegen.compilation_info attribute on the matmul.

import xtc.graphs.xtc.op as O
from xtc.backends.iree import Backend

I, J, K, dtype = 64, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

impl = Backend(gb.graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 16})
sch.tile("j", {"j1": 16})
sch.tile("k", {"k1": 16})
sch.vectorize(["j1"])
sched = sch.schedule()

print(impl.get_compiler().generate_mlir(sched))

# The compilation_info is a single (long) attribute on the matmul; CHECK-SAME
# keeps matching the same line, so each tiling level reads on its own row.

# CHECK:      func.func @matmul(%{{.*}} : tensor<64x64xf32>, %{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
# CHECK:      linalg.fill {__xtc_id_C_0_}
# CHECK:      linalg.matmul {__xtc_id_C_, compilation_info = #iree_codegen.compilation_info<
# CHECK-SAME:   lowering_config = #iree_cpu.lowering_config<
# CHECK-SAME:     distribution = [0, 0, 0],
# CHECK-SAME:     cache_parallel = [16, 0, 0],
# CHECK-SAME:     cache_reduction = [0, 0, 16],
# CHECK-SAME:     vector_common_parallel = [0, 16, 0],
# CHECK-SAME:     vector_reduction = [0, 0, 1]>,
# CHECK-SAME:   translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>>
# CHECK:      func.return %{{.*}} : tensor<64x64xf32>
