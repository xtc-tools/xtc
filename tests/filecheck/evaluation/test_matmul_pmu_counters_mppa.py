# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=mppa

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from sys import platform

I, J, K, dtype = 8, 16, 32, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph

impl = Backend(graph)

sch = impl.get_scheduler()
sch.define_memory_mesh(axes={"mx": 1, "my": 1})
sch.define_processor_mesh(axes={"px": 1, "py": 1, "psx": 1, "psy": 1})
sch.tile("i", {"i1": 2})
sch.pack_at("i1", 1)
sched = sch.schedule()

comp = impl.get_compiler(
    target="mppa",
    shared_lib=True,
    dump_file="matmul_pmu_counters_mppa",
)
module = comp.compile(sched)

pmu_counters = [
    "cycles", # Host
    "instructions", # Host
    "mppa.PCC.cluster.0.pe.0", # Mppa cycles on average PEs on cluster 0 FIXME
    "mppa.PCC.cluster.0.pe.avg", # Mppa cycles on average PEs on cluster 0
    "mppa.EBE.cluster.0.pe.0", # Mppa executed bundles on PE 0 on cluster 0
    "mppa.SC.cluster.max.pe.min", # Mppa stall cycles on min PEs on max clusters
    "mppa.DDSC.cluster.0.pe.0", # Mppa data dependency stall cycles on PE 0 cluster 0
]

evaluator = module.get_evaluator(
    validate=True,
    pmu_counters=pmu_counters,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
print(f"counters: {pmu_counters}")
print(f"results: {[int(x) for x in results]}")
# CHECK:       CODE: 0
# CHECK-NEXT:  counters:
# CHECK-NEXT:  results:
