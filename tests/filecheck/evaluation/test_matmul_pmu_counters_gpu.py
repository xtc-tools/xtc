# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from sys import platform

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
#sch.interchange(["k", "i", "j", "i1", "j1"])
#sch.vectorize(["j1"]) Not supported yet for GPUs
sch.unroll({"i1": 2})
sch.parallelize(["i"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir_gpu",
)
module = comp.compile(sched)

pmu_counters = [
    "cycles",
    "instructions",
    "gpu.cycles",
    "gpu.clocks",
    "gpu.smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", # FP32 mul
    "gpu.l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_elapsed", # Random metric from https://docs.nvidia.com/cupti/main/main.html#perfworks-metric-api
]

# Linux Perf counters
if platform == "linux":
    pmu_counters += [
        "clocks",
        "mem_load_retired.l1_miss", # Can also request CPU counters
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
