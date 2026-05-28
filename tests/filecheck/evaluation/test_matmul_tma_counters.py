# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from sys import platform

#I, J, K, dtype = 4, 32, 512, "float32"             # small
I, J, K, dtype = 1024, 2048, 4096, "float32"        # medium
#I, J, K, dtype = 4096, 8192, 16384, "float32"      # large

a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir",
)
module = comp.compile(sched)

tma_counters = []

# Linux Perf counters
if platform == "linux":
    tma_counters += [
        "TopdownL1"
    ]
elif platform == "darwin":
    # On MacOS, requires sudo to get counters
    # TODO: should be tested ideally
    tma_counters = []


evaluator = module.get_evaluator(
    validate=True,
    pmu_counters=tma_counters,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
print(f"counters: {tma_counters}")
print(f"results TopDownL1: {[int(x) for x in results]}")

print("=============\n")

tma_counters = []

# Linux Perf counters
if platform == "linux":
    tma_counters += [
        "TopdownL2"
    ]
elif platform == "darwin":
    # On MacOS, requires sudo to get counters
    # TODO: should be tested ideally
    tma_counters = []


evaluator = module.get_evaluator(
    validate=True,
    pmu_counters=tma_counters,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
print(f"counters: {tma_counters}")
print(f"results TopDownL2: {[int(x) for x in results]}")

print("=============\n")

tma_counters = []

# Linux Perf counters
if platform == "linux":
    tma_counters += [
        "TopdownL3"
    ]
elif platform == "darwin":
    # On MacOS, requires sudo to get counters
    # TODO: should be tested ideally
    tma_counters = []


evaluator = module.get_evaluator(
    validate=True,
    pmu_counters=tma_counters,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
print(f"counters: {tma_counters}")
print(f"results TopDownL2: {[int(x) for x in results]}")

# CHECK:       CODE: 0
# CHECK-NEXT:  counters:
# CHECK-NEXT:  results:
