# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
import sys

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

hw_counters = []

# Linux Perf counters
if sys.platform == "linux":
    hw_counters += [
        "TopdownL1","TopdownL2"
    ]
elif sys.platform == "darwin":
    # On MacOS, requires sudo to get counters
    # TODO: should be tested ideally
    hw_counters = []


evaluator = module.get_evaluator(
    validate=True,
    pmu_counters=hw_counters,
)
results, code, error = evaluator.evaluate()
print(f"{'CODE':<25}: {code}")
print(f"{'counters':<25}: {hw_counters}")
print(f"{'results':<25}: {[round(x, 2) for x in results]}")


use_colors = sys.stdout.isatty()

RED    = "\033[91m" if use_colors else ""
ORANGE = "\033[38;5;208m" if use_colors else ""
YELLOW = "\033[93m" if use_colors else ""
GREEN  = "\033[92m" if use_colors else ""
MAGENTA  = "\033[95m" if use_colors else ""
RESET  = "\033[0m" if use_colors else ""

def get_c(val):
    if val > 90: return RED
    if val > 75: return ORANGE
    if val > 50: return YELLOW
    if val > 10: return GREEN
    if val == -1: return MAGENTA
    return RESET

w = 25

print("-" * (w + 10))

# L1 Metrics
print(f"{'L1 Retiring':<{w}}: {get_c(results[0])}{results[0]:.2f}{RESET}")
print(f"{'L1 Bad speculation':<{w}}: {get_c(results[1])}{results[1]:.2f}{RESET}")
print(f"{'L1 Frontend bound':<{w}}: {get_c(results[2])}{results[2]:.2f}{RESET}")
print(f"{'L1 Backend bound':<{w}}: {get_c(results[3])}{results[3]:.2f}{RESET}")

print("")
# L2 Metrics
print(f"{'L2 Light ops':<{w}}: {get_c(results[4])}{results[4]:.2f}{RESET}")
print(f"{'L2 Heavy ops':<{w}}: {get_c(results[5])}{results[5]:.2f}{RESET}")
print(f"{'L2 Machine clear':<{w}}: {get_c(results[6])}{results[6]:.2f}{RESET}")
print(f"{'L2 Branch misspredict':<{w}}: {get_c(results[7])}{results[7]:.2f}{RESET}")
print(f"{'L2 Fetch latency':<{w}}: {get_c(results[8])}{results[8]:.2f}{RESET}")
print(f"{'L2 Core bound':<{w}}: {get_c(results[9])}{results[9]:.2f}{RESET}")
print(f"{'L2 Memory bound':<{w}}: {get_c(results[10])}{results[10]:.2f}{RESET}")

# CHECK:       CODE: 0
# CHECK-NEXT:  counters:
# CHECK-NEXT:  results:
