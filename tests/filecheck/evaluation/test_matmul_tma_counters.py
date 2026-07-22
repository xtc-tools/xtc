# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

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

# Schedule specification
schedule_spec = {
    "i": {},
    "k": {},
    "j": {},
    f"i#{16}": {"unroll": 8},
    f"j#{16}": {"vectorize": True}
}

# Compile
scheduler = impl.get_scheduler()
descript_scheduler(
    scheduler=scheduler,
    node_name="C",
    abstract_dims=["i", "j", "k"],
    spec=schedule_spec
)
sched = scheduler.schedule()

compiler = impl.get_compiler(
    dump_file="matmul_mlir",
    shared_lib=True,
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=False
)
module = compiler.compile(sched)

hw_counters = []

# Linux Perf counters
if sys.platform == "linux":
    hw_counters += [
        "TopdownL1","TopdownL2", "TopdownL3"
    ]
elif sys.platform == "darwin":
    # On MacOS, requires sudo to get counters
    # TODO: should be tested ideally
    hw_counters = []


evaluator = module.get_evaluator(
    validate=True,
    hw_counters=hw_counters,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
print(f"{'counters'}: {hw_counters}")
print(f"{'results'}: {[round(x, 2) for x in results]}")

# CHECK:       CODE: 0
# CHECK-NEXT:  counters:
# CHECK-NEXT:  results:
