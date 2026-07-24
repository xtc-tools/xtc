# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_iree

# End-to-end IREE backend: build a matmul graph, schedule it, compile to a vmfb
# and run it through the IREE runtime, validating against the numpy reference.

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

comp = impl.get_compiler(dump_file="matmul_iree")
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: CODE: 0
