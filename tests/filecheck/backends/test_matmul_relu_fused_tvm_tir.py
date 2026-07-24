# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul_relu") as gb:
    m = O.matmul(a, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, tir_schedule=True)

sch = impl.get_scheduler(default_node="matmul")
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["i", "j", "i1", "j1", "k"])
sch.fuse_consumer_at("j1")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_fused_tvm_tir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       O = sch.get_block("matmul")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  O_F0 = sch.get_consumers(O)[0]
# CHECK-NEXT:  i, i1, = sch.split(i, factors=[None, 2])
# CHECK-NEXT:  j, j1, = sch.split(j, factors=[None, 16])
# CHECK-NEXT:  sch.reorder(i, j, i1, j1, k)
# CHECK-NEXT:  sch.reverse_compute_at(O_F0, j1)
# CHECK:       CODE: 0
