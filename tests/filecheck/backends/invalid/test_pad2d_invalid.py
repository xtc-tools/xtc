# RUN: not python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad2d(a, padding={-2: 2}, name="A_pad")
    p2 = O.pad2d(b, padding=(0, 2), axes=(-2, -1), name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad(m_pad, padding={-2: (0, 2), -1: (0, 2)}, name="C")
graph = gb.graph

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad2d_invalid_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:  AssertionError: padding for pad2d of wrong size, expected 1, 2 or 4: {-2: 2} 
