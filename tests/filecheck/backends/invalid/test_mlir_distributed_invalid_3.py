# RUN: not python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_sdist

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
# Create meshes
sch.define_memory_mesh(axes={"mx": 2, "my": 2})
sch.define_processor_mesh(axes={"px": 2, "py": 42, "psx": 2, "psy": 8})
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.unroll({"i1": 2})
# Add distributed buffer
sch.distributed_buffer_at("k", 1, memory_axes=["mx", "*"])
# Bufferize in local memory
sch.pack_at("i", 1)
# Distribute i over px
sch.distribute("i", "px")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir_distributed",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: AssertionError: Memory mesh must be a subset of the processor mesh
