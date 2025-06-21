# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import TVMBackend as Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler = sch,
    node_name = "C_reduce",
    abstract_axis = ["i","j","k"],
    spec = {
        "k": {},
        "i": {},
        "j": {},
        "i#2": {"unroll": None},
        "j#16": {"vectorize": None},
    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# XFAIL: *
