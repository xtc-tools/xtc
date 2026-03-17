# RUN: not python %s --unroll 2>&1 | filecheck %s --check-prefix=CHECK-UNROLL
# RUN: not python %s --parallelize 2>&1 | filecheck %s --check-prefix=CHECK-PARALLELIZE
# RUN: not python %s --vectorize 2>&1 | filecheck %s --check-prefix=CHECK-VECTORIZE

import sys
import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph

impl = Backend(graph)

sch = impl.get_scheduler()

if "--unroll" in sys.argv:
    descript_scheduler(
        scheduler = sch,
        node_name = "C",
        abstract_dims = ["I","J","K"],
        spec = {
            "I": {},
            "K": {"unroll" : ["hello"]},
            "J": {},
        }
    )
elif "--parallelize" in sys.argv:
    descript_scheduler(
        scheduler = sch,
        node_name = "C",
        abstract_dims = ["I","J","K"],
        spec = {
            "I": {"parallelize" : ["hello"]},
            "K": {},
            "J": {},
        }
    )
elif "--vectorize" in sys.argv:
    descript_scheduler(
        scheduler = sch,
        node_name = "C",
        abstract_dims = ["I","J","K"],
        spec = {
            "I": {},
            "K": {},
            "J": {"vectorize" : ["hello"]},
        }
    )

# CHECK-UNROLL: `{"unroll" = ['hello']}`: unroll parameter should be True, False, or a string or integer.
# CHECK-PARALLELIZE: `{"parallelize" = ['hello']}`: parallelization parameter should be True, False or a string.
# CHECK-VECTORIZE: `{"vectorize" = ['hello']}`: vectorization parameter should be True, False or a string.
