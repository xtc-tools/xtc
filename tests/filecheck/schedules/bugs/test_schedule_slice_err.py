# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 50, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_axis=["i", "j", "k"],
    spec={
        'k': {},
        'j': {},
        'i[0:18]':
            {'i#18': {},
             'k#32': {},
             'k#16': {},
             'i#9': {'unroll': 9},
             'j#32': {'unroll': 32},
             'j#16': {'vectorize': None}},
        'i[18:]': {
            'i#32': {},
            'k#32': {},
            'k#16': {},
            'i#16': {'unroll': 16},
            'j#32': {'unroll': 32},
            'j#16': {'vectorize': None}
        }
    }
)

# XFAIL:*
