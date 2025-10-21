# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
    "L3": {
        "k": {},
        "i": {},
        "j": {},
    },
    "L2": {
        "i#i1": {},
        "j#j1": {},
    },
    "L1": {"j#j2": {}},
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['1 || j2 || j1 || 32', '1 || i1 || 21']
