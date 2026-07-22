# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test simple schedule on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
    "k": {},
    "i": {},
    "j": {},
    "i#i1": {},
    "j#j1": {},
    "j#j2": {}
}

strategy = Strategy(graph, spec, initialize=False)

for x in sorted(strategy._constraints):
    print(x)
print(sum(1 for _ in strategy.sample(100)))

# CHECK: i1 || {21}
# CHECK-NEXT: j1 || {32}
# CHECK-NEXT: j2 || {32, j1}
# CHECK-NEXT: 84
