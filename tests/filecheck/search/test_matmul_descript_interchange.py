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
    "i#i1": {"interchange": "int"},
    "j#j1": {"interchange": "int"},
    "j#j2": {}
}

strategy = Strategy(graph, spec, initialize=False)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK: ['1 <= int <= 2', 'i1 || {21}', 'j1 || {32}', 'j2 || {32, j1}']
# CHECK-NEXT: 100
