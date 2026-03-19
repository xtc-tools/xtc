# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy 3-axis on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph, backend="tvm")
spec = {
        "j": {},
        "k": {},
        "i": {},
        "j#jR": {},
        "k#kR": {},
        "i#iR": {},
}
strategy = Strategy(graph, spec, initialize=False)

print(sorted(list(strategy._constraints)))
print(sum(1 for _ in strategy.sample(100)))

# CHECK:['iR || {21}', 'jR || {32}', 'kR || {12}']
# CHECK-NEXT:100
