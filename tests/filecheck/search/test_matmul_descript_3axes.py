# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy 3-axis on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph, backend="tvm")
spec = {
    "DDR": {
        "j": {},
        "k": {},
        "i": {},
        "explore_axis_order": None,
    },
    "R": {
        "j#jR": {},
        "k#kR": {},
        "i#iR": {},
        "explore_axis_order": None,
    },
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['iR || {21}', 'jR || {32}', 'kR || {12}', 'order_DDR in {0, 1, 2, 3, 4, 5}', 'order_R in {0, 1, 2, 3, 4, 5}']
