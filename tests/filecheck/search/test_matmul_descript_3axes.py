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

# CHECK: ['1 || kR || 12', '1 || jR || 32', '1 || iR || 21', '0 <= order_DDR <= 5', '0 <= order_R <= 5']
