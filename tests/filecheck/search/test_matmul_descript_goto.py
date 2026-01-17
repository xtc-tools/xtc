# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
    "DDRj": {
        "j": {"parallelize": "j_parallel"},
    },
    "DDR": {
        "k": {},
        "i": {},
        "explore_axis_order": None,
        "pack": [("pack_B", 1, True), ("pack_A", 0, True)],
    },
    "L3": {
        "j#jL3": {},
    },
    "L2": {
        "i#iL2": {},
    },
    "L1": {
        "k#kL1": {"unroll": "k_unroll"},
    },
    "R": {"i#iR": {"unroll": None}, "j#jR": {"vectorize": "j_vectorise"}},
}
constraint = ["iR * jR <= 56"]
strategy = Strategy(graph, spec, constraints=constraint, initialize=False)

print(strategy._constraints)

# CHECK: ['iL2 || {21}', 'iR * jR <= 56', 'iR || {iL2, 21}', 'jL3 || {32}', 'jR || {jL3, 32}', 'j_parallel in {0, 1}', 'j_vectorise in {0, 1}', 'kL1 || {12}', 'k_unroll || kL1', 'order_DDR in {0, 1}', 'pack_A in {0, 1}', 'pack_B in {0, 1}']
