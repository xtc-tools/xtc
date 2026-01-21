# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
        "j": {"parallelize": "j_parallel"},
        "k": {},
        "i": {},
        "pack": ("pack_B", 1, True),
        "pack": ("pack_A", 0, True),
        "j#jL3": {},
        "i#iL2": {},
        "k#kL1": {"unroll": "k_unroll"},
        "i#iR": {"unroll": None}, "j#jR": {"vectorize": "j_vectorise"}
}
constraint = ["iR * jR <= 56"]
strategy = Strategy(graph, spec, constraints=constraint, initialize=False)

print(strategy._constraints)

# CHECK: ['iL2 || {21}', 'iR * jR <= 56', 'iR || {21, iL2}', 'jL3 || {32}', 'jR || {32, jL3}', 'j_parallel in {0, 1}', 'j_vectorise in {0, 1}', 'kL1 || {12}', 'k_unroll || kL1', 'pack_A in {0, 1}']
