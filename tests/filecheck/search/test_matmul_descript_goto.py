# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
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
        "i": {"pack": ( 1, None, True)},
        "j#jL3": {"pack": ( 0, None, True)},
        "i#iL2": {},
        "k#kL1": {"unroll": "k_unroll"},
        "i#iR": {"unroll": None}, "j#jR": {"vectorize": "j_vectorise"}
}
constraint = ["iR * jR <= 56"]
strategy = Strategy(graph, spec, constraints=constraint, initialize=False)

for x in sorted(strategy._constraints):
    print(x)
print(sum(1 for _ in strategy.sample(100)))

# CHECK: iL2 || {21}
# CHECK-NEXT: iR * jR <= 56
# CHECK-NEXT: iR || {21, iL2}
# CHECK-NEXT: jL3 || {32}
# CHECK-NEXT: jR || {32, jL3}
# CHECK-NEXT: j_parallel in {0, 1}
# CHECK-NEXT: j_vectorise in {0, 1}
# CHECK-NEXT: kL1 || {12}
# CHECK-NEXT: k_unroll || kL1
# CHECK-NEXT: 100
