# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
        "j": {},
        "k": {},
        "i": {},
        "i#iL3": {},
        "i#7": {},
        "j#jDDR": {},
        "i[:5]": {
        "i#iR1": {"unroll": None},
        "j#jR1": {"parallelize": None},
        },
        "i[5:]": {
        "i#iR2": {"unroll": None},
        "j#jR2": {"parallelize": None},
        },
}
strategy = Strategy(graph, spec, initialize=False)

for x in sorted(strategy._constraints):
    print(x)
print(sum(1 for _ in strategy.sample(100)))

# CHECK: iL3 || {21}
# CHECK-NEXT: iR1 || {21, iL3, 7}
# CHECK-NEXT: iR2 || {21, iL3, 7}
# CHECK-NEXT: jDDR || {32}
# CHECK-NEXT: jR1 || {32, jDDR}
# CHECK-NEXT: jR2 || {32, jDDR}
# CHECK-NEXT: 100
