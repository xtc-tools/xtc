# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test multiple splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
        "j": {},
        "k": {},
        "i": {},
        "i#iL2": {},
        "j#jDDR": {},
        "i[:6]": {
            "i#3": {},
            "i[:2:]": {
                "i#iR1": {"unroll": True},
                "j#jR1": {"vectorize": True},
            },
            "i[:iS:]": {"i#iR3": {}, "j#jR3": {}},
        },
        "i[6:]": {
            "i#iR2": {"unroll": True},
            "j#jR2": {"vectorize": True},
    },
}
strategy = Strategy(graph, spec, initialize=False)

for x in sorted(strategy._constraints):
    print(x)
print(sum(1 for _ in strategy.sample(100)))

# CHECK: iL2 || {21}
# CHECK-NEXT: iR1 || {21, iL2, 3}
# CHECK-NEXT: iR2 || {21, iL2}
# CHECK-NEXT: iR3 || {21, iL2, 3}
# CHECK-NEXT: iS + 2 == 3
# CHECK-NEXT: i_1_ + 6 == iL2
# CHECK-NEXT: i_1_ <= iL2
# CHECK-NEXT: jDDR || {32}
# CHECK-NEXT: jR1 || {32, jDDR}
# CHECK-NEXT: jR2 || {32, jDDR}
# CHECK-NEXT: jR3 || {32, jDDR}
# CHECK-NEXT: 100
