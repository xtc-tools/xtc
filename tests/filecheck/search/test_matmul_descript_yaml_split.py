# RUN: python %s -O 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = """
    j:
    k:
    i:
    i#iL3:
    i#iL2:
    j#jDDR:
    i[:iS]:
            i#iR1: unroll
            j#jR1: vectorize
            k#SR:
    i[iS:]:
            i#iR2: unroll
            j#jR2: unroll
"""
strategy = Strategy(graph, spec)

for x in sorted(strategy._constraints):
    print(x)
print(sum(1 for _ in strategy.sample(100)))

# CHECK: SR || {12}
# CHECK-NEXT: iL2 || {21, iL3}
# CHECK-NEXT: iL3 || {21}
# CHECK-NEXT: iR1 || {21, iL3, iL2}
# CHECK-NEXT: iR2 || {21, iL3, iL2}
# CHECK-NEXT: iS <= iL2
# CHECK-NEXT: i_1_ + iS == iL2
# CHECK-NEXT: i_1_ <= iL2
# CHECK-NEXT: jDDR || {32}
# CHECK-NEXT: jR1 || {32, jDDR}
# CHECK-NEXT: jR2 || {32, jDDR}
# CHECK-NEXT: 100
