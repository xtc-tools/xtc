# RUN: python %s -O 2>&1 | filecheck %s
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

print(strategy._constraints)
print(len(list(strategy.sample(100))))

# CHECK: ['SR || {12}', 'iL2 || {21, iL3}', 'iL3 || {21}', 'iR1 || {21, iL3, iL2}', 'iR2 || {21, iL3, iL2}', 'iS <= iL2', 'i_1_ + iS == iL2', 'i_1_ <= iL2', 'jDDR || {32}', 'jR1 || {32, jDDR}', 'jR2 || {32, jDDR}']
# CHECK-NEXT: 100
