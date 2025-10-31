# RUN: python %s -O 2>&1 | filecheck %s
"""
Test splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = """
DDR:
    j:
    k:
    i:
L3:
    i#iL3:
L2:
    i#iL2:
L1:
    j#jDDR:
    i[:iS]:
        R1:
            i#iR1: unroll
            j#jR1: vectorize
        SR1:
            k#SR:
    i[iS:]:
        R2:
            i#iR2: unroll
            j#jR2: unroll
"""
strategy = Strategy(graph, spec)

print(strategy._constraints)
print(len(list(strategy.sample(100))))

# CHECK: ['1 || SR || 12', '1 || jR1 || jDDR || 32', '1 || jR2 || jDDR || 32', '1 || iL2 || iL3 || 21', '1 || iR1 || iS', '1 || iR2 || i_1_', 'iS <= iL2', 'i_1_ + iS == iL2']
# CHECK-NEXT: 100
