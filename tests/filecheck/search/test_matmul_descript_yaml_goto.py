# RUN: python -O %s 2>&1 | filecheck %s
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = """
DDRj:
    j:
        parallelize: j_par
DDR:
    k:
    i:
    explore: True
    A: bufferize=pack_A pad
    B: bufferize=pack_B pad
L3:
    j: size=jL3
L2:
    i: size=iL2
L1:
    k: size=kL1 unroll=kU
R:
    i: size=iR unroll
    j: size=jR vectorize=jV
"""
constraint = ["iR * jR <= 56"]
strategy = Strategy(graph, spec, constraints=constraint)

print(strategy._constraints)
print(len(list(strategy.sample(100))))

# CHECK: ['1 || kU || kL1 || 12', '1 || jR || jL3 || 32', '1 || iR || iL2 || 21', '0 <= pack_A <= 1', '0 <= pack_B <= 1', '0 <= j_par <= 1', '0 <= jV <= 1', 'iR * jR <= 56', '0 <= order_DDR <= 1']
#CHECK-NEXT: 100
