# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test strategy PPRPRP with interchange on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = """
- PP:
- RP: interchange
- RP:"""
strategy = Strategy(graph, spec, initialize=False,)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK: ['1 <= prt_interchange_0 <= 6', 'prt_i_0 || {21}', 'prt_i_1 || {21, prt_i_0}', 'prt_i_2 || {21, prt_i_0, prt_i_1}', 'prt_j_0 || {32}', 'prt_j_1 || {32, prt_j_0}', 'prt_j_2 || {32, prt_j_0, prt_j_1}', 'prt_k_0 || {12}']
# CHECK-NEXT: 100
