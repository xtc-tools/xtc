# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_conv2d()
backend = utils.get_backend(graph)
spec = """
constraints:
    - footprint(A, L_r) > 1
schedule:
    - P:
    - R: level=L
    - P:
"""
strategy = Strategy(graph, spec, initialize=False)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK: ['prt_b_0 || {2}', 'prt_b_0*(prt_h_0-1)*2+(1-1)+1*(prt_w_0-1)*2+(7-1)+1*3 > 1', 'prt_f_0 || {32}', 'prt_h_0 || {2}', 'prt_w_0 || {2}']
# CHECK-NEXT: 48
