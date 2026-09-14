# RUN: python %s -O 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test splits on matmul
"""

import utils.search as utils
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

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK:       Traceback (most recent call last):
# CHECK-NEXT:    File "/home/cguillon/work/xtc-future/xtc/tests/filecheck/search/test_matmul_descript_yaml_split.py", line 8, in <module>
# CHECK-NEXT:      from xtc.search.strategies import Strategy_Descript as Strategy
# CHECK-NEXT:  ImportError: cannot import name 'Strategy_Descript' from 'xtc.search.strategies' (/home/cguillon/work/xtc-future/xtc/src/xtc/search/strategies.py)
