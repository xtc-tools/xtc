# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test strategy Goto on matmul
"""

import utils.search as utils
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

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK:       Traceback (most recent call last):
# CHECK-NEXT:    File "/home/cguillon/work/xtc-future/xtc/tests/filecheck/search/test_matmul_descript_goto.py", line 8, in <module>
# CHECK-NEXT:      from xtc.search.strategies import Strategy_Descript as Strategy
# CHECK-NEXT:  ImportError: cannot import name 'Strategy_Descript' from 'xtc.search.strategies' (/home/cguillon/work/xtc-future/xtc/src/xtc/search/strategies.py)
