# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test multiple splits on matmul
"""

import utils.search as utils
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

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK:       Traceback (most recent call last):
# CHECK-NEXT:    File "/home/cguillon/work/xtc-future/xtc/tests/filecheck/search/test_matmul_descript_split_in_split.py", line 8, in <module>
# CHECK-NEXT:      from xtc.search.strategies import Strategy_Descript as Strategy
# CHECK-NEXT:  ImportError: cannot import name 'Strategy_Descript' from 'xtc.search.strategies' (/home/cguillon/work/xtc-future/xtc/src/xtc/search/strategies.py)
