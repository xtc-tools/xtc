# RUN: python %s 2>&1 | filecheck %s
"""
Test splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
        "j": {},
        "k": {},
        "i": {},
        "i#iL3": {},
        "i#7": {},
        "j#jDDR": {},
        "i[:5]": {
        "i#iR1": {"unroll": None},
        "j#jR1": {"parallelize": None},
        },
        "i[5:]": {
        "i#iR2": {"unroll": None},
        "j#jR2": {"parallelize": None},
        },
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['iL3 || {21}', 'iR1 || {21, iL3, 7}', 'iR2 || {21, iL3, 7}', 'jDDR || {32}', 'jR1 || {32, jDDR}', 'jR2 || {32, jDDR}']
