# RUN: python -O %s 2>&1 | filecheck %s
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
        "i#iL2": {},
        "j#jDDR": {},
        "i[:6]": {
            "i#3": {},
            "i[:2:]": {
                "i#iR1": {"unroll": None},
                "j#jR1": {"vectorize": None},
            },
            "i[:iS:]": {"i#iR3": {}, "j#jR3": {}},
        },
        "i[6:]": {
            "i#iR2": {"unroll": None},
            "j#jR2": {"vectorize": None},
    },
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['iL2 || {21}', 'iR1 || {21, iL2, 3}', 'iR2 || {21, iL2}', 'iR3 || {21, iL2, 3}', 'iS + 2 == 3', 'i_1_ + 6 == iL2', 'i_1_ <= iL2', 'jDDR || {32}', 'jR1 || {32, jDDR}', 'jR2 || {32, jDDR}', 'jR3 || {32, jDDR}']
