# RUN: python %s 2>&1 | filecheck %s
"""
Test splits on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
    "DDR": {
        "j": {},
        "k": {},
        "i": {},
    },
    "L3": {"i#iL3": {}},
    "L2": {
        "i#7": {},
    },
    "L1": {
        "j#jDDR": {},
        "i[:5]": {
            "R1": {
                "i#iR1": {"unroll": None},
                "j#jR1": {"vectorize": None},
            },
        },
        "i[5:]": {
            "R2": {
                "i#iR2": {"unroll": None},
                "j#jR2": {"vectorize": None},
            },
        },
    },
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['1 || jR1 || jDDR || 32', '1 || jR3 || jDDR || 32', '1 || jR2 || jDDR || 32', '1 || iL2 || 21', '1 || iR1 || 2', '1 || iR3 || 1', '1 || iR2 || i_1_', 'i_1_ + 6 == iL2']
