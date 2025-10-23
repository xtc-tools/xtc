# RUN: python -O %s 2>&1 | filecheck %s
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
    "L3": {
        "i#iL2": {},
    },
    "L2": {
        "j#jDDR": {},
        "i[:6]": {
            "L1": {"i#3": {}},
            "R": {
                "i[:2:]": {
                    "RR": {
                        "i#iR1": {"unroll": None},
                        "j#jR1": {"vectorize": None},
                    }
                },
                "i[:iS:]": {"RR": {"i#iR3": {}, "j#jR3": {}}},
            },
        },
        "i[6:]": {
            "R": {
                "i#iR2": {"unroll": None},
                "j#jR2": {"vectorize": None},
            },
        },
    },
}
strategy = Strategy(graph, spec, initialize=False)

print(strategy._constraints)

# CHECK: ['1 || jR2 || jDDR || 32', '1 || jR1 || jDDR || 32', '1 || iR2 || 2', '1 || iR1 || 5', '7 || iL3 || 21']
