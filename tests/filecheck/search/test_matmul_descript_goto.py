# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy Goto on matmul
"""

import utils
from xtc.search.strategies import Strategy_Descript as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
spec = {
    "DDRj": {
        "j": {"parallelize": "j_parallel"},
    },
    "DDR": {
        "k": {},
        "i": {},
        "explore_axis_order": None,
        "pack": [("pack_B", 1, True), ("pack_A", 0, True)],
    },
    "L3": {
        "j#jL3": {},
    },
    "L2": {
        "i#iL2": {},
    },
    "L1": {
        "k#kL1": {"unroll": "k_unroll"},
    },
    "R": {"i#iR": {"unroll": None}, "j#jR": {"vectorize": "j_vectorise"}},
}
constraint = ["iR * jR <= 56"]
strategy = Strategy(graph, spec, constraints=constraint, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

# CHECK:       schedule O0: [1, 1, 1, 0]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 1}, 'j': {'./j1': 1}, 'k': {'./k1': 1}}, permutation={'.': ['./i', './j', './k', './i1', './j1', './k1']}, vectorization=[], parallelization=[], unrolling={'./k1': 1, './j1': 1, './i1': 1})]
# CHECK-NEXT:  schedule O1: [1, 1, 1, 0]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 1}, 'j': {'./j1': 1}, 'k': {'./k1': 1}}, permutation={'.': ['./i', './j', './k', './i1', './j1', './k1']}, vectorization=[], parallelization=[], unrolling={'./k1': 1, './j1': 1, './i1': 1})]
# CHECK-NEXT:  schedule O2: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 1}, 'j': {'./j1': 1}, 'k': {'./k1': 1}}, permutation={'.': ['./i', './j', './k', './i1', './k1', './j1']}, vectorization=['./j1'], parallelization=[], unrolling={'./j1': 1, './k1': 1, './i1': 1})]
# CHECK-NEXT:  schedule O3: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 1}, 'j': {'./j1': 1}, 'k': {'./k1': 1}}, permutation={'.': ['./i', './j', './k', './i1', './k1', './j1']}, vectorization=['./j1'], parallelization=[], unrolling={'./j1': 1, './k1': 1, './i1': 1})]
# CHECK-NEXT:  sample 0: [1, 1, 1, 0]
# CHECK-NEXT:  sample 1: [1, 1, 1, 1]
# CHECK-NEXT:  sample 2: [1, 1, 1, 2]
# CHECK-NEXT:  sample 3: [1, 1, 1, 3]
# CHECK-NEXT:  sample 4: [1, 1, 1, 4]
# CHECK-NEXT:  sample 5: [1, 1, 1, 5]
# CHECK-NEXT:  sample 6: [1, 1, 2, 0]
# CHECK-NEXT:  sample 7: [1, 1, 2, 1]
# CHECK-NEXT:  sample 8: [1, 1, 2, 2]
# CHECK-NEXT:  sample 9: [1, 1, 2, 3]
# CHECK-NEXT:  sample 10: [1, 1, 2, 4]
# CHECK-NEXT:  sample 11: [1, 1, 2, 5]
# CHECK-NEXT:  sample 12: [1, 1, 3, 0]
# CHECK-NEXT:  sample 13: [1, 1, 3, 1]
# CHECK-NEXT:  sample 14: [1, 1, 3, 2]
# CHECK-NEXT:  sample 15: [1, 1, 3, 3]
# CHECK-NEXT:  sample 16: [1, 1, 3, 4]
# CHECK-NEXT:  sample 17: [1, 1, 3, 5]
# CHECK-NEXT:  sample 18: [1, 1, 4, 0]
# CHECK-NEXT:  sample 19: [1, 1, 4, 1]
# CHECK-NEXT:  sample 20: [1, 1, 4, 2]
# CHECK-NEXT:  sample 21: [1, 1, 4, 3]
# CHECK-NEXT:  sample 22: [1, 1, 4, 4]
# CHECK-NEXT:  sample 23: [1, 1, 4, 5]
# CHECK-NEXT:  sample 24: [1, 1, 6, 0]
# CHECK-NEXT:  sample 25: [1, 1, 6, 1]
# CHECK-NEXT:  sample 26: [1, 1, 6, 2]
# CHECK-NEXT:  sample 27: [1, 1, 6, 3]
# CHECK-NEXT:  sample 28: [1, 1, 6, 4]
# CHECK-NEXT:  sample 29: [1, 1, 6, 5]
# CHECK-NEXT:  sample 30: [1, 2, 1, 0]
# CHECK-NEXT:  sample 31: [1, 2, 1, 1]
# CHECK-NEXT:  sample 32: [1, 2, 1, 2]
# CHECK-NEXT:  sample 33: [1, 2, 1, 3]
# CHECK-NEXT:  sample 34: [1, 2, 1, 4]
# CHECK-NEXT:  sample 35: [1, 2, 1, 5]
# CHECK-NEXT:  sample 36: [1, 2, 2, 0]
# CHECK-NEXT:  sample 37: [1, 2, 2, 1]
# CHECK-NEXT:  sample 38: [1, 2, 2, 2]
# CHECK-NEXT:  sample 39: [1, 2, 2, 3]
# CHECK-NEXT:  sample 40: [1, 2, 2, 4]
# CHECK-NEXT:  sample 41: [1, 2, 2, 5]
# CHECK-NEXT:  sample 42: [1, 2, 3, 0]
# CHECK-NEXT:  sample 43: [1, 2, 3, 1]
# CHECK-NEXT:  sample 44: [1, 2, 3, 2]
# CHECK-NEXT:  sample 45: [1, 2, 3, 3]
# CHECK-NEXT:  sample 46: [1, 2, 3, 4]
# CHECK-NEXT:  sample 47: [1, 2, 3, 5]
# CHECK-NEXT:  sample 48: [1, 2, 4, 0]
# CHECK-NEXT:  sample 49: [1, 2, 4, 1]
# CHECK-NEXT:  sample 50: [1, 2, 4, 2]
# CHECK-NEXT:  sample 51: [1, 2, 4, 3]
# CHECK-NEXT:  sample 52: [1, 2, 4, 4]
# CHECK-NEXT:  sample 53: [1, 2, 4, 5]
# CHECK-NEXT:  sample 54: [1, 2, 6, 1]
# CHECK-NEXT:  sample 55: [1, 2, 6, 4]
# CHECK-NEXT:  sample 56: [1, 4, 1, 0]
# CHECK-NEXT:  sample 57: [1, 4, 1, 1]
# CHECK-NEXT:  sample 58: [1, 4, 1, 2]
# CHECK-NEXT:  sample 59: [1, 4, 1, 3]
# CHECK-NEXT:  sample 60: [1, 4, 1, 4]
# CHECK-NEXT:  sample 61: [1, 4, 1, 5]
# CHECK-NEXT:  sample 62: [1, 4, 2, 0]
# CHECK-NEXT:  sample 63: [1, 4, 2, 1]
# CHECK-NEXT:  sample 64: [1, 4, 2, 2]
# CHECK-NEXT:  sample 65: [1, 4, 2, 3]
# CHECK-NEXT:  sample 66: [1, 4, 2, 4]
# CHECK-NEXT:  sample 67: [1, 4, 2, 5]
# CHECK-NEXT:  sample 68: [1, 4, 3, 1]
# CHECK-NEXT:  sample 69: [1, 4, 3, 4]
# CHECK-NEXT:  sample 70: [1, 4, 4, 1]
# CHECK-NEXT:  sample 71: [1, 4, 4, 4]
# CHECK-NEXT:  sample 72: [1, 4, 6, 1]
# CHECK-NEXT:  sample 73: [1, 4, 6, 4]
# CHECK-NEXT:  sample 74: [1, 8, 1, 0]
# CHECK-NEXT:  sample 75: [1, 8, 1, 1]
# CHECK-NEXT:  sample 76: [1, 8, 1, 2]
# CHECK-NEXT:  sample 77: [1, 8, 1, 3]
# CHECK-NEXT:  sample 78: [1, 8, 1, 4]
# CHECK-NEXT:  sample 79: [1, 8, 1, 5]
# CHECK-NEXT:  sample 80: [1, 8, 2, 1]
# CHECK-NEXT:  sample 81: [1, 8, 2, 4]
# CHECK-NEXT:  sample 82: [1, 8, 3, 1]
# CHECK-NEXT:  sample 83: [1, 8, 3, 4]
# CHECK-NEXT:  sample 84: [1, 8, 4, 1]
# CHECK-NEXT:  sample 85: [1, 8, 4, 4]
# CHECK-NEXT:  sample 86: [1, 8, 6, 1]
# CHECK-NEXT:  sample 87: [1, 8, 6, 4]
# CHECK-NEXT:  sample 88: [1, 16, 1, 1]
# CHECK-NEXT:  sample 89: [1, 16, 1, 4]
# CHECK-NEXT:  sample 90: [1, 16, 2, 1]
# CHECK-NEXT:  sample 91: [1, 16, 2, 4]
# CHECK-NEXT:  sample 92: [1, 16, 3, 1]
# CHECK-NEXT:  sample 93: [1, 16, 3, 4]
# CHECK-NEXT:  sample 94: [1, 16, 4, 1]
# CHECK-NEXT:  sample 95: [1, 16, 4, 4]
# CHECK-NEXT:  sample 96: [1, 16, 6, 1]
# CHECK-NEXT:  sample 97: [1, 16, 6, 4]
# CHECK-NEXT:  sample 98: [1, 32, 1, 1]
# CHECK-NEXT:  sample 99: [1, 32, 1, 4]
# CHECK-NEXT:  stats {'filtered': 100, 'all': 185}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 1}, 'j': {'./j1': 32}, 'k': {'./k1': 1}}, permutation={'.': ['./i', './j', './k', './k1', './i1', './j1']}, vectorization=['./j1'], parallelization=[], unrolling={'./j1': 32, './i1': 1, './k1': 1})]
