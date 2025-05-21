# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy OO (one level tiling for all axes) on matmul
"""
import utils
from xtc.search.strategies import Strategy_OO as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

# CHECK:       schedule O0: [1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'k', 'j', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'k1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O1: [1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'k', 'j', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'k1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O2: [1, 16, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'k', 'j', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'k1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O3: [3, 16, 12]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 3, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 12, 'k1': 1}}, permutation={'%2_reduce': ['i', 'k', 'j', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'k1': 12, 'i1': 3})]
# CHECK-NEXT:  sample 0: [1, 1, 1]
# CHECK-NEXT:  sample 1: [1, 1, 2]
# CHECK-NEXT:  sample 2: [1, 1, 3]
# CHECK-NEXT:  sample 3: [1, 1, 4]
# CHECK-NEXT:  sample 4: [1, 1, 6]
# CHECK-NEXT:  sample 5: [1, 2, 1]
# CHECK-NEXT:  sample 6: [1, 2, 2]
# CHECK-NEXT:  sample 7: [1, 2, 3]
# CHECK-NEXT:  sample 8: [1, 2, 4]
# CHECK-NEXT:  sample 9: [1, 2, 6]
# CHECK-NEXT:  sample 10: [1, 4, 1]
# CHECK-NEXT:  sample 11: [1, 4, 2]
# CHECK-NEXT:  sample 12: [1, 4, 3]
# CHECK-NEXT:  sample 13: [1, 4, 4]
# CHECK-NEXT:  sample 14: [1, 4, 6]
# CHECK-NEXT:  sample 15: [1, 8, 1]
# CHECK-NEXT:  sample 16: [1, 8, 2]
# CHECK-NEXT:  sample 17: [1, 8, 3]
# CHECK-NEXT:  sample 18: [1, 8, 4]
# CHECK-NEXT:  sample 19: [1, 8, 6]
# CHECK-NEXT:  sample 20: [1, 16, 1]
# CHECK-NEXT:  sample 21: [1, 16, 2]
# CHECK-NEXT:  sample 22: [1, 16, 3]
# CHECK-NEXT:  sample 23: [1, 16, 4]
# CHECK-NEXT:  sample 24: [1, 16, 6]
# CHECK-NEXT:  sample 25: [1, 32, 1]
# CHECK-NEXT:  sample 26: [1, 32, 2]
# CHECK-NEXT:  sample 27: [1, 32, 3]
# CHECK-NEXT:  sample 28: [1, 32, 4]
# CHECK-NEXT:  sample 29: [3, 1, 1]
# CHECK-NEXT:  sample 30: [3, 1, 2]
# CHECK-NEXT:  sample 31: [3, 2, 1]
# CHECK-NEXT:  sample 32: [3, 2, 2]
# CHECK-NEXT:  sample 33: [3, 4, 1]
# CHECK-NEXT:  sample 34: [3, 4, 2]
# CHECK-NEXT:  sample 35: [3, 8, 1]
# CHECK-NEXT:  sample 36: [3, 8, 2]
# CHECK-NEXT:  sample 37: [3, 16, 1]
# CHECK-NEXT:  sample 38: [3, 16, 2]
# CHECK-NEXT:  sample 39: [3, 32, 1]
# CHECK-NEXT:  sample 40: [7, 1, 1]
# CHECK-NEXT:  sample 41: [7, 2, 1]
# CHECK-NEXT:  sample 42: [7, 4, 1]
# CHECK-NEXT:  sample 43: [7, 8, 1]
# CHECK-NEXT:  sample 44: [7, 16, 1]
# CHECK-NEXT:  stats {'filtered': 45, 'all': 144}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 7, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'k', 'j', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'k1': 1, 'i1': 7})]
