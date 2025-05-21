# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy P1 (one level unordered tiling for all axes and vectorize) on matmul
"""
import utils
from xtc.search.strategies import Strategy_P1v as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

# CHECK:       schedule O0: [1, 1, 1, 0]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1', 'k1']}, vectorization=[], parallelization=[], unrolling={'k1': 1, 'j1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O1: [1, 1, 1, 0]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1', 'k1']}, vectorization=[], parallelization=[], unrolling={'k1': 1, 'j1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O2: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'k1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O3: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'k1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'k1': 1, 'i1': 1})]
# CHECK-NEXT:  sample 0: [1, 16, 1, 1]
# CHECK-NEXT:  sample 1: [1, 16, 1, 4]
# CHECK-NEXT:  sample 2: [1, 16, 2, 1]
# CHECK-NEXT:  sample 3: [1, 16, 2, 4]
# CHECK-NEXT:  sample 4: [1, 16, 3, 1]
# CHECK-NEXT:  sample 5: [1, 16, 3, 4]
# CHECK-NEXT:  sample 6: [1, 16, 4, 1]
# CHECK-NEXT:  sample 7: [1, 16, 4, 4]
# CHECK-NEXT:  sample 8: [1, 16, 6, 1]
# CHECK-NEXT:  sample 9: [1, 16, 6, 4]
# CHECK-NEXT:  sample 10: [1, 32, 1, 1]
# CHECK-NEXT:  sample 11: [1, 32, 1, 4]
# CHECK-NEXT:  sample 12: [1, 32, 2, 1]
# CHECK-NEXT:  sample 13: [1, 32, 2, 4]
# CHECK-NEXT:  sample 14: [1, 32, 3, 1]
# CHECK-NEXT:  sample 15: [1, 32, 3, 4]
# CHECK-NEXT:  sample 16: [1, 32, 4, 1]
# CHECK-NEXT:  sample 17: [1, 32, 4, 4]
# CHECK-NEXT:  sample 18: [3, 16, 1, 1]
# CHECK-NEXT:  sample 19: [3, 16, 1, 4]
# CHECK-NEXT:  sample 20: [3, 16, 2, 1]
# CHECK-NEXT:  sample 21: [3, 16, 2, 4]
# CHECK-NEXT:  sample 22: [3, 32, 1, 1]
# CHECK-NEXT:  sample 23: [3, 32, 1, 4]
# CHECK-NEXT:  sample 24: [7, 16, 1, 1]
# CHECK-NEXT:  sample 25: [7, 16, 1, 4]
# CHECK-NEXT:  stats {'filtered': 154, 'all': 864}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 7, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1, 'k1': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'k1', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'i1': 7, 'k1': 1})]
