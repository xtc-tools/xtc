# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PRP (one level tiling for parallel axes) on matmul
"""
import utils
from xtc.search.strategies import Strategy_PRP as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

# CHECK:       schedule O0: [1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O1: [1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 1, 'j1': 1}, 'k': {'k': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 1, 'i1': 1})]
# CHECK-NEXT:  schedule O2: [1, 16]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 1, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'i1': 1})]
# CHECK-NEXT:  schedule O3: [3, 16]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 3, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'i1': 3})]
# CHECK-NEXT:  sample 0: [1, 1]
# CHECK-NEXT:  sample 1: [1, 2]
# CHECK-NEXT:  sample 2: [1, 4]
# CHECK-NEXT:  sample 3: [1, 8]
# CHECK-NEXT:  sample 4: [1, 16]
# CHECK-NEXT:  sample 5: [1, 32]
# CHECK-NEXT:  sample 6: [3, 1]
# CHECK-NEXT:  sample 7: [3, 2]
# CHECK-NEXT:  sample 8: [3, 4]
# CHECK-NEXT:  sample 9: [3, 8]
# CHECK-NEXT:  sample 10: [3, 16]
# CHECK-NEXT:  sample 11: [3, 32]
# CHECK-NEXT:  sample 12: [7, 1]
# CHECK-NEXT:  sample 13: [7, 2]
# CHECK-NEXT:  sample 14: [7, 4]
# CHECK-NEXT:  sample 15: [7, 8]
# CHECK-NEXT:  sample 16: [7, 16]
# CHECK-NEXT:  stats {'filtered': 17, 'all': 24}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_fill', node_ident='__xtc_id_%2_fill_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {'i': 1}, 'j': {'j': 1}}, permutation={'.': ['i', 'j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2_reduce', node_ident='__xtc_id_%2_reduce_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'i': 7, 'i1': 1}, 'j': {'j': 16, 'j1': 1}, 'k': {'k': 1}}, permutation={'%2_reduce': ['i', 'j', 'k', 'i1', 'j1']}, vectorization=['j1'], parallelization=[], unrolling={'j1': 16, 'i1': 7})]
