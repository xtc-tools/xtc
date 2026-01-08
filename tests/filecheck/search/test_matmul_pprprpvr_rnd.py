# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PPRPRPvr (Ansor like tiling, vectorized and constraints) on matmul
"""
import utils
from xtc.search.strategies import Strategy_PPRPRPvr as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(
    graph,
    max_unroll=32,
    threads=4,
    vreg_num=4,
    l1_size=1024,
    l2_size=8*1024,
)

utils.print_random_samples(backend, strategy, 20)

# CHECK:       sample 0: [1, 21, 1, 1, 1, 32, 3]
# CHECK-NEXT:  sample 1: [7, 3, 1, 1, 1, 16, 2]
# CHECK-NEXT:  sample 2: [7, 3, 1, 1, 2, 16, 1]
# CHECK-NEXT:  sample 3: [7, 1, 1, 1, 1, 16, 2]
# CHECK-NEXT:  sample 4: [7, 3, 1, 1, 1, 16, 1]
# CHECK-NEXT:  sample 5: [21, 1, 1, 1, 1, 16, 1]
# CHECK-NEXT:  sample 6: [3, 7, 1, 1, 1, 16, 2]
# CHECK-NEXT:  sample 7: [21, 1, 1, 2, 1, 16, 12]
# CHECK-NEXT:  sample 8: [7, 1, 1, 1, 1, 16, 6]
# CHECK-NEXT:  sample 9: [1, 21, 1, 1, 2, 16, 3]
# CHECK-NEXT:  sample 10: [3, 1, 1, 1, 1, 16, 4]
# CHECK-NEXT:  sample 11: [3, 1, 1, 1, 2, 16, 4]
# CHECK-NEXT:  sample 12: [7, 1, 1, 2, 1, 16, 2]
# CHECK-NEXT:  sample 13: [1, 7, 3, 2, 1, 16, 4]
# CHECK-NEXT:  sample 14: [3, 1, 1, 1, 2, 16, 1]
# CHECK-NEXT:  sample 15: [1, 21, 1, 1, 1, 16, 1]
# CHECK-NEXT:  sample 16: [3, 1, 1, 1, 1, 16, 12]
# CHECK-NEXT:  sample 17: [1, 3, 1, 2, 1, 16, 2]
# CHECK-NEXT:  sample 18: [1, 1, 1, 1, 2, 16, 3]
# CHECK-NEXT:  sample 19: [7, 1, 3, 1, 1, 16, 2]
# CHECK-NEXT:  stats {'filtered_l2': 2, 'filtered_l1': 2, 'filtered_reg': 3, 'filtered_vec': 3, 'filtered': 70}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 21, './i2': 3, './i3': 3}, 'j': {'./j1': 16, './j2': 16, './j3': 16}, 'k': {'./k1': 2}}, permutation={'.': ['./i', './j', './i1', './j1', './k', './i2', './j2', './k1', './i3', './j3']}, vectorization=['./j3'], parallelization=['./i'], unrolling={'./j3': 16, './i3': 3, './k1': 2})]
