# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PPRPRPvr (Ansor like tiling, vectorized and constraints) on matmul
"""
import utils
from xtc.search.strategies import Strategy_PRP as Strategy

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

# CHECK:       sample 0: [21, 2]
# CHECK-NEXT:  sample 1: [7, 16]
# CHECK-NEXT:  sample 2: [3, 1]
# CHECK-NEXT:  sample 3: [3, 4]
# CHECK-NEXT:  sample 4: [1, 32]
# CHECK-NEXT:  sample 5: [1, 8]
# CHECK-NEXT:  sample 6: [1, 1]
# CHECK-NEXT:  sample 7: [1, 16]
# CHECK-NEXT:  sample 8: [21, 16]
# CHECK-NEXT:  sample 9: [7, 32]
# CHECK-NEXT:  sample 10: [7, 1]
# CHECK-NEXT:  sample 11: [21, 1]
# CHECK-NEXT:  sample 12: [7, 8]
# CHECK-NEXT:  sample 13: [7, 2]
# CHECK-NEXT:  sample 14: [7, 4]
# CHECK-NEXT:  sample 15: [21, 4]
# CHECK-NEXT:  sample 16: [3, 32]
# CHECK-NEXT:  sample 17: [1, 4]
# CHECK-NEXT:  sample 18: [3, 2]
# CHECK-NEXT:  sample 19: [21, 8]
# CHECK-NEXT:  stats {'filtered': 19}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['i', 'j'], loop_stamps=[], splits={}, tiles={'i': {}, 'j': {}}, permutation={'.': ['./i', './j']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['i', 'j', 'k'], loop_stamps=[], splits={}, tiles={'i': {'./i1': 21}, 'j': {'./j1': 8}, 'k': {}}, permutation={'.': ['./i', './j', './k', './i1', './j1']}, vectorization=['./j1'], parallelization=['./i'], unrolling={'./j1': 8, './i1': 21})]
