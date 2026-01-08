# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PPRPRPvr (Ansor like tiling, vectorized and constraints) on conv2d
"""
import utils
from xtc.search.strategies import Strategy_PRP as Strategy

graph = utils.get_graph_conv2d()
backend = utils.get_backend(graph)
strategy = Strategy(
    graph,
    max_unroll=32,
    threads=4,
    vreg_num=4,
    l1_size=1024,
    l2_size=2*1024,
)

utils.print_random_samples(backend, strategy, 20)

# CHECK:       sample 0: [2, 1, 1, 32]
# CHECK-NEXT:  sample 1: [2, 2, 1, 16]
# CHECK-NEXT:  sample 2: [1, 1, 1, 4]
# CHECK-NEXT:  sample 3: [1, 1, 1, 32]
# CHECK-NEXT:  sample 4: [1, 2, 2, 1]
# CHECK-NEXT:  sample 5: [1, 2, 2, 8]
# CHECK-NEXT:  sample 6: [1, 1, 2, 16]
# CHECK-NEXT:  sample 7: [1, 2, 1, 32]
# CHECK-NEXT:  sample 8: [2, 2, 2, 8]
# CHECK-NEXT:  sample 9: [2, 2, 2, 4]
# CHECK-NEXT:  sample 10: [2, 1, 1, 2]
# CHECK-NEXT:  sample 11: [2, 1, 1, 4]
# CHECK-NEXT:  sample 12: [2, 1, 2, 16]
# CHECK-NEXT:  sample 13: [2, 2, 2, 32]
# CHECK-NEXT:  sample 14: [2, 1, 1, 1]
# CHECK-NEXT:  sample 15: [2, 1, 2, 32]
# CHECK-NEXT:  sample 16: [2, 1, 2, 8]
# CHECK-NEXT:  sample 17: [2, 1, 2, 4]
# CHECK-NEXT:  sample 18: [2, 2, 1, 8]
# CHECK-NEXT:  sample 19: [2, 2, 1, 4]
# CHECK-NEXT:  stats {'filtered': 20}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 2}, 'h': {'./h1': 2}, 'w': {'./w1': 1}, 'f': {'./f1': 4}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=['./b'], unrolling={'./f1': 4, './w1': 1, './h1': 2, './b1': 2})]
