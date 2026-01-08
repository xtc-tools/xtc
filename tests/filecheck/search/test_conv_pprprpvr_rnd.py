# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PPRPRPvr (Ansor like tiling, vectorized and constraints) on conv2d
"""
import utils
from xtc.search.strategies import Strategy_PPRPRPvr as Strategy

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

# CHECK:       sample 0: [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 16, 1, 7, 1]
# CHECK-NEXT:  sample 1: [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 16, 1, 1, 1]
# CHECK-NEXT:  sample 2: [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 16, 1, 1, 3]
# CHECK-NEXT:  sample 3: [2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 16, 1, 7, 1]
# CHECK-NEXT:  sample 4: [1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 16, 1, 1, 3]
# CHECK-NEXT:  sample 5: [1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 16, 1, 1, 1]
# CHECK-NEXT:  sample 6: [1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 16, 1, 1, 1]
# CHECK-NEXT:  sample 7: [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 16, 1, 1, 1]
# CHECK-NEXT:  sample 8: [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 16, 7, 1, 1]
# CHECK-NEXT:  sample 9: [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 16, 1, 7, 1]
# CHECK-NEXT:  sample 10: [1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 16, 1, 7, 1]
# CHECK-NEXT:  sample 11: [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 16, 1, 1, 1]
# CHECK-NEXT:  sample 12: [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 16, 1, 1, 3]
# CHECK-NEXT:  sample 13: [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 16, 1, 7, 1]
# CHECK-NEXT:  sample 14: [1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 16, 1, 1, 1]
# CHECK-NEXT:  sample 15: [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 16, 7, 1, 1]
# CHECK-NEXT:  sample 16: [1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 16, 7, 1, 1]
# CHECK-NEXT:  sample 17: [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 16, 1, 7, 1]
# CHECK-NEXT:  sample 18: [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 16, 7, 1, 1]
# CHECK-NEXT:  sample 19: [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 16, 1, 1, 1]
# CHECK-NEXT:  stats {'filtered_l2': 5, 'filtered_l1': 5, 'filtered_reg': 6, 'filtered_vec': 6, 'filtered': 100}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 2, './b2': 2, './b3': 1}, 'h': {'./h1': 2, './h2': 2, './h3': 1}, 'w': {'./w1': 2, './w2': 1, './w3': 1}, 'f': {'./f1': 16, './f2': 16, './f3': 16}, 'r': {'./r1': 1}, 's': {'./s1': 1}, 'c': {'./c1': 1}}, permutation={'.': ['./b', './h', './w', './f', './b1', './h1', './w1', './f1', './r', './s', './c', './b2', './h2', './w2', './f2', './r1', './s1', './c1', './b3', './h3', './w3', './f3']}, vectorization=['./f3'], parallelization=['./b'], unrolling={'./f3': 16, './w3': 1, './h3': 1, './b3': 1, './c1': 1, './s1': 1, './r1': 1})]
