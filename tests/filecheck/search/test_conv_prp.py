# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PRP (one level tiling of parallel axes) on conv2d
"""
import utils
from xtc.search.strategies import Strategy_PRP as Strategy

graph = utils.get_graph_conv2d()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

# CHECK:       schedule O0: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 1}, 'h': {'./h1': 1}, 'w': {'./w1': 1}, 'f': {'./f1': 1}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=[], unrolling={'./f1': 1, './w1': 1, './h1': 1, './b1': 1}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={})]
# CHECK-NEXT:  schedule O1: [1, 1, 1, 1]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 1}, 'h': {'./h1': 1}, 'w': {'./w1': 1}, 'f': {'./f1': 1}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=[], unrolling={'./f1': 1, './w1': 1, './h1': 1, './b1': 1}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={})]
# CHECK-NEXT:  schedule O2: [1, 1, 2, 16]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 1}, 'h': {'./h1': 1}, 'w': {'./w1': 2}, 'f': {'./f1': 16}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=[], unrolling={'./f1': 16, './w1': 2, './h1': 1, './b1': 1}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={})]
# CHECK-NEXT:  schedule O3: [1, 1, 2, 16]
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 1}, 'h': {'./h1': 1}, 'w': {'./w1': 2}, 'f': {'./f1': 16}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=[], unrolling={'./f1': 16, './w1': 2, './h1': 1, './b1': 1}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={})]
# CHECK-NEXT:  sample 0: [1, 1, 1, 1]
# CHECK-NEXT:  sample 1: [1, 1, 1, 2]
# CHECK-NEXT:  sample 2: [1, 1, 1, 4]
# CHECK-NEXT:  sample 3: [1, 1, 1, 8]
# CHECK-NEXT:  sample 4: [1, 1, 1, 16]
# CHECK-NEXT:  sample 5: [1, 1, 1, 32]
# CHECK-NEXT:  sample 6: [1, 1, 2, 1]
# CHECK-NEXT:  sample 7: [1, 1, 2, 2]
# CHECK-NEXT:  sample 8: [1, 1, 2, 4]
# CHECK-NEXT:  sample 9: [1, 1, 2, 8]
# CHECK-NEXT:  sample 10: [1, 1, 2, 16]
# CHECK-NEXT:  sample 11: [1, 1, 2, 32]
# CHECK-NEXT:  sample 12: [1, 2, 1, 1]
# CHECK-NEXT:  sample 13: [1, 2, 1, 2]
# CHECK-NEXT:  sample 14: [1, 2, 1, 4]
# CHECK-NEXT:  sample 15: [1, 2, 1, 8]
# CHECK-NEXT:  sample 16: [1, 2, 1, 16]
# CHECK-NEXT:  sample 17: [1, 2, 1, 32]
# CHECK-NEXT:  sample 18: [1, 2, 2, 1]
# CHECK-NEXT:  sample 19: [1, 2, 2, 2]
# CHECK-NEXT:  sample 20: [1, 2, 2, 4]
# CHECK-NEXT:  sample 21: [1, 2, 2, 8]
# CHECK-NEXT:  sample 22: [1, 2, 2, 16]
# CHECK-NEXT:  sample 23: [1, 2, 2, 32]
# CHECK-NEXT:  sample 24: [2, 1, 1, 1]
# CHECK-NEXT:  sample 25: [2, 1, 1, 2]
# CHECK-NEXT:  sample 26: [2, 1, 1, 4]
# CHECK-NEXT:  sample 27: [2, 1, 1, 8]
# CHECK-NEXT:  sample 28: [2, 1, 1, 16]
# CHECK-NEXT:  sample 29: [2, 1, 1, 32]
# CHECK-NEXT:  sample 30: [2, 1, 2, 1]
# CHECK-NEXT:  sample 31: [2, 1, 2, 2]
# CHECK-NEXT:  sample 32: [2, 1, 2, 4]
# CHECK-NEXT:  sample 33: [2, 1, 2, 8]
# CHECK-NEXT:  sample 34: [2, 1, 2, 16]
# CHECK-NEXT:  sample 35: [2, 1, 2, 32]
# CHECK-NEXT:  sample 36: [2, 2, 1, 1]
# CHECK-NEXT:  sample 37: [2, 2, 1, 2]
# CHECK-NEXT:  sample 38: [2, 2, 1, 4]
# CHECK-NEXT:  sample 39: [2, 2, 1, 8]
# CHECK-NEXT:  sample 40: [2, 2, 1, 16]
# CHECK-NEXT:  sample 41: [2, 2, 1, 32]
# CHECK-NEXT:  sample 42: [2, 2, 2, 1]
# CHECK-NEXT:  sample 43: [2, 2, 2, 2]
# CHECK-NEXT:  sample 44: [2, 2, 2, 4]
# CHECK-NEXT:  sample 45: [2, 2, 2, 8]
# CHECK-NEXT:  sample 46: [2, 2, 2, 16]
# CHECK-NEXT:  stats {'filtered': 47, 'all': 48}
# CHECK-NEXT:  [MlirNodeSchedule(node_name='%2_0', node_ident='__xtc_id_%2_0_', dims=['b', 'h', 'w', 'f'], loop_stamps=[], splits={}, tiles={'b': {}, 'h': {}, 'w': {}, 'f': {}}, permutation={'.': ['./b', './h', './w', './f']}, vectorization=[], parallelization=[], unrolling={}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={}), MlirNodeSchedule(node_name='%2', node_ident='__xtc_id_%2_', dims=['b', 'h', 'w', 'f', 'r', 's', 'c'], loop_stamps=[], splits={}, tiles={'b': {'./b1': 2}, 'h': {'./h1': 2}, 'w': {'./w1': 2}, 'f': {'./f1': 16}, 'r': {}, 's': {}, 'c': {}}, permutation={'.': ['./b', './h', './w', './f', './r', './s', './c', './b1', './h1', './w1', './f1']}, vectorization=['./f1'], parallelization=[], unrolling={'./f1': 16, './w1': 2, './h1': 2, './b1': 2}, packed_buffers={}, memory_mesh={}, processor_mesh={}, distribution={}, distributed_buffers={})]
