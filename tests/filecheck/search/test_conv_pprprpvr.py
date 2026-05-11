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

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

<<<<<<< HEAD
=======
>>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
<<<<<<< HEAD
=======
>>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK:         File "/home/ruicesista/Documents/xtc/xtc/tests/filecheck/search/test_conv_pprprpvr.py", line 39
# CHECK-NEXT:      >>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK-NEXT:                  ^
# CHECK-NEXT:  SyntaxError: invalid decimal literal
