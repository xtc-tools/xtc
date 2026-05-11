# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy P1 (one level unordered tiling for all axes) on matmul
"""
import utils
from xtc.search.strategies import Strategy_P1 as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)

<<<<<<< HEAD
=======
>>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK:         File "/home/ruicesista/Documents/xtc/xtc/tests/filecheck/search/test_matmul_p1.py", line 237
# CHECK-NEXT:      >>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK-NEXT:                  ^
# CHECK-NEXT:  SyntaxError: invalid decimal literal
