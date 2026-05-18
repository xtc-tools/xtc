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

<<<<<<< HEAD
=======
>>>>>>> 7fae6dc (Add mapping order for the gpu thread and block)
<<<<<<< HEAD
=======
>>>>>>> 7fae6dc (Add mapping order for the gpu thread and block)
# CHECK:         File "/home/ruicesista/Documents/xtc/xtc/tests/filecheck/search/test_matmul_prp.py", line 17
# CHECK-NEXT:      >>>>>>> 7fae6dc (Add mapping order for the gpu thread and block)
# CHECK-NEXT:              ^
# CHECK-NEXT:  SyntaxError: invalid decimal literal
