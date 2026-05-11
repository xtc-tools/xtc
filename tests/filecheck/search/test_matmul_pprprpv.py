# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy PPRPRPv (Ansor like tiling and vectorized) on matmul
"""
import utils
from xtc.search.strategies import Strategy_PPRPRPv as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph, max_unroll=8, threads=4)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy,100)

<<<<<<< HEAD
=======
>>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
<<<<<<< HEAD
=======
>>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK:         File "/home/ruicesista/Documents/xtc/xtc/tests/filecheck/search/test_matmul_pprprpv.py", line 32
# CHECK-NEXT:      >>>>>>> 59236d6 (mlir: Add 2 primitive for gpu on the scheduler)
# CHECK-NEXT:                  ^
# CHECK-NEXT:  SyntaxError: invalid decimal literal
