# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test strategy Goto on matmul
"""

import utils.search as utils
from xtc.search.strategies import Strategy_Descript as Strategy

import xtc.graphs.xtc.op as O

graph = utils.get_graph_matmul()
I, J, K, dtype = 1024, 1024, 1024, "float32"

a = O.tensor((I, K), dtype)
b = O.tensor((K, J), dtype)

with O.graph(name="matmul") as gb:
    O.matmul(a, b)
graph = gb.graph
backend = utils.get_backend(graph, "tvm")

nb_registers = 32
nb_fma = 2
fma_latency = 4
ilp = nb_fma*fma_latency
vector_size = 16
elt_size = 4
reorder_buffer = 256
nb_words_L1 = 32*1024//elt_size
nb_words_L2 = 1024*1024//elt_size
nb_words_L3 = 36*1024*1024//elt_size

spec = f"""
    constraints: 
        - 1 + nvr + nvr * mr <= {nb_registers}
        - nr == {vector_size} * nvr
        - nvr * mr >= {ilp}
        - nvr * mr * kr <= {reorder_buffer}
        - kc * nr <= {nb_words_L1}
        - kc * mc <= {nb_words_L2}
        - kc * nc <= {nb_words_L3}
    j:
    k:
    B: pack
    i:
    A: pack
    j#nc:
    i#mc:
    k#kc: unroll=kr
    i#mr: unroll full
    j#nr: vectorize full
"""
print(spec)

strategy = Strategy(graph, spec, partial_tiles=True, partial_unrolls=True, initialize=False)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

#CHECK-NEXT: 100
# CHECK:       Traceback (most recent call last):
# CHECK-NEXT:    File "/home/cguillon/work/xtc-future/xtc/tests/filecheck/search/test_matmul_descript_yaml_goto.py", line 8, in <module>
# CHECK-NEXT:      from xtc.search.strategies import Strategy_Descript as Strategy
# CHECK-NEXT:  ImportError: cannot import name 'Strategy_Descript' from 'xtc.search.strategies' (/home/cguillon/work/xtc-future/xtc/src/xtc/search/strategies.py)
