# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_xvs
"""
Test strategy Goto on matmul
"""

import utils
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
    B: pack=pack_B pad
    i:
    A: pack pad=pad_A
    j#nc:
    i#mc:
    k#kc: unroll=kr
    i#mr: unroll full
    j#nr: vectorize full
"""

strategy = Strategy(graph, spec, partial_tiles=True, partial_unrolls=True, initialize=False)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK: ['1 + nvr + nvr * mr <= 32', 'kc * mc <= 262144', 'kc * nc <= 9437184', 'kc * nr <= 8192', 'kc <= 1024', 'kr <= kc', 'mc <= 1024', 'mr || {1024, mc}', 'nc <= 1024', 'nr == 16 * nvr', 'nr || {1024, nc}', 'nvr * mr * kr <= 256', 'nvr * mr >= 8', 'pack_B in {0, 1}', 'pad_A in {0,1}']
#CHECK-NEXT: 100
