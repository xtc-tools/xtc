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
ilp = nb_fma * fma_latency
vector_size = 16
elt_size = 4
reorder_buffer = 256
nb_words_L1 = 32 * 1024 // elt_size
nb_words_L2 = 1024 * 1024 // elt_size
nb_words_L3 = 36 * 1024 * 1024 // elt_size
RoB = 200

spec = f"""
constraints: 
    - 1 + nvr + nvr * mr <= {nb_registers}
    - nr == {vector_size} * nvr
    - ilp(nvr, mr, mc, nc, kc) >= {ilp}
    - nvr * mr * kr <= {reorder_buffer}
    - footprint(B, L1) <= {nb_words_L1}
    - footprint(A, L2) <= {nb_words_L2}
    - footprint(B, L3) <= {nb_words_L3}
schedule:
    j:
    k:
    B: pack=pack_B pad
    i: level=L3
    A: pack pad=pad_A
    j#nc: level=L2
    i#mc: level=L1
    k#kc: unroll=kr
    i#mr: unroll full
    j#nr: vectorize full
"""


def fn_ilp(nvr, mr, mc, nc, kc):
    if nvr * mr * kc >= RoB:
        return nvr * mr
    return min(nvr * mr * mc * nc, RoB / kc)


strategy = Strategy(
    graph,
    spec,
    functions={"ilp": fn_ilp},
    partial_tiles=True,
    partial_unrolls=True,
    initialize=False,
)

print(sorted(strategy._constraints))
print(sum(1 for _ in strategy.sample(100)))

# CHECK: ['1 + nvr + nvr * mr <= 32', 'ilp(nvr, mr, mc, nc, kc) >= 8', 'kc * nc <= 9437184', 'kc * nr <= 8192', 'kc <= 1024', 'kr <= kc', 'mc * kc <= 262144', 'mc <= 1024', 'mr || {1024, mc}', 'nc <= 1024', 'nr == 16 * nvr', 'nr || {1024, nc}', 'nvr * mr * kr <= 256', 'pack_B in {0, 1}', 'pad_A in {0,1}']
# CHECK-NEXT: 100
