# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
"""
Test strategy PPWRPRP (Ansor like tiling for all axes and buffer) on conv2d
"""
import utils
from xtc.search.strategies import Strategy_PPWRPRP as Strategy

graph = utils.get_graph_conv2d()
backend = utils.get_backend(graph, "tvm")
strategy = Strategy(graph, max_unroll=8, threads=4)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 200)

# CHECK:       schedule O0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# CHECK-NEXT:  O = obj['%2']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "local")
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  b, b1 = sch[O].split(b, factor=1)
# CHECK-NEXT:  h, h1 = sch[O].split(h, factor=1)
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=1)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=1)
# CHECK-NEXT:  b1, b_ = sch[O].split(b1, factor=1)
# CHECK-NEXT:  h1, h_ = sch[O].split(h1, factor=1)
# CHECK-NEXT:  w1, w_ = sch[O].split(w1, factor=1)
# CHECK-NEXT:  f1, f_ = sch[O].split(f1, factor=1)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, b1, h1, w1, f1, b_, h_, w_, f_)
# CHECK-NEXT:  sch[O].parallel(b)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], f1)
# CHECK-NEXT:  b, h, w, f, = O_W0.op.axis
# CHECK-NEXT:  r, s, c, = O_W0.op.reduce_axis
# CHECK-NEXT:  b2 = b
# CHECK-NEXT:  h2 = h
# CHECK-NEXT:  w2 = w
# CHECK-NEXT:  f2 = f
# CHECK-NEXT:  r, r1 = sch[O_W0].split(r, factor=1)
# CHECK-NEXT:  s, s1 = sch[O_W0].split(s, factor=1)
# CHECK-NEXT:  c, c1 = sch[O_W0].split(c, factor=1)
# CHECK-NEXT:  b2, b3 = sch[O_W0].split(b2, factor=1)
# CHECK-NEXT:  h2, h3 = sch[O_W0].split(h2, factor=1)
# CHECK-NEXT:  w2, w3 = sch[O_W0].split(w2, factor=1)
# CHECK-NEXT:  f2, f3 = sch[O_W0].split(f2, factor=1)
# CHECK-NEXT:  sch[O_W0].reorder(r, s, c, b2, h2, w2, f2, r1, s1, c1, b3, h3, w3, f3)
# CHECK-NEXT:  sch[O_W0].unroll(f3)
# CHECK-NEXT:  sch[O_W0].unroll(w3)
# CHECK-NEXT:  sch[O_W0].unroll(h3)
# CHECK-NEXT:  sch[O_W0].unroll(b3)
# CHECK-NEXT:  sch[O_W0].unroll(c1)
# CHECK-NEXT:  sch[O_W0].unroll(s1)
# CHECK-NEXT:  sch[O_W0].unroll(r1)
# CHECK-NEXT:  sch[O_W0].vectorize(f3)
# CHECK-NEXT:  
# CHECK-NEXT:  schedule O1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# CHECK-NEXT:  O = obj['%2']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "local")
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  b, b1 = sch[O].split(b, factor=1)
# CHECK-NEXT:  h, h1 = sch[O].split(h, factor=1)
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=1)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=1)
# CHECK-NEXT:  b1, b_ = sch[O].split(b1, factor=1)
# CHECK-NEXT:  h1, h_ = sch[O].split(h1, factor=1)
# CHECK-NEXT:  w1, w_ = sch[O].split(w1, factor=1)
# CHECK-NEXT:  f1, f_ = sch[O].split(f1, factor=1)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, b1, h1, w1, f1, b_, h_, w_, f_)
# CHECK-NEXT:  sch[O].parallel(b)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], f1)
# CHECK-NEXT:  b, h, w, f, = O_W0.op.axis
# CHECK-NEXT:  r, s, c, = O_W0.op.reduce_axis
# CHECK-NEXT:  b2 = b
# CHECK-NEXT:  h2 = h
# CHECK-NEXT:  w2 = w
# CHECK-NEXT:  f2 = f
# CHECK-NEXT:  r, r1 = sch[O_W0].split(r, factor=1)
# CHECK-NEXT:  s, s1 = sch[O_W0].split(s, factor=1)
# CHECK-NEXT:  c, c1 = sch[O_W0].split(c, factor=1)
# CHECK-NEXT:  b2, b3 = sch[O_W0].split(b2, factor=1)
# CHECK-NEXT:  h2, h3 = sch[O_W0].split(h2, factor=1)
# CHECK-NEXT:  w2, w3 = sch[O_W0].split(w2, factor=1)
# CHECK-NEXT:  f2, f3 = sch[O_W0].split(f2, factor=1)
# CHECK-NEXT:  sch[O_W0].reorder(r, s, c, b2, h2, w2, f2, r1, s1, c1, b3, h3, w3, f3)
# CHECK-NEXT:  sch[O_W0].unroll(f3)
# CHECK-NEXT:  sch[O_W0].unroll(w3)
# CHECK-NEXT:  sch[O_W0].unroll(h3)
# CHECK-NEXT:  sch[O_W0].unroll(b3)
# CHECK-NEXT:  sch[O_W0].unroll(c1)
# CHECK-NEXT:  sch[O_W0].unroll(s1)
# CHECK-NEXT:  sch[O_W0].unroll(r1)
# CHECK-NEXT:  sch[O_W0].vectorize(f3)
# CHECK-NEXT:  
# CHECK-NEXT:  schedule O2: [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 16, 1, 1, 1, 1]
# CHECK-NEXT:  O = obj['%2']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "local")
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  b, b1 = sch[O].split(b, factor=1)
# CHECK-NEXT:  h, h1 = sch[O].split(h, factor=1)
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=2)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=16)
# CHECK-NEXT:  b1, b_ = sch[O].split(b1, factor=1)
# CHECK-NEXT:  h1, h_ = sch[O].split(h1, factor=1)
# CHECK-NEXT:  w1, w_ = sch[O].split(w1, factor=2)
# CHECK-NEXT:  f1, f_ = sch[O].split(f1, factor=16)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, b1, h1, w1, f1, b_, h_, w_, f_)
# CHECK-NEXT:  sch[O].parallel(b)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], f1)
# CHECK-NEXT:  b, h, w, f, = O_W0.op.axis
# CHECK-NEXT:  r, s, c, = O_W0.op.reduce_axis
# CHECK-NEXT:  b2 = b
# CHECK-NEXT:  h2 = h
# CHECK-NEXT:  w2 = w
# CHECK-NEXT:  f2 = f
# CHECK-NEXT:  r, r1 = sch[O_W0].split(r, factor=1)
# CHECK-NEXT:  s, s1 = sch[O_W0].split(s, factor=1)
# CHECK-NEXT:  c, c1 = sch[O_W0].split(c, factor=1)
# CHECK-NEXT:  b2, b3 = sch[O_W0].split(b2, factor=1)
# CHECK-NEXT:  h2, h3 = sch[O_W0].split(h2, factor=1)
# CHECK-NEXT:  w2, w3 = sch[O_W0].split(w2, factor=2)
# CHECK-NEXT:  f2, f3 = sch[O_W0].split(f2, factor=16)
# CHECK-NEXT:  sch[O_W0].reorder(r, s, c, b2, h2, w2, f2, r1, s1, c1, b3, h3, w3, f3)
# CHECK-NEXT:  sch[O_W0].unroll(f3)
# CHECK-NEXT:  sch[O_W0].unroll(w3)
# CHECK-NEXT:  sch[O_W0].unroll(h3)
# CHECK-NEXT:  sch[O_W0].unroll(b3)
# CHECK-NEXT:  sch[O_W0].unroll(c1)
# CHECK-NEXT:  sch[O_W0].unroll(s1)
# CHECK-NEXT:  sch[O_W0].unroll(r1)
# CHECK-NEXT:  sch[O_W0].vectorize(f3)
# CHECK-NEXT:  
# CHECK-NEXT:  schedule O3: [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 16, 1, 1, 3, 1]
# CHECK-NEXT:  O = obj['%2']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "local")
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  b, b1 = sch[O].split(b, factor=1)
# CHECK-NEXT:  h, h1 = sch[O].split(h, factor=1)
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=2)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=16)
# CHECK-NEXT:  b1, b_ = sch[O].split(b1, factor=1)
# CHECK-NEXT:  h1, h_ = sch[O].split(h1, factor=1)
# CHECK-NEXT:  w1, w_ = sch[O].split(w1, factor=2)
# CHECK-NEXT:  f1, f_ = sch[O].split(f1, factor=16)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, b1, h1, w1, f1, b_, h_, w_, f_)
# CHECK-NEXT:  sch[O].parallel(b)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], f1)
# CHECK-NEXT:  b, h, w, f, = O_W0.op.axis
# CHECK-NEXT:  r, s, c, = O_W0.op.reduce_axis
# CHECK-NEXT:  b2 = b
# CHECK-NEXT:  h2 = h
# CHECK-NEXT:  w2 = w
# CHECK-NEXT:  f2 = f
# CHECK-NEXT:  r, r1 = sch[O_W0].split(r, factor=1)
# CHECK-NEXT:  s, s1 = sch[O_W0].split(s, factor=1)
# CHECK-NEXT:  c, c1 = sch[O_W0].split(c, factor=3)
# CHECK-NEXT:  b2, b3 = sch[O_W0].split(b2, factor=1)
# CHECK-NEXT:  h2, h3 = sch[O_W0].split(h2, factor=1)
# CHECK-NEXT:  w2, w3 = sch[O_W0].split(w2, factor=2)
# CHECK-NEXT:  f2, f3 = sch[O_W0].split(f2, factor=16)
# CHECK-NEXT:  sch[O_W0].reorder(r, s, c, b2, h2, w2, f2, r1, s1, c1, b3, h3, w3, f3)
# CHECK-NEXT:  sch[O_W0].unroll(f3)
# CHECK-NEXT:  sch[O_W0].unroll(w3)
# CHECK-NEXT:  sch[O_W0].unroll(h3)
# CHECK-NEXT:  sch[O_W0].unroll(b3)
# CHECK-NEXT:  sch[O_W0].unroll(c1)
# CHECK-NEXT:  sch[O_W0].unroll(s1)
# CHECK-NEXT:  sch[O_W0].unroll(r1)
# CHECK-NEXT:  sch[O_W0].vectorize(f3)
# CHECK-NEXT:  
# CHECK-NEXT:  sample 0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 2: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 4: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 5: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 6: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 7: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 8: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 9: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 10: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 11: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 12: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 13: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 14: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 15: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 16: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 0]
# CHECK-NEXT:  sample 17: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1]
# CHECK-NEXT:  sample 18: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 0]
# CHECK-NEXT:  sample 19: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 1]
# CHECK-NEXT:  sample 20: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 7, 1, 0]
# CHECK-NEXT:  sample 21: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 7, 1, 1]
# CHECK-NEXT:  sample 22: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 1, 1, 0]
# CHECK-NEXT:  sample 23: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 1, 1, 1]
# CHECK-NEXT:  sample 24: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 0]
# CHECK-NEXT:  sample 25: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1]
# CHECK-NEXT:  sample 26: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 3, 0]
# CHECK-NEXT:  sample 27: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 3, 1]
# CHECK-NEXT:  sample 28: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 7, 1, 0]
# CHECK-NEXT:  sample 29: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 7, 1, 1]
# CHECK-NEXT:  sample 30: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 1, 1, 0]
# CHECK-NEXT:  sample 31: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 1, 1, 1]
# CHECK-NEXT:  sample 32: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 0]
# CHECK-NEXT:  sample 33: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1]
# CHECK-NEXT:  sample 34: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 3, 0]
# CHECK-NEXT:  sample 35: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 3, 1]
# CHECK-NEXT:  sample 36: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 7, 1, 0]
# CHECK-NEXT:  sample 37: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 7, 1, 1]
# CHECK-NEXT:  sample 38: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 7, 1, 1, 0]
# CHECK-NEXT:  sample 39: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 7, 1, 1, 1]
# CHECK-NEXT:  sample 40: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 0]
# CHECK-NEXT:  sample 41: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1]
# CHECK-NEXT:  sample 42: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 3, 0]
# CHECK-NEXT:  sample 43: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 3, 1]
# CHECK-NEXT:  sample 44: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 45: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 46: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 47: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 48: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 49: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 50: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 51: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 52: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 53: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 54: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 55: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 56: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 57: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 58: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 59: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 60: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 0]
# CHECK-NEXT:  sample 61: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1]
# CHECK-NEXT:  sample 62: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 3, 0]
# CHECK-NEXT:  sample 63: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 3, 1]
# CHECK-NEXT:  sample 64: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 7, 1, 0]
# CHECK-NEXT:  sample 65: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 7, 1, 1]
# CHECK-NEXT:  sample 66: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 7, 1, 1, 0]
# CHECK-NEXT:  sample 67: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 7, 1, 1, 1]
# CHECK-NEXT:  sample 68: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 1, 1, 0]
# CHECK-NEXT:  sample 69: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 1, 1, 1]
# CHECK-NEXT:  sample 70: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 1, 3, 0]
# CHECK-NEXT:  sample 71: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 1, 3, 1]
# CHECK-NEXT:  sample 72: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 7, 1, 0]
# CHECK-NEXT:  sample 73: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 7, 1, 1]
# CHECK-NEXT:  sample 74: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 7, 1, 1, 0]
# CHECK-NEXT:  sample 75: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 7, 1, 1, 1]
# CHECK-NEXT:  sample 76: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 1, 1, 0]
# CHECK-NEXT:  sample 77: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 1, 1, 1]
# CHECK-NEXT:  sample 78: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 1, 3, 0]
# CHECK-NEXT:  sample 79: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 1, 3, 1]
# CHECK-NEXT:  sample 80: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 7, 1, 0]
# CHECK-NEXT:  sample 81: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 1, 7, 1, 1]
# CHECK-NEXT:  sample 82: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 7, 1, 1, 0]
# CHECK-NEXT:  sample 83: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 16, 7, 1, 1, 1]
# CHECK-NEXT:  sample 84: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 85: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 86: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 87: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 88: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 89: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 90: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 91: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 92: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 93: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 94: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 95: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 96: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 97: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 98: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 99: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 100: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 0]
# CHECK-NEXT:  sample 101: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1]
# CHECK-NEXT:  sample 102: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 3, 0]
# CHECK-NEXT:  sample 103: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 3, 1]
# CHECK-NEXT:  sample 104: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 7, 1, 0]
# CHECK-NEXT:  sample 105: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 7, 1, 1]
# CHECK-NEXT:  sample 106: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 7, 1, 1, 0]
# CHECK-NEXT:  sample 107: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 7, 1, 1, 1]
# CHECK-NEXT:  sample 108: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 1, 1, 0]
# CHECK-NEXT:  sample 109: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 1, 1, 1]
# CHECK-NEXT:  sample 110: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 1, 3, 0]
# CHECK-NEXT:  sample 111: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 1, 3, 1]
# CHECK-NEXT:  sample 112: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 7, 1, 0]
# CHECK-NEXT:  sample 113: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 7, 1, 1]
# CHECK-NEXT:  sample 114: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 7, 1, 1, 0]
# CHECK-NEXT:  sample 115: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 7, 1, 1, 1]
# CHECK-NEXT:  sample 116: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 117: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 118: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 119: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 120: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 121: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 122: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 123: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 124: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 125: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 126: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 127: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 128: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 129: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 130: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 131: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 132: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 1, 1, 0]
# CHECK-NEXT:  sample 133: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 1, 1, 1]
# CHECK-NEXT:  sample 134: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 1, 3, 0]
# CHECK-NEXT:  sample 135: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 1, 3, 1]
# CHECK-NEXT:  sample 136: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 7, 1, 0]
# CHECK-NEXT:  sample 137: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 1, 7, 1, 1]
# CHECK-NEXT:  sample 138: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 7, 1, 1, 0]
# CHECK-NEXT:  sample 139: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 7, 1, 1, 1]
# CHECK-NEXT:  sample 140: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 141: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 142: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 143: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 144: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 145: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 146: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 147: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 148: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 149: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 150: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 151: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 152: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 153: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 154: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 155: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 156: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 157: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 158: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 159: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 160: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 161: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 162: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 163: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 164: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0]
# CHECK-NEXT:  sample 165: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
# CHECK-NEXT:  sample 166: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 0]
# CHECK-NEXT:  sample 167: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1]
# CHECK-NEXT:  sample 168: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 7, 1, 0]
# CHECK-NEXT:  sample 169: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 7, 1, 1]
# CHECK-NEXT:  sample 170: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 7, 1, 1, 0]
# CHECK-NEXT:  sample 171: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 7, 1, 1, 1]
# CHECK-NEXT:  sample 172: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 0]
# CHECK-NEXT:  sample 173: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1]
# CHECK-NEXT:  sample 174: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 0]
# CHECK-NEXT:  sample 175: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1]
# CHECK-NEXT:  sample 176: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 7, 1, 0]
# CHECK-NEXT:  sample 177: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 7, 1, 1]
# CHECK-NEXT:  sample 178: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 7, 1, 1, 0]
# CHECK-NEXT:  sample 179: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 7, 1, 1, 1]
# CHECK-NEXT:  sample 180: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 1, 0]
# CHECK-NEXT:  sample 181: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 1, 1]
# CHECK-NEXT:  sample 182: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 3, 0]
# CHECK-NEXT:  sample 183: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 3, 1]
# CHECK-NEXT:  sample 184: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 7, 1, 0]
# CHECK-NEXT:  sample 185: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 7, 1, 1]
# CHECK-NEXT:  sample 186: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 7, 1, 1, 0]
# CHECK-NEXT:  sample 187: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 7, 1, 1, 1]
# CHECK-NEXT:  sample 188: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 1, 1, 0]
# CHECK-NEXT:  sample 189: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 1, 1, 1]
# CHECK-NEXT:  sample 190: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 1, 3, 0]
# CHECK-NEXT:  sample 191: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 1, 3, 1]
# CHECK-NEXT:  sample 192: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 7, 1, 0]
# CHECK-NEXT:  sample 193: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 1, 7, 1, 1]
# CHECK-NEXT:  sample 194: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 7, 1, 1, 0]
# CHECK-NEXT:  sample 195: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 8, 7, 1, 1, 1]
# CHECK-NEXT:  sample 196: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 16, 1, 1, 1, 0]
# CHECK-NEXT:  sample 197: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 16, 1, 1, 1, 1]
# CHECK-NEXT:  sample 198: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 16, 1, 1, 3, 0]
# CHECK-NEXT:  sample 199: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 16, 1, 1, 3, 1]
# CHECK-NEXT:  stats {'filtered': 200, 'all': 404}
# CHECK-NEXT:  O = obj['%2']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "local")
# CHECK-NEXT:  b, h, w, f, = O.op.axis
# CHECK-NEXT:  r, s, c, = O.op.reduce_axis
# CHECK-NEXT:  b, b1 = sch[O].split(b, factor=1)
# CHECK-NEXT:  h, h1 = sch[O].split(h, factor=1)
# CHECK-NEXT:  w, w1 = sch[O].split(w, factor=1)
# CHECK-NEXT:  f, f1 = sch[O].split(f, factor=32)
# CHECK-NEXT:  b1, b_ = sch[O].split(b1, factor=1)
# CHECK-NEXT:  h1, h_ = sch[O].split(h1, factor=1)
# CHECK-NEXT:  w1, w_ = sch[O].split(w1, factor=1)
# CHECK-NEXT:  f1, f_ = sch[O].split(f1, factor=16)
# CHECK-NEXT:  sch[O].reorder(b, h, w, f, b1, h1, w1, f1, b_, h_, w_, f_)
# CHECK-NEXT:  sch[O].parallel(b)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], f1)
# CHECK-NEXT:  b, h, w, f, = O_W0.op.axis
# CHECK-NEXT:  r, s, c, = O_W0.op.reduce_axis
# CHECK-NEXT:  b2 = b
# CHECK-NEXT:  h2 = h
# CHECK-NEXT:  w2 = w
# CHECK-NEXT:  f2 = f
# CHECK-NEXT:  r, r1 = sch[O_W0].split(r, factor=1)
# CHECK-NEXT:  s, s1 = sch[O_W0].split(s, factor=1)
# CHECK-NEXT:  c, c1 = sch[O_W0].split(c, factor=3)
# CHECK-NEXT:  b2, b3 = sch[O_W0].split(b2, factor=1)
# CHECK-NEXT:  h2, h3 = sch[O_W0].split(h2, factor=1)
# CHECK-NEXT:  w2, w3 = sch[O_W0].split(w2, factor=1)
# CHECK-NEXT:  f2, f3 = sch[O_W0].split(f2, factor=16)
# CHECK-NEXT:  sch[O_W0].reorder(r, s, c, b2, h2, w2, f2, r1, s1, c1, b3, h3, w3, f3)
# CHECK-NEXT:  sch[O_W0].unroll(f3)
# CHECK-NEXT:  sch[O_W0].unroll(w3)
# CHECK-NEXT:  sch[O_W0].unroll(h3)
# CHECK-NEXT:  sch[O_W0].unroll(b3)
# CHECK-NEXT:  sch[O_W0].unroll(c1)
# CHECK-NEXT:  sch[O_W0].unroll(s1)
# CHECK-NEXT:  sch[O_W0].unroll(r1)
# CHECK-NEXT:  sch[O_W0].vectorize(f3)
