# RUN: python %s --mlir 2>&1 | filecheck %s --check-prefix=CHECK-MLIR
# RUN: python %s --tvm 2>&1 | filecheck %s --check-prefix=CHECK-TVM
# REQUIRES: module_tvm

import sys
import xtc.graphs.xtc.op as O

I, J, K, dtype = 128, 256, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    p = O.relu(a, name="relu")
    m = O.matmul(p, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph

if "--mlir" in sys.argv:
    from xtc.backends.mlir import Backend

elif "--tvm" in sys.argv:
    from xtc.backends.tvm import Backend

else:
    assert False

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul")
sch.set_dims(["I", "J", "K"])
sch.tile("I", {"I1": 64, "I0": 2})
sch.tile("J", {"J1": 128, "J0": 16})
sch.tile("K", {"K0": 32})
sch.interchange(["J", "K", "I", "J1", "I1", "K0", "I0", "J0"])
sch.unroll({"K0": 8, "I0": 2})
sch.vectorize(["J0"])
sch.parallelize(["J"])
if "--tvm" in sys.argv:
    sch.buffer_at("J")
    sch.pack_at("K", 1, pad=True)
    sch.fuse_producer_at("I", 0)
    sch.fuse_consumer_at("J")

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())

# CHECK-MLIR:      loop J // parallelized
# CHECK-MLIR-NEXT:   loop K
# CHECK-MLIR-NEXT:     loop I
# CHECK-MLIR-NEXT:       tile(J, 128)
# CHECK-MLIR-NEXT:         tile(I, 64)
# CHECK-MLIR-NEXT:           tile(K, 32)  // unroll(8)
# CHECK-MLIR-NEXT:             tile(I, 2)  // unroll(2)
# CHECK-MLIR-NEXT:               tile(J, 16)  // vectorized
# CHECK-MLIR-NEXT:                 ...

# CHECK-TVM:      loop J // parallelized, buffer, fuse_consumer
# CHECK-TVM-NEXT:   loop K  // pack(1, pad)
# CHECK-TVM-NEXT:     loop I  // fuse_producer(0)
# CHECK-TVM-NEXT:       tile(J, 128)
# CHECK-TVM-NEXT:         tile(I, 64)
# CHECK-TVM-NEXT:           tile(K, 32)  // unroll(8)
# CHECK-TVM-NEXT:             tile(I, 2)  // unroll(2)
# CHECK-TVM-NEXT:               tile(J, 16)  // vectorized
# CHECK-TVM-NEXT:                 ...

# Test with split (MLIR only - TVM does not support split)
if "--mlir" in sys.argv:
    print("---")

    impl2 = Backend(graph)
    sch2 = impl2.get_scheduler(default_node="matmul")
    sch2.set_dims(["I", "J", "K"])
    sch2.split("I", {"I_lo": 0, "I_hi": 2})
    sch2.tile("J", {"J0": 16}, root="./I_lo")
    sch2.tile("J", {"J0": 16}, root="./I_hi")
    sch2.interchange(["K", "I_lo", "I_hi"])
    sch2.interchange(["J", "J0"], root="./I_lo")
    sch2.interchange(["J", "J0"], root="./I_hi")

    loop_nest2 = sch2.get_loop_nest()
    print(loop_nest2.root_node.pretty_print())

# CHECK-MLIR:      ---
# CHECK-MLIR-NEXT: loop K
# CHECK-MLIR-NEXT:   split(I, 0, 2)
# CHECK-MLIR-NEXT:     loop J
# CHECK-MLIR-NEXT:       tile(J, 16)
# CHECK-MLIR-NEXT:         ...
# CHECK-MLIR-NEXT:   split(I, 2, ...)
# CHECK-MLIR-NEXT:     loop J
# CHECK-MLIR-NEXT:       tile(J, 16)
# CHECK-MLIR-NEXT:         ...
