# RUN: python %s --mlir 2>&1 | filecheck %s --check-prefix=CHECK-MLIR
# RUN: python %s --tvm 2>&1 | filecheck %s --check-prefix=CHECK-TVM
# REQUIRES: module_tvm

import sys
import xtc.graphs.xtc.op as O

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph

if "--mlir" in sys.argv:
    from xtc.backends.mlir import Backend

elif "--tvm" in sys.argv:
    from xtc.backends.tvm import Backend

else:
    assert False
    
impl = Backend(graph)
sch = impl.get_scheduler()
sch.set_dims(["I", "J", "K"])
sch.tile("I", {"I0": 2})
sch.tile("J", {"J0": 16})
sch.interchange(["K", "I", "J", "I0", "J0"])
sch.unroll({"I0": 2})
sch.vectorize(["J0"])
if "--tvm" in sys.argv:
    sch.buffer_at("J")
    sch.pack_at("I", 0, pad=True)

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())

# CHECK-MLIR:      loop K
# CHECK-MLIR-NEXT:   loop I
# CHECK-MLIR-NEXT:     loop J
# CHECK-MLIR-NEXT:       tile(I, 2)  // unroll(2)
# CHECK-MLIR-NEXT:         tile(J, 16)  // vectorized
# CHECK-MLIR-NEXT:           ...

# CHECK-TVM:      loop K
# CHECK-TVM-NEXT:   loop I  // pack(0, pad)
# CHECK-TVM-NEXT:     loop J  // buffer
# CHECK-TVM-NEXT:       tile(I, 2)  // unroll(2)
# CHECK-TVM-NEXT:         tile(J, 16)  // vectorized
# CHECK-TVM-NEXT:           ...

# Test with split (MLIR only - TVM does not support split)
if "--mlir" in sys.argv:
    print("---")

    impl2 = Backend(graph)
    sch2 = impl2.get_scheduler()
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
