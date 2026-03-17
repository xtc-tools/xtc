# RUN: python %s 2>&1 | filecheck %s --check-prefix=CHECK-VALID
# RUN: not python %s --unused-dim 2>&1 | filecheck %s --check-prefix=CHECK-UNUSED-DIM
# RUN: not python %s --vect-inconsistency 2>&1 | filecheck %s --check-prefix=CHECK-VECT
# RUN: not python %s --tile-before-axis 2>&1 | filecheck %s --check-prefix=CHECK-ORDER
# RUN: not python %s --tile-too-large 2>&1 | filecheck %s --check-prefix=CHECK-SIZE

import sys
import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph


def make_scheduler():
    impl = Backend(graph)
    return impl.get_scheduler()


if len(sys.argv) == 1:
    sch = make_scheduler()
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 2})
    sch.tile("J", {"J0": 16})
    sch.interchange(["K", "I", "J", "I0", "J0"])
    sch.vectorize(["J0"])

    loop_nest = sch.get_loop_nest()
    loop_nest.check()
    print("ok")

# CHECK-VALID: ok

elif "--unused-dim" in sys.argv:
    sch = make_scheduler()
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 2})
    sch.interchange(["I", "J", "I0"])

    loop_nest = sch.get_loop_nest()
    loop_nest.check()

# CHECK-UNUSED-DIM: K defined but never used

elif "--vect-inconsistency" in sys.argv:
    sch = make_scheduler()
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 2})
    sch.tile("J", {"J0": 16})
    sch.interchange(["K", "I", "J", "J0", "I0"])
    sch.vectorize(["J0"])

    loop_nest = sch.get_loop_nest()
    loop_nest.check()

# CHECK-VECT: Inner loop I0 isn't vectorized but an outer one is.

elif "--tile-before-axis" in sys.argv:
    sch = make_scheduler()
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 2})
    sch.tile("J", {"J0": 16})
    sch.interchange(["K", "I0", "I", "J", "J0"])

    loop_nest = sch.get_loop_nest()
    loop_nest.check()

# CHECK-ORDER: `I#2`: I has not been materialized yet.

elif "--tile-too-large" in sys.argv:
    sch = make_scheduler()
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 4})
    sch.tile("I", {"I00": 8})
    sch.tile("J", {"J0": 16})
    sch.interchange(["K", "I", "J", "I0", "I00", "J0"])

    loop_nest = sch.get_loop_nest()
    loop_nest.check()

# CHECK-SIZE: Inner loop I00 on axis I must be smaller than outer loop.
