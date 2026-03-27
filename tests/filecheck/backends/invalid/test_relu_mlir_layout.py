# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A", layout=[1, 0])

with O.graph(name="relu_layout") as gb:
    O.relu(a, name="relu")

graph = gb.graph
Backend(graph)

# CHECK: NotImplementedError: tensor layout is not yet implemented in MLIR backend
# XFAIL: *
