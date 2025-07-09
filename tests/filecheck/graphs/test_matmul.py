# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()

with O.graph(name="matmul") as gb:
    O.matmul(x, y)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((5,3), "float32"),
    T.TensorType((3,4), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1)
# CHECK-NEXT:  
# CHECK-NEXT:  [5x4xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=5x3xfloat32, data=-4 -3 -2 -1...-2 -1 0 1), Tensor(type=3x4xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2)]
# CHECK-NEXT:  Outputs: [Tensor(type=5x4xfloat32, data=8 17 8 -1...8 -1 -1 -1)]
