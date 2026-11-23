# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()
z = O.tensor()

with O.graph(name="add_mul") as gb:
    a = O.add(x, y)
    O.mul(a, z)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((2, 4), "float32"),
    T.TensorType((4,), "float32"),
    T.TensorType((2, 1), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")
# CHECK:       graph:
# CHECK-NEXT:    name: add_mul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    - %2
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %3: add(%0, %1)
# CHECK-NEXT:    - %4: mul(%3, %2)
# CHECK-NEXT:  
# CHECK-NEXT:  [2x4xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=2x4xfloat32, data=-1 -0.875 -0.75 -0.625 -0.5 -0.375 -0.25 -0.125), Tensor(type=4xfloat32, data=-1 -0.875 -0.75 -0.625), Tensor(type=2x1xfloat32, data=-1 -0.875)]
# CHECK-NEXT:  Outputs: [Tensor(type=2x4xfloat32, data=2 1.75 1.5 1.25 1.3125 1.09375 0.875 0.65625)]
