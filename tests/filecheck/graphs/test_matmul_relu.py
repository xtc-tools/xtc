# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()

with O.graph(name="matmul_relu") as gb:
    z = O.matmul(x, y)
    O.relu(z)

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
# CHECK-NEXT:    name: matmul_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %2: matmul(%0, %1)
# CHECK-NEXT:      %3: relu(%2)
# CHECK-NEXT:  
# CHECK-NEXT:  [TensorType(shape=(5, 4), dtype=float32)]
# CHECK-NEXT:  Inputs: [Tensor(type=TensorType(shape=(5, 3), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-2.0 -1.0 0.0 1.0), Tensor(type=TensorType(shape=(3, 4), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...4.0 -4.0 -3.0 -2.0)]
# CHECK-NEXT:  Outputs: [Tensor(type=TensorType(shape=(5, 4), dtype=float32), data=8.0 17.0 8.0 0.0...8.0 0.0 0.0 0.0)]
