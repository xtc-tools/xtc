# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

i = O.tensor()
k = O.tensor()

with O.graph(name="conv_relu") as gb:
    p = O.pad2d(i, padding=1)
    c = O.conv2d(p, k, stride=(2, 2))
    O.relu(c, threshold=0.1)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((2, 4, 4, 3), "float32"),
    T.TensorType((3, 3, 3, 8), "float32"),
]
out_types = graph.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5) for t in inp_types]
print(f"Inputs: {inps}")
outs = graph.forward(inps)
print(f"Outputs: {outs}")

# CHECK:       graph:
# CHECK-NEXT:    name: conv_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding=1)
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2))
# CHECK-NEXT:    - %4: relu(%3, threshold=0.1)
# CHECK-NEXT:  
# CHECK-NEXT:  [2x2x2x8xfloat32]
# CHECK-NEXT:  Inputs: [Tensor(type=2x4x4x3xfloat32, data=-4 -3 -2 -1...-2 -1 0 1), Tensor(type=3x3x3x8xfloat32, data=-4 -3 -2 -1...1 2 3 4)]
# CHECK-NEXT:  Outputs: [Tensor(type=2x2x2x8xfloat32, data=0.1 0.1 0.1 10...9 0.1 9 9)]
