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
# CHECK-NEXT:      %2: pad2d(%0, padding=1)
# CHECK-NEXT:      %3: conv2d(%2, %1, stride=(2, 2))
# CHECK-NEXT:      %4: relu(%3, threshold=0.1)
# CHECK-NEXT:  
# CHECK-NEXT:  [TensorType(shape=(2, 2, 2, 8), dtype=float32)]
# CHECK-NEXT:  Inputs: [Tensor(type=TensorType(shape=(2, 4, 4, 3), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-2.0 -1.0 0.0 1.0), Tensor(type=TensorType(shape=(3, 3, 3, 8), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...1.0 2.0 3.0 4.0)]
# CHECK-NEXT:  Outputs: [Tensor(type=TensorType(shape=(2, 2, 2, 8), dtype=float32), data=0.1 0.1 0.1 10.0...9.0 0.1 9.0 9.0)]
