# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

i = O.tensor() # format NHWC
k = O.tensor() # format OHWI

with O.graph(name="conv_relu") as gb:
    p = O.pad2d(i, padding=2, axis=(1, 2))
    t = O.transpose(k, axes=(1, 2, 3, 0))
    c = O.conv2d(p, t, stride=(2, 2))
    O.relu(c, threshold=0.1)

graph = gb.graph
print(graph)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((2, 6, 6, 3), "float32"),
    T.TensorType((8, 5, 5, 3), "float32"),
]
out_types = graph.forward_types(inp_types)
print(graph)

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
# CHECK-NEXT:    - %5
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding=(2, 2, 2, 2), axis=(1, 2), constant_value=0)
# CHECK-NEXT:    - %3: transpose(%1, axes=(1, 2, 3, 0))
# CHECK-NEXT:    - %4: conv2d(%2, %3, stride=(2, 2))
# CHECK-NEXT:    - %5: relu(%4, threshold=0.1)
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 2x6x6x3xfloat32
# CHECK-NEXT:    - %1 : 8x5x5x3xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %5 : 2x3x3x8xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding=(2, 2, 2, 2), axis=(1, 2), constant_value=0) : [2x6x6x3xfloat32] -> [2x10x10x3xfloat32]
# CHECK-NEXT:    - %3: transpose(%1, axes=(1, 2, 3, 0)) : [8x5x5x3xfloat32] -> [5x5x3x8xfloat32]
# CHECK-NEXT:    - %4: conv2d(%2, %3, stride=(2, 2)) : [2x10x10x3xfloat32, 5x5x3x8xfloat32] -> [2x3x3x8xfloat32]
# CHECK-NEXT:    - %5: relu(%4, threshold=0.1) : [2x3x3x8xfloat32] -> [2x3x3x8xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  Inputs: [Tensor(type=2x6x6x3xfloat32, data=-4 -3 -2 -1...1 2 3 4), Tensor(type=8x5x5x3xfloat32, data=-4 -3 -2 -1...-2 -1 0 1)]
# CHECK-NEXT:  Outputs: [Tensor(type=2x3x3x8xfloat32, data=18 18 18 18...5 113 0.1 5)]
