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
print(f"outputs: {out_types}")
print(graph)


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
# CHECK-NEXT:    - %2: matmul(%0, %1)
# CHECK-NEXT:    - %3: relu(%2)
# CHECK-NEXT:  
# CHECK-NEXT:  outputs: [5x4xfloat32]
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 5x3xfloat32
# CHECK-NEXT:    - %1 : 3x4xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 5x4xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) : [5x3xfloat32, 3x4xfloat32] -> [5x4xfloat32]
# CHECK-NEXT:    - %3: relu(%2) : [5x4xfloat32] -> [5x4xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  Inputs: [Tensor(type=5x3xfloat32, data=-4 -3 -2 -1...-2 -1 0 1), Tensor(type=3x4xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2)]
# CHECK-NEXT:  Outputs: [Tensor(type=5x4xfloat32, data=8 17 8 0...8 0 0 0)]
