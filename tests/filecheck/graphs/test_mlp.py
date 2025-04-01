# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

img_size = 32*32*3
img = O.tensor(shape=(1, img_size), dtype="float32")
w1 = O.tensor(shape=(img_size, 512), dtype="float32")
w2 = O.tensor(shape=(512, 256), dtype="float32")
w3 = O.tensor(shape=(256, 128), dtype="float32")
w4 = O.tensor(shape=(128, 10), dtype="float32")

# Mulit Layer Perceptron with 3 relu(fc) + 1 fc
with O.graph(name="mlp4") as gb:
    with O.graph(name="l1"):
        l1 = O.matmul(img, w1)
        l1 = O.relu(l1)
    with O.graph(name="l2"):
        l2 = O.matmul(l1, w2)
        l2 = O.relu(l2)
    with O.graph(name="l3"):
        l3 = O.matmul(l2, w3)
        l3 = O.relu(l3)
    with O.graph(name="l4"):
        O.matmul(l3, w4)

mlp4 = gb.graph
print(mlp4)

import xtc.graphs.xtc.ty as T
from xtc.utils.numpy import np_init

inps = [
    T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5)
    for t in [inp.forward_types([])[0] for inp in mlp4.inputs_nodes]
]

print(f"Inputs: {inps}")
outs = mlp4.forward(inps)
print(f"Outputs: {outs}")


# CHECK:       graph:
# CHECK-NEXT:    name: mlp4
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    - %2
# CHECK-NEXT:    - %3
# CHECK-NEXT:    - %4
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %11
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %5: matmul(%0, %1)
# CHECK-NEXT:      %6: relu(%5)
# CHECK-NEXT:      %7: matmul(%6, %2)
# CHECK-NEXT:      %8: relu(%7)
# CHECK-NEXT:      %9: matmul(%8, %3)
# CHECK-NEXT:      %10: relu(%9)
# CHECK-NEXT:      %11: matmul(%10, %4)
# CHECK-NEXT:  
# CHECK-NEXT:  Inputs: [Tensor(type=TensorType(shape=(1, 3072), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...4.0 -4.0 -3.0 -2.0), Tensor(type=TensorType(shape=(3072, 512), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-2.0 -1.0 0.0 1.0), Tensor(type=TensorType(shape=(512, 256), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-3.0 -2.0 -1.0 0.0), Tensor(type=TensorType(shape=(256, 128), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...0.0 1.0 2.0 3.0), Tensor(type=TensorType(shape=(128, 10), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...3.0 4.0 -4.0 -3.0)]
# CHECK-NEXT:  Outputs: [Tensor(type=TensorType(shape=(1, 10), dtype=float32), data=35636482000.0 52931383000.0 10267247000.0 -26914288000.0...-16248222000.0 1046679500.0 18341579000.0 35636470000.0)]
