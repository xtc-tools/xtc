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
# CHECK-NEXT:    - %0 : 1x3072xfloat32
# CHECK-NEXT:    - %1 : 3072x512xfloat32
# CHECK-NEXT:    - %2 : 512x256xfloat32
# CHECK-NEXT:    - %3 : 256x128xfloat32
# CHECK-NEXT:    - %4 : 128x10xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %11 : 1x10xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %5: matmul(%0, %1) : [1x3072xfloat32, 3072x512xfloat32] -> [1x512xfloat32]
# CHECK-NEXT:    - %6: relu(%5) : [1x512xfloat32] -> [1x512xfloat32]
# CHECK-NEXT:    - %7: matmul(%6, %2) : [1x512xfloat32, 512x256xfloat32] -> [1x256xfloat32]
# CHECK-NEXT:    - %8: relu(%7) : [1x256xfloat32] -> [1x256xfloat32]
# CHECK-NEXT:    - %9: matmul(%8, %3) : [1x256xfloat32, 256x128xfloat32] -> [1x128xfloat32]
# CHECK-NEXT:    - %10: relu(%9) : [1x128xfloat32] -> [1x128xfloat32]
# CHECK-NEXT:    - %11: matmul(%10, %4) : [1x128xfloat32, 128x10xfloat32] -> [1x10xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  Inputs: [Tensor(type=1x3072xfloat32, data=-4 -3 -2 -1...4 -4 -3 -2), Tensor(type=3072x512xfloat32, data=-4 -3 -2 -1...-2 -1 0 1), Tensor(type=512x256xfloat32, data=-4 -3 -2 -1...-3 -2 -1 0), Tensor(type=256x128xfloat32, data=-4 -3 -2 -1...0 1 2 3), Tensor(type=128x10xfloat32, data=-4 -3 -2 -1...3 4 -4 -3)]
# CHECK-NEXT:  Outputs: [Tensor(type=1x10xfloat32, data=3.56365e+10 5.29314e+10 1.02672e+10 -2.69143e+10...-1.62482e+10 1.04668e+09 1.83416e+10 3.56365e+10)]
