# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

img = O.tensor()
w1 = O.tensor()
w2 = O.tensor()
w3 = O.tensor()
w4 = O.tensor()

fc = lambda i, w, nout: O.matmul(O.reshape(i, shape=(1, -1)), O.reshape(w, shape=(-1, nout)))

# Mulit Layer Perceptron with 3 relu(fc) + 1 fc
with O.graph(name="mlp4") as gb:
    with O.graph(name="l1"):
        l1 = O.relu(fc(img, w1, 512))
    with O.graph(name="l2"):
        l2 = O.relu(fc(l1, w2, 256))
    with O.graph(name="l3"):
        l3 = O.relu(fc(l2, w3, 128))
    with O.graph(name="l4"):
        l4 = fc(l3, w4, 10)
    O.reshape(l4, shape=(-1,))

mlp4 = gb.graph
print(mlp4)

import xtc.graphs.xtc.ty as T

inp_types = [
    T.TensorType((32, 32, 3), "float32"),
    T.TensorType((32*32*3, 512), "float32"),
    T.TensorType((512, 256), "float32"),
    T.TensorType((256, 128), "float32"),
    T.TensorType((128, 10), "float32"),
]

out_types = mlp4.forward_types(inp_types)
print(out_types)

from xtc.utils.numpy import np_init

inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)-5) for t in inp_types]
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
# CHECK-NEXT:    - %20
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %5: reshape(%0, shape=(1, -1))
# CHECK-NEXT:      %6: reshape(%1, shape=(-1, 512))
# CHECK-NEXT:      %7: matmul(%5, %6)
# CHECK-NEXT:      %8: relu(%7)
# CHECK-NEXT:      %9: reshape(%8, shape=(1, -1))
# CHECK-NEXT:      %10: reshape(%2, shape=(-1, 256))
# CHECK-NEXT:      %11: matmul(%9, %10)
# CHECK-NEXT:      %12: relu(%11)
# CHECK-NEXT:      %13: reshape(%12, shape=(1, -1))
# CHECK-NEXT:      %14: reshape(%3, shape=(-1, 128))
# CHECK-NEXT:      %15: matmul(%13, %14)
# CHECK-NEXT:      %16: relu(%15)
# CHECK-NEXT:      %17: reshape(%16, shape=(1, -1))
# CHECK-NEXT:      %18: reshape(%4, shape=(-1, 10))
# CHECK-NEXT:      %19: matmul(%17, %18)
# CHECK-NEXT:      %20: reshape(%19, shape=(-1,))
# CHECK-NEXT:  
# CHECK-NEXT:  [TensorType(shape=(10,), dtype=float32)]
# CHECK-NEXT:  Inputs: [Tensor(type=TensorType(shape=(32, 32, 3), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...4.0 -4.0 -3.0 -2.0), Tensor(type=TensorType(shape=(3072, 512), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-2.0 -1.0 0.0 1.0), Tensor(type=TensorType(shape=(512, 256), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...-3.0 -2.0 -1.0 0.0), Tensor(type=TensorType(shape=(256, 128), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...0.0 1.0 2.0 3.0), Tensor(type=TensorType(shape=(128, 10), dtype=float32), data=-4.0 -3.0 -2.0 -1.0...3.0 4.0 -4.0 -3.0)]
# CHECK-NEXT:  Outputs: [Tensor(type=TensorType(shape=(10,), dtype=float32), data=35636482000.0 52931383000.0 10267247000.0 -26914288000.0...-16248222000.0 1046679500.0 18341579000.0 35636470000.0)]
