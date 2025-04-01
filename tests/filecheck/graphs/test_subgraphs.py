# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()

# Create a whole graph and two subviews at the same time
# from nested graph builders
with O.graph(name="matmul_relu") as mrb:
    with O.graph(name="matmul") as mb:
        z = O.matmul(x, y)
    with O.graph(name="relu") as rb:
        O.relu(z)

matmul_relu = mrb.graph
print(matmul_relu)
print(mb.graph)
print(rb.graph)
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
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %2: matmul(%0, %1)
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %3: relu(%2)
