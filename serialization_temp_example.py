from pathlib import Path
import xtc.graphs.xtc.op as O

if not Path("output.yaml").exists():
    I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")
    c = O.tensor((J, I), dtype, name="C")

    with O.graph(name = "matmul_relu") as gb:
        m = O.matmul(a, b, name="M")
        q = O.relu(m, threshold=.1)
        r = O.relu(m, threshold=.1)
        k = O.matmul(c, r, name="K")

    graph = gb.graph
    print(graph)
    graph.dump("output.yaml")

print("loading from yaml....")

with O.graph(name = "matmul_relu") as gb2:
    gb2.load("output.yaml")    
    
print(gb2.graph)
