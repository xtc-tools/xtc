import tempfile
import xtc.graphs.xtc.op as O

def test_matmul_relu_to_from_dict():
    I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")

    with O.graph(name="matmul_relu") as gb:
        m = O.matmul(a, b, name="matmul")
        O.relu(m, name="relu")

    graph_dict = gb.graph.to_dict()
    with O.graph() as gb2:
        gb2.from_dict(graph_dict)
    assert graph_dict != {}
    assert graph_dict == gb2.graph.to_dict()
        

def test_conv2d_pad_sdump_sload():
    N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
    a = O.tensor((N, H, W, C), dtype, name="I")
    b = O.tensor((R, S, C, F), dtype, name="W")

    with O.graph(name="pad_conv2d_nhwc_mini") as gb:
        p = O.pad(a, padding={1: (2), 2: (2, 2)}, name="pad")
        O.conv2d(p, b, stride=(SH, SW), name="conv")

    graph_str = gb.graph.dumps()
    with O.graph(name="matmul_relu") as gb2:
        gb2.loads(graph_str)
    assert graph_str != ""
    assert graph_str == gb2.graph.dumps()

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        gb.graph.dump(f.name)
        with O.graph() as gb3:
            gb3.load(f.name)
        assert gb.graph.to_dict() == gb3.graph.to_dict()

def test_mlp_fc_custom_output():
    img = O.tensor()
    w1 = O.tensor()
    w2 = O.tensor()
    w3 = O.tensor()
    w4 = O.tensor()
    fc = lambda i, w, nout: O.matmul(O.reshape(i, shape=(1, -1)), O.reshape(w, shape=(-1, nout)))
    # Multi Layer Perceptron with 3 relu(fc) + 1 fc
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
        O.outputs(l1)
   
    graph_dict = gb.graph.to_dict()
    with O.graph() as gb2:
        gb2.from_dict(graph_dict)
    assert graph_dict != {}
    assert graph_dict == gb2.graph.to_dict()
    
