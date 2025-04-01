#!/usr/bin/env python3

import numpy as np
import numpy.typing
import functools, operator
import xtc.graphs.xtc.op as O
import xtc.graphs.xtc.ty as T


def np_init(shape: tuple[int, ...], dtype: str) -> numpy.typing.NDArray:
    elts = functools.reduce(operator.mul, shape, 1)
    return (((np.arange(elts) % 21) - 10) / 10).reshape(shape).astype(dtype)


def main():
    x = O.tensor()
    y = O.tensor()
    print(x, y)
    with O.graph(name="matmul_relu") as gb:
        with O.graph(name="matmul") as mb:
            O.inputs(y, x)
            m = O.matmul(x, y)
        with O.graph(name="relu") as rb:
            O.relu(m)

    print(gb.graph)
    print(mb.graph)
    print(rb.graph)
    graph = gb.graph
    inp_dims = [
        T.TensorType(shape=(5, 3), dtype="float32"),
        T.TensorType(shape=(3, 4), dtype="float32"),
    ]
    out_dims = graph.forward_types(inp_dims)
    print(f"Inputs: {graph.inputs}")
    print(f"Outputs: {graph.outputs}")
    print(f"Inputs types: {[str(x) for x in inp_dims]}")
    print(f"Outputs types: {[str(x) for x in out_dims]}")
    inps = [T.Tensor(np_init(t.constant_shape, t.constant_dtype)) for t in inp_dims]
    outs = graph.forward(inps)
    print(f"Inputs: {inps}")
    print(f"Outputs: {outs}")


if __name__ == "__main__":
    main()
