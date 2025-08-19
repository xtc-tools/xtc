
def requires_tvm(*arg):
    import pytest
    def has_tvm():
        try:
            import tvm
            return True
        except:
            return False
    return pytest.mark.skipif(not has_tvm(), reason="requires TVM")(*arg)


def matmul_impl(i, j, k, dtype, name):
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend

    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")

    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")

    return TVMBackend(gb.graph)
