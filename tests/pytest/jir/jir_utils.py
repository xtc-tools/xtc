
def requires_jir(*arg):
    import pytest
    def has_jir():
        try:
            import jir
            return True
        except:
            return False
    return pytest.mark.skipif(not has_jir(), reason="requires JIR")(*arg)


def matmul_impl(i, j, k, dtype, name):
    import xtc.graphs.xtc.op as O
    from xtc.backends.jir import JIRBackend

    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")

    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")

    return JIRBackend(gb.graph)
