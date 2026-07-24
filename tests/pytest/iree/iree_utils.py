def requires_iree(*arg):
    """Skip the decorated test when the IREE compiler/runtime is unavailable."""
    import pytest

    def has_iree():
        try:
            import iree.compiler
            import iree.runtime

            return True
        except Exception:
            return False

    return pytest.mark.skipif(not has_iree(), reason="requires IREE")(*arg)


def matmul_impl(i, j, k, dtype, name, **kwargs):
    """Build an IREE backend for a single ``i x j x k`` matmul graph."""
    import xtc.graphs.xtc.op as O
    from xtc.backends.iree import Backend

    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")

    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")

    return Backend(gb.graph, **kwargs)
