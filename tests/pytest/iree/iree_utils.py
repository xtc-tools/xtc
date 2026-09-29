def requires_iree(*arg):
    """Skip the decorated test when the IREE compiler is unavailable."""
    import pytest

    def has_iree():
        try:
            import iree.compiler  # noqa: F401

            return True
        except Exception:
            return False

    return pytest.mark.skipif(not has_iree(), reason="requires IREE")(*arg)


def requires_iree_runtime(*arg):
    """Skip when the IREE compiler or the native runtime shim is unavailable.

    The C measurement path needs both the compiler (to build the ``.vmfb``) and
    the ``xtc_iree_shim`` library built by scripts/iree/build_runtime.sh.
    """
    import pytest

    def has_runtime():
        try:
            import iree.compiler  # noqa: F401
            from xtc.utils.tools import has_iree_runtime

            return has_iree_runtime()
        except Exception:
            return False

    return pytest.mark.skipif(
        not has_runtime(), reason="requires IREE runtime shim"
    )(*arg)


def matmul_impl(i, j, k, dtype, name, **kwargs):
    """Build an IREE backend for a single ``i x j x k`` matmul graph."""
    import xtc.graphs.xtc.op as O
    from xtc.backends.iree import Backend

    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")

    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")

    return Backend(gb.graph, **kwargs)
