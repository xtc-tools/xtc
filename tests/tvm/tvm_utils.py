
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
    from xtc.backends.tvm import TVMBackend
    from xtc.backends.tvm.TVMOps import TVMOperators, TVMOperation
    op = TVMOperation(TVMOperators.matmul, (i, j, k, dtype), name=name)
    impl = TVMBackend(
        source_op = op,
        dims=dict(i=i, j=j, k=k),
        parallel_dims=["i", "j"],
    )
    return impl
