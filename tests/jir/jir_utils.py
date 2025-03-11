
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
    from xtc.backends.jir import JIRImplementer
    from xtc.backends.jir.JIROps import JIROperators, JIROperation
    op = JIROperation(JIROperators.matmul, (i, j, k, dtype), name=name)
    impl = JIRImplementer(
        source_op=op,
        dims=dict(i=i, j=j, k=k),
    )
    return impl
