from iree_utils import matmul_impl

I, J, K, DTYPE = 64, 64, 64, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


def _generate(sched_func):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sched_func(sch)
    return impl.get_compiler().generate_mlir(sch.schedule())


def test_generate_mlir_value_semantics():
    # The output memref argument is dropped and the result tensor is returned.
    text = _generate(lambda sch: None)
    assert (
        "func.func @matmul(%0 : tensor<64x64xf32>, %1 : tensor<64x64xf32>) "
        "-> tensor<64x64xf32>" in text
    )
    assert "func.return %" in text
    assert "memref" not in text


def test_generate_mlir_injects_compilation_info():
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.vectorize(["j1"])

    text = _generate(sched)
    # compilation_info is spliced next to the matmul's marker, not the fill's.
    assert "linalg.matmul {__xtc_id_C_, compilation_info = " in text
    assert "#iree_codegen.compilation_info<" in text
    assert "distribution" in text
    assert "CPUDoubleTilingExpert" in text
    # The fill keeps its bare marker (it is not scheduled).
    assert "linalg.fill {__xtc_id_C_0_}" in text


def test_generate_mlir_without_schedule_keeps_bare_markers():
    # A nop schedule injects nothing; IREE picks its own codegen.
    text = _generate(lambda sch: None)
    assert "compilation_info" not in text
    assert "__xtc_id_C_" in text


def test_generate_mlir_does_not_mutate_backend():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    before = str(impl.mlir_backend.xdsl_func)
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 16})
    sch.tile("j", {"j1": 16})
    impl.get_compiler().generate_mlir(sch.schedule())
    # The clone protects the backend function from the value-semantics rewrite.
    assert str(impl.mlir_backend.xdsl_func) == before
