from iree_utils import matmul_impl


def test_backend_construction_reuses_mlir_emission():
    impl = matmul_impl(64, 64, 64, "float32", "matmul")

    # The backend exposes the source graph unchanged.
    assert impl.graph is impl._graph

    # Emission is delegated to the MLIR backend, in tensor form, and each
    # scheduled op keeps its __xtc_id_<node>_ marker (anchor for the schedule).
    mlir_text = str(impl.mlir_backend.xdsl_func)
    assert "linalg.matmul" in mlir_text
    assert "__xtc_id_C_" in mlir_text


def test_backend_nodes_info():
    impl = matmul_impl(64, 64, 64, "float32", "matmul")

    assert set(impl.nodes_info) == {"C"}
    info = impl.nodes_info["C"]
    assert info["op_id"] == "__xtc_id_C_"
    assert info["dims"] == ["i", "j", "k"]
    # i, j are parallel; k is the reduction axis.
    assert info["kinds"] == ["P", "P", "R"]


def test_scheduler_and_compiler_available():
    impl = matmul_impl(64, 64, 64, "float32", "matmul")
    from xtc.backends.iree import IREEScheduler, IREECompiler

    assert isinstance(impl.get_scheduler(), IREEScheduler)
    assert isinstance(impl.get_compiler(), IREECompiler)
