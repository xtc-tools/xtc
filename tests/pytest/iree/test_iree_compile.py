from pathlib import Path

from iree_utils import requires_iree, matmul_impl

I, J, K, DTYPE = 64, 64, 64, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


def _compile(tmp_path, sched_func):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sched_func(sch)
    dump = tmp_path / "matmul_iree"
    return impl.get_compiler(dump_file=str(dump)).compile(sch.schedule())


@requires_iree
def test_compile_produces_vmfb(tmp_path):
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.vectorize(["j1"])

    module = _compile(tmp_path, sched)

    assert module.file_type == "vmfb"
    assert module.name == "matmul"
    assert module.payload_name == "matmul"
    vmfb = Path(module.file_name)
    assert vmfb.suffix == ".vmfb"
    assert vmfb.exists() and vmfb.stat().st_size > 0


@requires_iree
def test_compile_nop_schedule(tmp_path):
    # No lowering config: IREE still compiles the bare linalg to a vmfb.
    module = _compile(tmp_path, lambda sch: None)
    assert Path(module.file_name).stat().st_size > 0
