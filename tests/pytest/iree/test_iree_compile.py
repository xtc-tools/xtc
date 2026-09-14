import os
import tempfile
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
def test_compile_without_dump_file_uses_temp_vmfb():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 16})
    module = impl.get_compiler().compile(sch.schedule())
    try:
        directory, base = os.path.split(os.path.realpath(module.file_name))
        assert directory == os.path.realpath(tempfile.gettempdir())
        assert base.startswith("matmul_") and base.endswith(".vmfb")
        assert os.path.getsize(module.file_name) > 0
    finally:
        os.unlink(module.file_name)


@requires_iree
def test_print_source_ir_dumps_annotated_mlir(tmp_path, capsys):
    # print_source_ir echoes the annotated MLIR handed to iree-compile onto
    # stderr; without it nothing is printed (so the flag actually controls it).
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.vectorize(["j1"])

    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = _schedule_of(impl, sched)

    impl.get_compiler(dump_file=str(tmp_path / "off")).compile(schedule)
    assert "IREE input MLIR" not in capsys.readouterr().err

    impl.get_compiler(
        dump_file=str(tmp_path / "on"), print_source_ir=True
    ).compile(schedule)
    err = capsys.readouterr().err
    assert "IREE input MLIR" in err
    assert "func.func @matmul" in err
    assert "compilation_info" in err


def _schedule_of(impl, sched_func):
    sch = impl.get_scheduler()
    sched_func(sch)
    return sch.schedule()


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
