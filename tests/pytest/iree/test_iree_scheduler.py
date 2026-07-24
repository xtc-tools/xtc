import pytest

from iree_utils import matmul_impl

I, J, K, DTYPE = 64, 64, 64, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


def _schedule_str(sched_func):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sched_func(sch)
    return str(sch.schedule())


def test_no_tiling_yields_no_config():
    # An empty schedule (no tiling at all) produces no lowering config, so IREE
    # is left to pick its own codegen.
    assert _schedule_str(lambda sch: None) == "{}"


def test_tile_and_vectorize():
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.vectorize(["j1"])

    text = _schedule_str(sched)
    assert "iree_cpu.lowering_config" in text
    assert "distribution = [0, 0, 0]" in text
    assert "cache_parallel = [16, 0, 0]" in text
    assert "cache_reduction = [0, 0, 16]" in text
    assert "vector_common_parallel = [0, 16, 0]" in text
    assert "vector_reduction = [0, 0, 1]" in text
    assert "CPUDoubleTilingExpert" in text


def test_three_level_tiling_with_vectorized_innermost():
    def sched(sch):
        # i: distribution (32) + cache_parallel (8) + vector (4, from vectorize).
        sch.tile("i", {"i1": 32, "i2": 8, "i3": 4})
        sch.tile("k", {"k1": 16})
        sch.parallelize(["i"])
        sch.vectorize(["i3"])

    text = _schedule_str(sched)
    assert "distribution = [32, 0, 0]" in text
    assert "cache_parallel = [8, 0, 0]" in text
    assert "vector_common_parallel = [4, 0, 0]" in text
    assert "cache_reduction = [0, 0, 16]" in text
    assert "CPUDoubleTilingExpert" in text


def test_vectorize_non_innermost_level_raises():
    # Only the innermost tile level of a dim may be vectorized.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 32, "i2": 8})
    sch.vectorize(["i1"])  # outer level, not innermost
    with pytest.raises(NotImplementedError):
        sch.schedule().lowering_configs()


def test_multilevel_tiling():
    def sched(sch):
        sch.tile("i", {"i1": 32, "i2": 8})
        sch.tile("j", {"j1": 32, "j2": 8})
        sch.tile("k", {"k1": 16})
        sch.parallelize(["i", "j"])

    text = _schedule_str(sched)
    # Two-level tiling on parallelized dims -> distribution (outer) +
    # cache_parallel (inner). The outer level is the only one distributable.
    assert "distribution = [32, 32, 0]" in text
    assert "cache_parallel = [8, 8, 0]" in text
    assert "cache_reduction = [0, 0, 16]" in text
    # No vectorize -> default pipeline, no explicit vector levels.
    assert "CPUDefault" in text
    assert "vector_common_parallel" not in text


def test_multilevel_without_parallelize_raises():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 32, "i2": 8})
    with pytest.raises(NotImplementedError):
        sch.schedule().lowering_configs()


def test_selective_parallelize():
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.parallelize(["i"])

    text = _schedule_str(sched)
    # Only i is distributed (threaded); j stays a sequential cache_parallel loop.
    assert "distribution = [16, 0, 0]" in text
    assert "cache_parallel = [0, 16, 0]" in text
    assert "cache_reduction = [0, 0, 16]" in text


def test_unsupported_primitives_raise():
    # Primitives IREE would silently override must fail loudly, not warn.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    with pytest.raises(NotImplementedError):
        sch.interchange(["k", "i", "j"])
    with pytest.raises(NotImplementedError):
        sch.unroll({"i": 4})
    with pytest.raises(NotImplementedError):
        sch.split("i", {"i1": 32})
    with pytest.raises(NotImplementedError):
        sch.buffer_at("i")
    with pytest.raises(NotImplementedError):
        sch.pack_at("i", 0)
    with pytest.raises(NotImplementedError):
        sch.fuse_producer_at("i", 0)


def test_vectorize_reduction_axis():
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16, "j2": 8})
        sch.tile("k", {"k1": 4})
        sch.parallelize(["i", "j"])
        sch.vectorize(["j2", "k1"])  # inner parallel tile j2 and the reduction k1

    text = _schedule_str(sched)
    # The reduction tile (k) is vectorized into vector_reduction (-> vector.contract
    # in IREE), so it is consumed there and dropped from cache_reduction; j2 feeds
    # vector_common_parallel as its register tile.
    assert "distribution = [16, 16, 0]" in text
    assert "vector_common_parallel = [0, 8, 0]" in text
    assert "vector_reduction = [0, 0, 4]" in text
    assert "cache_reduction" not in text
    assert "CPUDoubleTilingExpert" in text


def test_vectorize_only_parallel_keeps_reduction_scalar():
    def sched(sch):
        sch.tile("i", {"i1": 16})
        sch.tile("j", {"j1": 16})
        sch.tile("k", {"k1": 16})
        sch.parallelize(["i"])
        sch.vectorize(["j1"])  # reduction k not vectorized -> stays scalar (1)

    text = _schedule_str(sched)
    assert "vector_common_parallel = [0, 16, 0]" in text
    assert "vector_reduction = [0, 0, 1]" in text
    assert "cache_reduction = [0, 0, 16]" in text


def test_reduction_only_tiling_single_workgroup():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sch.tile("k", {"k1": 16})
    text = str(sch.schedule())
    assert "distribution" not in text
    assert "cache_reduction = [0, 0, 16]" in text


def test_deeper_reduction_tile_levels_raise():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 16})
    sch.tile("k", {"k1": 16, "k2": 4})
    with pytest.raises(NotImplementedError):
        sch.schedule().lowering_configs()


