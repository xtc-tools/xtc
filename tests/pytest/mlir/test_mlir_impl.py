
import pytest

from mlir_utils import requires_mlir, matmul_impl

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)

def sched_nop(sch):
    # Expected in MLIR schedule
    return [
        "permutation={'.': ['./i', './j', './k']}"
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in MLIR schedule
    return [
        "permutation={'.': ['./i', './i1', './i2', './j', './j1', './j2', './k', './k1']}",
        "'./j': {'./j1': 64, './j2': 64}",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1", "j", "k", "j1", "k1", "i2", "j2"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "k1": 13, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in MLIR schedule
    return [
        "permutation={'.': ['./i', './i1', './j', './k', './j1', './k1', './i2', './j2']}",
        "unrolling={'./j2': 64, './k1': 13, './i2': 4}",
        "vectorization=['./j2']",
        "parallelization=['./i', './i1']",
    ]

def sched_tile3wc(sch):
    sch.tile("i", {"i1": 64, "i2": 32, "i3": 4})
    sch.tile("j", {"j1": 256, "j2": 64, "j3": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "j", "i1", "j1", "k", "i2", "j2", "k1", "i3", "j3"])
    sch.parallelize(["i", "j"])
    sch.unroll({"k1": 13, "i3": 4, "j3": 64})
    sch.vectorize(["j3"])
    sch.buffer_at("j")
    sch.buffer_at("j1")
    # Expected in MLIR schedule
    return [
        "permutation={'.': ['./i', './j', './i1', './j1', './k', './i2', './j2', './k1', './i3', './j3']",
        "vectorization=['./j3']",
        "parallelization=['./i', './j']",
        "unrolling={'./k1': 13, './i3': 4, './j3': 64}",
    ]

def sched_nested_split(sch):
    sch.split("i", {"i_top": 0, "i_bot": 6})
    sch.interchange(["i_top", "i_bot"])
    i_tile = {"./i_top": 6, "./i_bot": 5}
    for iroot in ["./i_top", "./i_bot"]:
        sch.split("j", {"j_top": 0, "j_bot": 32}, root=iroot)
        sch.interchange(["i", "j_top", "j_bot"], root=iroot)
        for jname, j_reg in (("j_top", 32), ("j_bot", 8)):
            leaf = f"{iroot}/{jname}"
            sch.tile("i", {"i_col": i_tile[iroot]}, root=leaf)
            sch.tile("j", {"j_reg": j_reg, "j_lane": 8}, root=leaf)
            sch.interchange(["i", "j", "k", "i_col", "j_reg", "j_lane"], root=leaf)
            sch.vectorize(["j_lane"], root=leaf)
            sch.unroll({"i_col": i_tile[iroot], "j_reg": j_reg // 8}, root=leaf)
    # Expected in MLIR schedule
    return [
        "'./i': {'./i_top': 0, './i_bot': 6}",
        "'./i_top/j': {'./i_top/j_top': 0, './i_top/j_bot': 32}",
        "'./i_top/j_top/i': {'./i_top/j_top/i_col': 6}",
        "'./i_bot/j_top/i': {'./i_bot/j_top/i_col': 5}",
        "vectorization=['./i_top/j_top/j_lane', './i_top/j_bot/j_lane', "
        "'./i_bot/j_top/j_lane', './i_bot/j_bot/j_lane']",
    ]

def check_schedule(impl, sched_func):
    sch = impl.get_scheduler()
    expected = sched_func(sch)
    schedule = sch.schedule()
    schedule_str = str(schedule)
    print(f"MLIR schedule:\n{schedule_str}")
    for substr in expected:
        assert substr in schedule_str
    return schedule

def check_op_evaluate(impl, schedule, init_zero=False):
    result = impl.evaluate(
        schedule,
        evaluate_args=dict(
            init_zero=init_zero,
        ),
    )
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

def check_evaluate(impl, schedule):
    result = impl.evaluate(schedule)
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

@requires_mlir
def test_sched_nop():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_mlir
@pytest.mark.parametrize("dtype", ("float32", "float64", "float16"))
def test_sched_nop_dtype(dtype):
    impl = matmul_impl(64, 64, 64, dtype, "matmul_dtype")
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile2():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile2p():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2p)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile3wc():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile3wc)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_nested_split():
    # i = 16 (split 6 + 10), j = 40 (split 32 + 8), k = 8: the two i-tiers get
    # different i-tiles (6 and 5), so a nested-split tiling-state leak would tile
    # i_top's extent-6 leaf by 5 and fail to vectorize. Evaluating checks the
    # schedule actually compiles and validates.
    impl = matmul_impl(16, 40, 8, DTYPE, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_nested_split)
    check_evaluate(impl, schedule)
