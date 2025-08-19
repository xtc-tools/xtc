
from tvm_utils import requires_tvm, matmul_impl

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)

def sched_nop(sch):
    # Expected in TVM schedule
    return [
        "reorder(i, j, k)"
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in TVM schedule
    return [
        "reorder(i, j, k, i1, j1, k1, i2, j2)",
        "split(j, factor=64)",
        "split(j1, factor=64)",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1", "j", "k", "j1", "k1", "i2", "j2"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "k1": 13, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in TVM schedule
    return [
        "reorder(i, i1, j, k, j1, k1, i2, j2)",
        "unroll(j2)",
        "vectorize(j2)",
        "fuse(i, i1)",
        "parallel(i1)",
    ]

def sched_tile3wc(sch):
    sch.tile("i", {"i1": 64, "i2": 32, "i3": 4})
    sch.tile("j", {"j1": 256, "j2": 64, "j3": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "j", "i1", "j1", "k", "i2", "j2", "k1", "i3", "j3"])
    sch.parallelize(["i", "j"])
    sch.unroll({"k1": 13, "i3": 4, "j3": 64})
    sch.vectorize(["j3"])
    sch.buffer_at("j", "write")
    sch.buffer_at("j1", "write")
    # Expected in TVM schedule
    return [
        "sch[O].reorder(i, j, i_, j_)",
        "sch[O_W0].compute_at(sch[O], j)",
        "sch[O_W0].reorder(i1, j1, i_, j_)",
        "sch[O_W1].compute_at(sch[O_W0], j1)",
        "sch[O_W1].reorder(k, i2, j2, k1, i3, j3)",
    ]

def check_self_sched(impl, sch):
    schedule_str = sch.get_schedule_str()
    print(f"Schedule str: {schedule_str}")
    sch2 = impl.get_scheduler()
    exec(schedule_str, {"sch": sch2}, {})
    schedule_str2 = sch2.get_schedule_str()
    assert schedule_str == schedule_str2

def check_schedule(impl, sched_func):
    sch = impl.get_scheduler()
    expected = sched_func(sch)
    check_self_sched(impl, sch)
    schedule_str = sch.get_schedule_str()
    print(f"Schedule:\n{schedule_str}")
    schedule = sch.schedule()
    schedule_str = str(schedule)
    print(f"TVM schedule:\n{schedule_str}")
    for substr in expected:
        assert substr in schedule_str
    return schedule

def check_evaluate(impl, schedule):
    result = impl.evaluate(schedule)
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

@requires_tvm
def test_sched_nop():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile2():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile2p():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2p)
    check_evaluate(impl, schedule)

@requires_tvm
def test_sched_tile3wc():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile3wc)
    check_evaluate(impl, schedule)
