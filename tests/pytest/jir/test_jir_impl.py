
from jir_utils import requires_jir, matmul_impl

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)

def sched_nop(sch):
    # Expected in JIR schedule
    return [
        "commands: []",
        "I: 128",
        "J: 256",
        "K: 91",
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in JIR schedule
    return [
        "tile target=JJ tile=J_1 inner=JJ_j1",
        "tile target=JJ_j1 tile=J_2 inner=JJ_j2",
        "J_1: 64",
        "J_2: 64",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1", "j", "k", "j1", "k1", "i2", "j2"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "k1": 13, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in JIR schedule
    return [
        "update_props target=JJ_j2 vector=64",
        "update_props target=II parallel",
        "update_props target=II_i1 parallel",
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
    # Expected in JIR schedule
    # TODO: buffer not implemented yet
    return [
        "update_props target=JJ_j3 vector=64",
        "update_props target=II_i3 unroll=4",
        "update_props target=KK_k1 unroll=13",
        "update_props target=II parallel",
        "update_props target=JJ parallel",
    ]

def check_schedule(impl, sched_func):
    sch = impl.get_scheduler()
    expected = sched_func(sch)
    schedule = sch.schedule()
    schedule_str = str(schedule)
    print(f"JIR schedule:\n{schedule_str}")
    for substr in expected:
        assert substr in schedule_str
    return schedule

def check_evaluate(impl, schedule):
    result = impl.evaluate(schedule)
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

@requires_jir
def test_sched_nop():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_jir
def test_sched_tile2():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2)
    check_evaluate(impl, schedule)

@requires_jir
def test_sched_tile2p():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile2p)
    check_evaluate(impl, schedule)

@requires_jir
def test_sched_tile3wc():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile3wc)
    check_evaluate(impl, schedule)
