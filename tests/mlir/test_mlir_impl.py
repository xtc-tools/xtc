
from mlir_utils import requires_mlir, matmul_impl, matmul_node_impl, matmul_ref, get_matmul_params

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)

def sched_nop(sch):
    # Expected in MLIR schedule
    return [
        "permutation=['i', 'j', 'k']"
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in MLIR schedule
    return [
        "permutation=['i', 'j', 'k', 'i1', 'j1', 'k1', 'i2', 'j2']",
        "'j': {'j': 64, 'j1': 64, 'j2': 1}",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "k1": 13, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in MLIR schedule
    return [
        "permutation=['i', 'i1']",
        "unrolling={'j2': 64, 'k1': 13, 'i2': 4}",
        "vectorization=['j2']",
        "parallelization=['i', 'i1']",
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
    # Expected in MLIR schedule
    return [
        "permutation=['i', 'j', 'i1', 'j1', 'k', 'i2', 'j2', 'k1', 'i3', 'j3']",
        "vectorization=['j3']",
        "parallelization=['i', 'j']",
        "unrolling={'k1': 13, 'i3': 4, 'j3': 64}",
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
    # For now when a graph implementer, need to pass
    # reference impl and parameters
    result = impl.evaluate(
        schedule,
        evaluate_args=dict(
            reference_impl=matmul_ref,
            parameters=get_matmul_params(*MATMUL_ARGS),
        ),
    )
    print(f"Result: {result}")
    assert isinstance(result, float) and float(result) > 0

@requires_mlir
def test_sched_nop():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = check_schedule(impl, sched_nop)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile2():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = check_schedule(impl, sched_tile2)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile2p():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = check_schedule(impl, sched_tile2p)
    check_evaluate(impl, schedule)

@requires_mlir
def test_sched_tile3wc():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = check_schedule(impl, sched_tile3wc)
    check_evaluate(impl, schedule)

@requires_mlir
def test_node_matmul():
    impl = matmul_node_impl(*MATMUL_ARGS, "matmul")
    schedule = impl.get_scheduler().schedule()
    check_op_evaluate(impl, schedule, init_zero=True)
