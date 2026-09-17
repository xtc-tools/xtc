import pytest
from pathlib import Path
from tvm_utils import requires_tvm, matmul_impl

I, J, K, DTYPE = 128, 256, 91, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


@requires_tvm
@pytest.mark.parametrize(
    ("shape", "dims"),
    (
        ((32,), ("i",)),
        ((4, 32), ("i", "j")),
        ((1, 4, 4, 16), ("i", "j", "k", "l")),
    ),
)
def test_relu_inherits_input_rank(shape, dims):
    import xtc.graphs.xtc.op as O
    from xtc.backends.tvm import TVMBackend

    inp = O.tensor(shape, "float32", name="A")
    with O.graph(name="relu_graph") as gb:
        O.relu(inp, name="relu")

    impl = TVMBackend(gb.graph)
    relu = impl._ops["relu"]
    assert relu.operator.dims() == dims
    assert relu.operator.dims("P") == dims
    assert relu.operator.dims("R") == ()
    assert relu.operator.dims_sizes() == dict(zip(dims, shape))
    assert relu.operator.inputs_dims() == (shape,)
    assert relu.operator.outputs_dims() == (shape,)


def sched_nop(sch):
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, j, k)"
    ]

def sched_tile2(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 64})
    sch.tile("k", {"k1": 13})
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, i1, i2, j, j1, j2, k, k1)",
        "split(i, factors=[None, 16, 4])",
        "split(j, factors=[None, 1, 64])",
    ]

def sched_tile2p(sch):
    sch.tile("i", {"i1": 64, "i2": 4})
    sch.tile("j", {"j1": 64, "j2": 16})
    sch.tile("k", {"k1": 13})
    sch.interchange(["i", "i1", "j", "k", "j1", "k1", "i2", "j2"])
    sch.parallelize(["i", "i1"])
    sch.unroll({"j2": 64, "i2": 4})
    sch.vectorize(["j2"])
    # Expected in TVM schedule
    print(sch)
    return [
        "reorder(i, i1, j, k, j1, k1, i2, j2)",
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
    sch.buffer_at("j")
    sch.buffer_at("j1")
    print(sch)
    # Expected in TVM schedule
    return [
        "reorder(i, j, i1, j1, k, i2, j2, k1, i3, j3)",
        "reverse_compute_at(O_W0, j)",
        "reverse_compute_at(O_W0, j1)",
    ]

def sched_tile_unroll_vec(sch):
    sch.tile("i", {"i1": 8})
    sch.tile("j", {"j1": 48})
    sch.tile("k", {"k1": 50})
    sch.interchange(["j", "k", "i", "k1", "i1", "j1"])
    sch.parallelize(["j"])
    sch.unroll({"k1": 32, "i1": 8})
    sch.vectorize(["j1"])
    print(sch)
    # Expected in TVM schedule
    return [
        "reorder(j, k, i, k1, __u_k1, i1, j1, __v_j1)",
        "unroll(__u_k1)",
        "unroll(i1)",
        "unroll(j1)",
        "vectorize(__v_j1)",
    ]

def check_schedule(impl, sched_func):
    sch = impl.get_scheduler()
    expected = sched_func(sch)
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

@requires_tvm
def test_sched_tile_unroll_vec():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    schedule = check_schedule(impl, sched_tile_unroll_vec)
    check_evaluate(impl, schedule)

def check_compile_evaluate(imp, schedule, compiler_args, evaluate_args):
    compiler = imp.get_compiler(
        **compiler_args,
    )
    module = compiler.compile(schedule)
    evaluator = module.get_evaluator(
        **evaluate_args,
    )
    results, code, error_msg = evaluator.evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"
    print(f"Results: {results}")
    assert isinstance(results[0], float) and float(results[0]) > 0

@requires_tvm
@pytest.mark.parametrize(
    "module_type",
    (
        "shared_lib",
        "emit_c",
        "ar_lib",
    ),
)
@pytest.mark.parametrize(
    "bare_ptr",
    (
        False,
        True,
    ),
)
def test_backend_variant(tmpdir, module_type, bare_ptr):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    print(impl.graph)
    libpath = Path(tmpdir) / impl.graph.name
    schedule = check_schedule(impl, sched_tile2p)
    check_compile_evaluate(
        impl,
        schedule,
        {
            "dump_file": str(libpath),
            module_type: True,
            "bare_ptr": bare_ptr,
        },
        {
            "validate": True,
        }
    )
