from iree_utils import requires_iree, matmul_impl

I, J, K, DTYPE = 64, 64, 64, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


def _schedule(impl, sched_func):
    sch = impl.get_scheduler()
    sched_func(sch)
    return sch.schedule()


def _tiled_vectorized(sch):
    sch.tile("i", {"i1": 16})
    sch.tile("j", {"j1": 16})
    sch.tile("k", {"k1": 16})
    sch.vectorize(["j1"])


@requires_iree
def test_evaluate_validates_and_times():
    # evaluate(validate=True) returns the best per-call time only when the
    # output matches the numpy reference; a mismatch would return a string.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, _tiled_vectorized))
    assert isinstance(result, float)
    assert result > 0


@requires_iree
def test_evaluate_nop_schedule():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, lambda sch: None))
    assert isinstance(result, float) and result > 0


@requires_iree
def test_evaluate_parallelized():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")

    def sched(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    schedule = _schedule(impl, sched)
    assert schedule.parallelized is True
    result = impl.evaluate(schedule)
    assert isinstance(result, float) and result > 0


@requires_iree
def test_executor_execute_succeeds(tmp_path):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = _schedule(impl, _tiled_vectorized)
    dump = tmp_path / "matmul_iree"
    module = impl.get_compiler(dump_file=str(dump)).compile(schedule)
    executor = module.get_executor(validate=True)
    assert executor.execute() == 0


@requires_iree
def test_evaluator_default_thread_policy(tmp_path):
    # A non-parallelized schedule defaults to single-threaded (local-sync),
    # a parallelized one to multi-threaded (local-task).
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    dump = tmp_path / "matmul_iree"

    seq = impl.get_compiler(dump_file=str(dump)).compile(
        _schedule(impl, _tiled_vectorized)
    )
    assert seq.get_evaluator()._driver == "local-sync"

    impl2 = matmul_impl(*MATMUL_ARGS, "matmul")

    def par(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    dump2 = tmp_path / "matmul_iree_par"
    mod = impl2.get_compiler(dump_file=str(dump2)).compile(_schedule(impl2, par))
    assert mod.get_evaluator()._driver == "local-task"
