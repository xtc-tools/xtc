import pytest

from iree_utils import requires_iree, requires_iree_runtime, matmul_impl

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


@requires_iree_runtime
def test_evaluate_validates_and_times():
    # evaluate(validate=True) returns the best per-call time only when the
    # output matches the numpy reference; a mismatch would return a string.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, _tiled_vectorized))
    assert isinstance(result, float)
    assert result > 0


@requires_iree_runtime
def test_close_unloads_and_allows_reload():
    # close() releases the dlopen handle and drops the singleton; the next
    # IREERuntime() reloads the shim on demand.
    from xtc.runtimes.iree.IREERuntime import IREERuntime

    runtime = IREERuntime()
    assert runtime._library() is not None
    runtime.close()
    assert runtime._lib is None
    assert IREERuntime._instance is None
    # A fresh runtime reloads cleanly and works.
    reloaded = IREERuntime()
    assert reloaded._library() is not None


@requires_iree_runtime
def test_evaluate_nop_schedule():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, lambda sch: None))
    assert isinstance(result, float) and result > 0


@requires_iree_runtime
def test_evaluate_with_parameters_writeback(tmp_path):
    # When explicit (inputs, outputs) arrays are passed, the evaluator runs the
    # kernel on them and writes the results back into the output arrays in place.
    import numpy as np

    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    module = impl.get_compiler(dump_file=str(tmp_path / "m")).compile(
        _schedule(impl, _tiled_vectorized)
    )
    a = np.random.rand(I, K).astype("float32")
    b = np.random.rand(K, J).astype("float32")
    out = np.zeros((I, J), dtype="float32")
    with np.errstate(all="ignore"):
        expected = a @ b

    evaluator = module.get_evaluator(
        parameters=([a, b], [out]), repeat=1, number=1, min_repeat_ms=0
    )
    _, code, _ = evaluator.evaluate()
    assert code == 0
    assert np.allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_init_zero_builds_zero_inputs():
    # init_zero must fill every input with zeros; the default filler is non-zero.
    import numpy as np
    from types import SimpleNamespace
    from xtc.targets.iree.IREEEvaluator import IREEEvaluator

    def spec():
        return [{"shape": (4, 4), "dtype": "float32"}]

    stub = SimpleNamespace(
        _np_inputs_spec=spec, _np_outputs_spec=spec, _reference_impl=None
    )
    zeroed = IREEEvaluator(stub, init_zero=True)._make_inputs()
    assert zeroed and all(np.all(x == 0) for x in zeroed)
    # Contrast: the default filler produces at least one non-zero input.
    filled = IREEEvaluator(stub, init_zero=False)._make_inputs()
    assert not all(np.all(x == 0) for x in filled)


@requires_iree_runtime
def test_evaluate_parallelized():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")

    def sched(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    schedule = _schedule(impl, sched)
    assert schedule.parallelized is True
    result = impl.evaluate(schedule)
    assert isinstance(result, float) and result > 0


@requires_iree_runtime
def test_executor_execute_succeeds(tmp_path):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = _schedule(impl, _tiled_vectorized)
    dump = tmp_path / "matmul_iree"
    module = impl.get_compiler(dump_file=str(dump)).compile(schedule)
    executor = module.get_executor(validate=True)
    assert executor.execute() == 0


def test_pmu_counters_require_single_thread():
    # PMU restriction is a pure policy check in __init__ (per-task perf_event
    # counters miss the local-task pool), so it needs neither compiler nor shim.
    from types import SimpleNamespace
    from xtc.targets.iree.IREEEvaluator import IREEEvaluator

    stub = SimpleNamespace(
        _np_inputs_spec=None, _np_outputs_spec=None, _reference_impl=None
    )
    with pytest.raises(NotImplementedError):
        IREEEvaluator(stub, pmu_counters=["INSTRUCTIONS"], single_thread=False)
    # Single-threaded (local-sync) is allowed.
    IREEEvaluator(stub, pmu_counters=["INSTRUCTIONS"], single_thread=True)


@requires_iree
def test_evaluator_default_thread_policy(tmp_path):
    # A non-parallelized schedule defaults to single-threaded (local-sync),
    # a parallelized one to multi-threaded (local-task). This only inspects the
    # evaluator's thread policy, so it needs the compiler but not the shim.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    dump = tmp_path / "matmul_iree"

    seq = impl.get_compiler(dump_file=str(dump)).compile(
        _schedule(impl, _tiled_vectorized)
    )
    assert seq.get_evaluator()._single_thread is True

    impl2 = matmul_impl(*MATMUL_ARGS, "matmul")

    def par(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    dump2 = tmp_path / "matmul_iree_par"
    mod = impl2.get_compiler(dump_file=str(dump2)).compile(_schedule(impl2, par))
    assert mod.get_evaluator()._single_thread is False
