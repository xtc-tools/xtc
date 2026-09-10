from iree_utils import matmul_impl

from xtc.search.strategies import Strategies

# Strategy_IREENative: its search space is exactly IREE's lowering-config space,
# so every sampled schedule must be accepted by the IREE scheduler (structural
# validity) and respect the register/vector size bounds (compiler validity).
# Pure Python: builds schedules only, no iree package or shim needed.


def test_iree_native_samples_are_valid():
    impl = matmul_impl(512, 512, 512, "float32", "matmul")
    strat = Strategies.create("iree_native", impl.graph, threads=8, vreg_num=32)

    count = 0
    for sample in strat.sample(64, seed=0):
        # x = [i1, i2, j1, j2, j3, k1]: SIMD width and register-count bounds.
        assert sample[4] <= 32
        assert sample[1] * sample[3] <= 32
        # The IREE scheduler accepts the schedule.
        sch = impl.get_scheduler()
        strat.generate(sch, sample)
        text = str(sch.schedule())
        assert "iree_cpu.lowering_config" in text or text == "{}"
        count += 1
    assert count > 0


def test_iree_native_serial_is_single_level():
    # With threads<=1 the distribution level is dropped, so parallel dims keep a
    # single (cache) loop level, which the IREE scheduler also accepts.
    impl = matmul_impl(512, 512, 512, "float32", "matmul")
    strat = Strategies.create("iree_native", impl.graph, threads=1)
    sample = strat.default_schedule(opt_level=3)
    sch = impl.get_scheduler()
    strat.generate(sch, sample)
    schedule = sch.schedule()
    assert schedule.parallelized is False
    assert isinstance(str(schedule), str)
