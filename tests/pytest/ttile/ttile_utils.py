
def requires_tvm(*arg):
    import pytest
    def has_tvm():
        try:
            import tvm
            return True
        except:
            return False
    return pytest.mark.skipif(not has_tvm(), reason="requires TVM")(*arg)

def requires_pmu(*arg):
    import pytest
    from sys import platform
    def has_pmu():
        # On macos, pmu counter requires root privileges, skip for tests
        return platform != "darwin"
    return pytest.mark.skipif(not has_pmu(), reason="requires privileges for PMU counters")(*arg)
