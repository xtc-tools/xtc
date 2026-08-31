from xtc.backends.iree.IREEConfig import IREEConfig


def test_compile_args_default_target_only():
    args = IREEConfig().iree_compile_args()
    assert args == ["--iree-llvmcpu-target-cpu=host"]


def test_compile_args_include_triple_and_features():
    cfg = IREEConfig(
        target_cpu="sentinel-cpu",
        target_triple="sentinel-triple",
        target_cpu_features="+sentinel-feature",
        extra_args=["--iree-opt-level=O3"],
    )
    args = cfg.iree_compile_args()
    assert "--iree-llvmcpu-target-cpu=sentinel-cpu" in args
    assert "--iree-llvmcpu-target-triple=sentinel-triple" in args
    assert "--iree-llvmcpu-target-cpu-features=+sentinel-feature" in args
    assert "--iree-opt-level=O3" in args


def test_print_transformed_ir_adds_dump_flags():
    # print_transformed_ir asks IREE to dump its IR after vectorization; off by
    # default so it never perturbs a normal compile.
    assert not any(
        "print-ir-after" in a for a in IREEConfig().iree_compile_args()
    )
    args = IREEConfig(print_transformed_ir=True).iree_compile_args()
    assert "--mlir-print-ir-after=iree-codegen-generic-vectorization" in args
    assert "--mlir-disable-threading" in args
    assert "--mlir-print-debuginfo=false" in args
