import os
from MMImplementer import MMImplementer

home = os.environ.get("HOME", "")

i = 512
j = 128
k = 1024

impl = MMImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    dims={"i": i, "j": j, "k": k},
    parallel_dims=["i", "j"],
    reduction_dims=["k"],
)

impl.tile("i", {"i1": 8})
impl.tile("j", {"j1": 8})
impl.tile("k", {"k1": 8})
impl.interchange(["i", "j", "k", "i1", "j1", "k1"])
impl.vectorize(["k1"])
impl.parallelize(["i"])
impl.unroll({"j1": 8, "i1": 8})

e = impl.evaluate(
    # print_source_ir=True,
    # print_transformed_ir=True,
    # print_ir_after=['convert-vector-to-llvm',],
    # print_ir_before=['test-transform-dialect-erase-schedule'],
    print_assembly=True,
    # color = True,
)
# print(e)
