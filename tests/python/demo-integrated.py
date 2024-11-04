import os, sys
from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import f32

from MlirNodeImplementer import MlirNodeImplementer
from MlirGraphImplementer import MlirGraphImplementer
from xdsl_aux import parse_xdsl_module

home = os.environ.get("HOME", "")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 16

source = """
func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%C : memref<512x128xf32>)
  linalg.matmul ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>) outs(%C : memref<512x128xf32>)
  return
}
"""

module = parse_xdsl_module(source)

xdsl_func_list = []
xdsl_fill_list = []
xdsl_matmul_list = []
for o in module.walk():
    if isinstance(o, func.FuncOp):
        xdsl_func_list.append(o)
    elif isinstance(o, linalg.FillOp):
        xdsl_fill_list.append(o)
    elif isinstance(o, linalg.MatmulOp):
        xdsl_matmul_list.append(o)
assert len(xdsl_func_list) == 1
assert len(xdsl_fill_list) == 1
assert len(xdsl_matmul_list) == 1
xdsl_func = xdsl_func_list[0]
xdsl_fill = xdsl_fill_list[0]
xdsl_matmul = xdsl_matmul_list[0]

fill_impl = MlirNodeImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op=xdsl_fill,
    dims={"i": i, "j": j},
    parallel_dims=["i", "j"],
    reduction_dims=[],
    vectors_size=vectors_size,
    payload_name="fill",
)

matmul_impl = MlirNodeImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op=xdsl_matmul,
    dims={"i": i, "j": j, "k": k},
    parallel_dims=["i", "j"],
    reduction_dims=["k"],
    vectors_size=vectors_size,
    payload_name="matmul",
)

impl_graph = MlirGraphImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    vectors_size=vectors_size,
    xdsl_func=xdsl_func,
    nodes=[fill_impl, matmul_impl],
)

impl_graph.nodes["fill"].tile("i", {"i1": 4})
impl_graph.nodes["fill"].tile("j", {"j1": 64})
impl_graph.nodes["fill"].interchange(["i", "j", "i1", "j1"])
impl_graph.nodes["fill"].vectorize(["j1"])
impl_graph.nodes["fill"].parallelize(["i"])
impl_graph.nodes["fill"].unroll({"i1": 4})

impl_graph.nodes["matmul"].tile("i", {"i1": 4})
impl_graph.nodes["matmul"].tile("j", {"j1": 64})
impl_graph.nodes["matmul"].tile("k", {"k1": 8})
impl_graph.nodes["matmul"].interchange(["i", "j", "k", "k1", "i1", "j1"])
impl_graph.nodes["matmul"].vectorize(["j1"])
impl_graph.nodes["matmul"].parallelize(["i"])
impl_graph.nodes["matmul"].unroll({"i1": 4, "k1": 8})

e = impl_graph.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_lowered_ir=False,
    print_assembly=True,
    color=True,
    debug=False,
)
