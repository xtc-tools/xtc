import os,sys

sys.path.append('../')

from MlirNodeImplementer import MlirNodeImplementer
from MlirGraphImplementer import MlirGraphImplementer

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
    FloatAttr,
)
from xdsl.dialects.arith import (
    Constant,
    Mulf,
    Addf,
    FastMathFlagsAttr
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 16

# Definition of the payload
A_ty =MemRefType(elt_type, [i, k])
B_ty = MemRefType(elt_type, [k, j])
C_ty = MemRefType(elt_type, [i, j])
fun_block = Block(arg_types=[A_ty,B_ty,C_ty])
A = fun_block.args[0]
B = fun_block.args[1]
C = fun_block.args[2]
fZero = Constant(FloatAttr(0.0, f32))
linalg_fill = linalg.FillOp(inputs=(fZero.result,), outputs=(C,), res=[])
linalg_matmul = linalg.MatmulOp(
    inputs = (A,B),
    outputs = (C,),
)
fun_block.add_ops([
    fZero,
    linalg_fill,
    linalg_matmul,
    func.Return()
])
fun = func.FuncOp.from_region(
    "myfun",
    [A_ty,B_ty,C_ty],
    [],
    Region(fun_block)
)

fill_impl = MlirNodeImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_fill,
    dims = {'i':i,'j':j},
    parallel_dims = ['i','j'],
    reduction_dims = [],
    vectors_size = vectors_size,
    payload_name = "fill",
)

matmul_impl = MlirNodeImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_matmul,
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
    vectors_size = vectors_size,
    payload_name = "matmul",
)

impl_graph = MlirGraphImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    vectors_size = vectors_size,
    xdsl_func = fun,
    nodes = [fill_impl,matmul_impl],
)

impl_graph.nodes["fill"].tile("i",{'i1':4})
impl_graph.nodes["fill"].tile("j",{'j1':64})
impl_graph.nodes["fill"].interchange(['i','j','i1','j1'])
impl_graph.nodes["fill"].vectorize(['j1'])
impl_graph.nodes["fill"].parallelize(['i'])
impl_graph.nodes["fill"].unroll({'i1':4})

impl_graph.nodes["matmul"].tile("i",{'i1':4})
impl_graph.nodes["matmul"].tile("j",{'j1':64})
impl_graph.nodes["matmul"].tile("k",{'k1':8})
impl_graph.nodes["matmul"].interchange(['i','j','k','k1','i1','j1'])
impl_graph.nodes["matmul"].vectorize(['j1'])
impl_graph.nodes["matmul"].parallelize(['i'])
impl_graph.nodes["matmul"].unroll({'i1':4,'k1':8})

e = impl_graph.evaluate(
    print_source_ir=True,
    print_transformed_ir=False,
    print_lowered_ir = False,
    print_assembly=False,
    color = True,
    debug = False,
)
