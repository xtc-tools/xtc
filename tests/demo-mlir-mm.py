import os,sys

from MlirNodeImplementer import MlirNodeImplementer as MlirImplementer

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
)
from xdsl.dialects.arith import (
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

linalg_matmul = linalg.MatmulOp(
    inputs = (
        TestSSAValue(MemRefType(elt_type,(i,k))),
        TestSSAValue(MemRefType(elt_type,(k,j))),
    ),
    outputs = (TestSSAValue(MemRefType(elt_type,(i,j))),),
)

impl = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_matmul,
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
    loop_stamps = ["JoeDassin"],
    vectors_size = vectors_size
)

impl.tile("i",{'i1':4})
impl.tile("j",{'j1':64})
impl.tile("k",{'k1':8})
impl.interchange(['i','j','k','k1','i1','j1'])
impl.vectorize(['j1'])
impl.parallelize(['i'])
impl.unroll({'i1':4,'k1':8})

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_lowered_ir = False,
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
