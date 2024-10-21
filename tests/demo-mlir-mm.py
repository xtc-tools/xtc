import os,sys

sys.path.append('../')

from MlirImplementer import MlirImplementer

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

block = Block(arg_types=(elt_type,elt_type,elt_type))
mulf = Mulf(
    block.args[0],
    block.args[1],
    FastMathFlagsAttr("fast"),
)
addf = Addf(
    block.args[2],
    mulf.results[0],
    FastMathFlagsAttr("fast"),
)
block.add_ops([
    mulf,
    addf,
    linalg.YieldOp(addf.results[0])
])

linalg_generic = linalg.Generic(
    (
        TestSSAValue(MemRefType(elt_type, [i, k])),
        TestSSAValue(MemRefType(elt_type, [k, j])),
    ),
    (TestSSAValue(MemRefType(elt_type, [i, j])),),
    Region(block),
    (
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(2)))
        ),
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(2),AffineExpr.dimension(1)))
        ),
        AffineMapAttr(
            AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(1)))
        ),
    ),
    (
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.reduction(),
    ),
)

linalg_matmul = linalg.MatmulOp(
    inputs = (
        TestSSAValue(MemRefType(elt_type,(i,k))),
        TestSSAValue(MemRefType(elt_type,(k,j))),
    ),
    outputs = (TestSSAValue(MemRefType(elt_type,(i,j))),),
)

impl = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = linalg_generic,
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
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
