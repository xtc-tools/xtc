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
from xdsl.dialects.arith import Mulf,Addf
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 16

def get_impl(op):
    impl = MlirImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op = op,
        dims = {'i':i,'j':j,'k':k},
        parallel_dims = ['i','j'],
        reduction_dims = ['k'],
        vectors_size = vectors_size
    )
    return impl

def matmul_op():
    matmul = linalg.MemRefMatmulOp(
        inputs = (
            TestSSAValue(MemRefType(elt_type,(i,k))),
            TestSSAValue(MemRefType(elt_type,(k,j))),
        ),
        outputs = (TestSSAValue(MemRefType(elt_type,(i,j))),),
    )
    return matmul

def matmul_generic_op():
    
    block = Block(arg_types=(elt_type,elt_type,elt_type))
    mulf = Mulf(block.args[0], block.args[1])
    addf = Addf(block.args[2],mulf.results[0])
    block.add_ops([
        mulf,
        addf,
        linalg.YieldOp(addf.results[0])
    ])

    matmul = linalg.Generic(
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

    return matmul

def mm0():
    impl = get_impl(matmul_op())
    return impl

def mm1():
    impl = get_impl(matmul_op())
    impl.tile("i",{'i1':8})
    impl.tile("j",{'j1':8})
    impl.tile("k",{'k1':8})
    impl.interchange(['i','j','k','i1','k1','j1'])
    impl.vectorize(['j1'])
    impl.parallelize(['i'])
    impl.unroll({'k1':8,'i1':8})
    return impl

def generic_mm1():
    impl = get_impl(matmul_generic_op())
    impl.tile("i",{'i1':8})
    impl.tile("j",{'j1':8})
    impl.tile("k",{'k1':8})
    impl.interchange(['i','j','k','i1','k1','j1'])
    impl.vectorize(['j1'])
    impl.parallelize(['i'])
    impl.unroll({'k1':8,'i1':8})
    return impl

def mm4():
    impl = get_impl(matmul_op())
    impl.tile("i",{'i1':4})
    impl.tile("j",{'j1':64})
    impl.tile("k",{'k1':8})
    impl.interchange(['i','j','k','k1','i1','j1'])
    impl.vectorize(['j1'])
    impl.parallelize(['i'])
    impl.unroll({'i1':4,'k1':8})
    return impl
