from setup_mlir_conv import generic_conv0

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

d0 = 2
d1 = 3
d2 = 4
d3 = 2
d4 = 3
d5 = 2
d6 = 2
elt_type = f32
vectors_size = 16

def dim(i):
    return AffineExpr.dimension(i)

def get_impl(op):
    impl = MlirImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op = op,
        dims = {'d0':d0,'d1':d1,'d2':d2, 'd3':d3, 'd4':d4, 'd5':d5, 'd6':d6},
        parallel_dims = ['d0','d1','d2','d3','d4'],
        reduction_dims = ['d5','d6'],
        vectors_size = vectors_size
    )
    return impl

def conv_generic_op():

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

    conv = linalg.Generic(
        (
            TestSSAValue(MemRefType(elt_type, [d0, d1+d5, d2+d6, d3])),
            TestSSAValue(MemRefType(elt_type, [d5, d6, d3, d4])),
        ),
        (TestSSAValue(MemRefType(elt_type, [d0, d1, d2, d3, d4])),),
        Region(block),
        (
            AffineMapAttr(
                AffineMap(7,0,(dim(0), dim(1) + dim(5), dim(2) + dim(6), dim(3)))
            ),
            AffineMapAttr(
                AffineMap(7,0,(dim(5), dim(6), dim(3), dim(4)))
            ),
            AffineMapAttr(
                AffineMap(7,0,(dim(0), dim(1), dim(2), dim(3), dim(4)))
            ),
        ),
        (
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.reduction(),
            linalg.IteratorTypeAttr.reduction(),
        ),
    )


    return conv

impl = get_impl(conv_generic_op())

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
