#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.dialects.arith import Mulf, Addf
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region

i = 512
j = 128
k = 1024
elt_type = f32

block = Block(arg_types=(elt_type, elt_type, elt_type))
mulf = Mulf(block.args[0], block.args[1])
addf = Addf(block.args[2], mulf.results[0])
block.add_ops([mulf, addf, linalg.YieldOp(addf.results[0])])


op = linalg.Generic(
    (
        TestSSAValue(MemRefType(elt_type, [i, k])),
        TestSSAValue(MemRefType(elt_type, [k, j])),
    ),
    (TestSSAValue(MemRefType(elt_type, [i, j])),),
    Region(block),
    (
        AffineMapAttr(
            AffineMap(3, 0, (AffineExpr.dimension(0), AffineExpr.dimension(2)))
        ),
        AffineMapAttr(
            AffineMap(3, 0, (AffineExpr.dimension(2), AffineExpr.dimension(1)))
        ),
        AffineMapAttr(
            AffineMap(3, 0, (AffineExpr.dimension(0), AffineExpr.dimension(1)))
        ),
    ),
    (
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.reduction(),
    ),
)
print(str(op))
