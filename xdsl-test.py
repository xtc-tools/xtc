#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os

from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import TensorType, f32
from xdsl.ir import Block

import Implementer
from Implementer import Implementer

home = os.environ.get("HOME", "")

i = 512
j = 128
k = 1024
elt_type = f32

operands_types = [TensorType(elt_type, shape) for shape in [[i, k], [k, j], [i, j]]]
block0 = Block(arg_types=operands_types)
matmul = linalg.MatmulOp(
    (block0.args[0], block0.args[1]),
    (block0.args[2],),
)

impl = Implementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op=matmul,
    dims={"i": i, "j": j, "k": k},
    parallel_dims=["i", "j"],
)

impl.tile("i", {"i1": 8})
impl.tile("j", {"j1": 8})
impl.tile("k", {"k1": 8})
impl.interchange(["i", "k", "j", "i1", "j1", "k1"])
impl.vectorize(["k1"])
impl.parallelize(["i"])
impl.unroll({"j1": 8, "i1": 8})

e = impl.evaluate(
    # print_source_ir=True,
    # print_transformed_ir=True,
    # print_ir_after=['convert-vector-to-llvm'],
    # print_ir_before=['test-transform-dialect-erase-schedule'],
    print_assembly=True,
    # color = True,
)
print(e)
