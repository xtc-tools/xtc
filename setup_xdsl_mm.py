#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os

from XdslImplementer import XdslImplementer

from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import TensorType, f32
from xdsl.ir import Block

home = os.environ.get("HOME", "")

i = 512
j = 128
k = 1024
elt_type = f32


def mm1():
    operands_types = [TensorType(elt_type, shape) for shape in [[i, k], [k, j], [i, j]]]
    block0 = Block(arg_types=operands_types)
    matmul = linalg.MatmulOp(
        (block0.args[0], block0.args[1]),
        (block0.args[2],),
    )

    impl = XdslImplementer(
        mlir_install_dir=f"{home}/bin/llvm-xdsl",
        source_op=matmul,
        dims={"i": i, "j": j, "k": k},
        parallel_dims=["i", "j"],
        reduction_dims=["k"],
    )

    impl.tile("i", {"i1": 8})
    impl.tile("j", {"j1": 8})
    impl.tile("k", {"k1": 16})
    impl.interchange(["i", "j", "k", "i1", "k1", "j1"])
    impl.vectorize(["j1"])
    impl.parallelize(["i"])
    impl.unroll({"k1": 8, "i1": 8})

    return impl
