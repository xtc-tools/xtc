#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
import os
from xdsl.dialects import func, linalg
from xdsl_aux import parse_xdsl_module
from MlirNodeImplementer import MlirNodeImplementer
from MlirGraphImplementer import MlirGraphImplementer

DEFAULT_LLVM_DIR = "/home/hpompougnac/bin/llvm"


def main():
    parser = argparse.ArgumentParser(description="Blabla.")
    parser.add_argument(
        "filename",
        metavar="F",
        type=str,
        help="The source file.",
    )
    parser.add_argument(
        "--llvm-dir",
        type=str,
        default=f"{DEFAULT_LLVM_DIR}",
        help="The directory where LLVM binaries are installed.",
    )
    parser.add_argument(
        "--concluding-passes",
        metavar="N",
        type=str,
        nargs="*",
        default=[],
        help="Conclude the transform script with MLIR arbitrary passes.",
    )
    parser.add_argument(
        "--vectors-size",
        type=int,
        default=16,
        choices=[0, 4, 8, 16],
        help="The size of the vector registers (0,4,8 ou 16).",
    )
    parser.add_argument(
        "--always-vectorize",
        action="store_true",
        default=True,
        help="Vectorize even if no vectorization dimension has been specified..",
    )
    parser.add_argument(
        "--print-source-ir",
        action="store_true",
        default=False,
        help="Print the source IR.",
    )
    parser.add_argument(
        "--print-transformed-ir",
        action="store_true",
        default=False,
        help="Print the IR after application of the transform dialect.",
    )
    parser.add_argument(
        "--print-lowered-ir",
        action="store_true",
        default=False,
        help="Print the IR at LLVM level.",
    )
    parser.add_argument(
        "--print-assembly",
        action="store_true",
        default=False,
        help="Print the generated assembly.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate the generated code.",
    )
    parser.add_argument(
        "--color", action="store_true", default=True, help="Allow colors."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print debug messages."
    )

    args = parser.parse_args()
    if not os.path.exists(args.filename):
        parser.error(f"{args.filename} does not exist.")

    with open(args.filename, "r") as f:
        source = f.read()

    module = parse_xdsl_module(source)
    myfunc = None
    for o in module.walk():
        if isinstance(o, func.FuncOp):
            myfunc = o
            break
    assert myfunc
    #
    annotated_operations = []
    for o in myfunc.walk():
        for attr_name in o.attributes:
            if "loop." in attr_name:
                annotated_operations.append(o)
                break
    #
    count = 0
    impls = []
    for o in annotated_operations:
        dims = {}
        parallel_dims = []
        reduction_dims = []
        # Parse the initial specification
        for attr_name in o.attributes:
            for name, size in o.attributes["loop.dims"].data.items():
                dims[name] = size.value.data
            for d in o.attributes["loop.parallel_dims"].data:
                parallel_dims.append(d.data)
            for d in o.attributes["loop.reduction_dims"].data:
                reduction_dims.append(d.data)
            #
        loop_stamps = []
        if "loop.add_attributes" in o.attributes:
            for stamp in o.attributes["loop.add_attributes"].data:
                loop_stamps.append(stamp.data)
        implementer_name = f"v{count}"
        count += 1
        impl = MlirNodeImplementer(
            mlir_install_dir=args.llvm_dir,
            source_op=o,
            dims=dims,
            parallel_dims=parallel_dims,
            reduction_dims=reduction_dims,
            always_vectorize=args.always_vectorize,
            payload_name=implementer_name,
            concluding_passes=args.concluding_passes,
            loop_stamps=loop_stamps,
        )
        #
        if "loop.tiles_names" in o.attributes:
            for dim, ts in o.attributes["loop.tiles_names"].data.items():
                tiles_on_dim = {}
                for t in ts:
                    t = t.data
                    size = o.attributes["loop.tiles_sizes"].data[t].value.data
                    tiles_on_dim[t] = size
                impl.tile(dim, tiles_on_dim)
        if "loop.interchange" in o.attributes:
            interchange = []
            for d in o.attributes["loop.interchange"].data:
                interchange.append(d.data)
            impl.interchange(interchange)
        if "loop.vectorize" in o.attributes:
            vectorize = []
            for d in o.attributes["loop.vectorize"].data:
                vectorize.append(d.data)
            impl.vectorize(vectorize)
        if "loop.parallelize" in o.attributes:
            parallelize = []
            for d in o.attributes["loop.parallelize"].data:
                parallelize.append(d.data)
            impl.parallelize(parallelize)
        if "loop.unroll" in o.attributes:
            unroll = {}
            for name, size in o.attributes["loop.unroll"].data.items():
                unroll[name] = size.value.data
            impl.unroll(unroll)
        #
        impls.append(impl)
        #
    impl_graph = MlirGraphImplementer(
        mlir_install_dir=args.llvm_dir,
        always_vectorize=args.always_vectorize,
        xdsl_func=myfunc,
        nodes=impls,
        concluding_passes=args.concluding_passes,
    )
    if args.evaluate:
        e = impl_graph.evaluate(
            print_source_ir=args.print_source_ir,
            print_transformed_ir=args.print_transformed_ir,
            print_lowered_ir=args.print_lowered_ir,
            print_assembly=args.print_assembly,
            color=args.color,
            debug=args.debug,
        )
        print(e)
    else:
        print_source = args.print_source_ir or not (
            args.print_transformed_ir or args.print_lowered_ir or args.print_assembly
        )
        e = impl_graph.compile(
            print_source_ir=print_source,
            print_transformed_ir=args.print_transformed_ir,
            print_lowered_ir=args.print_lowered_ir,
            print_assembly=args.print_assembly,
            color=args.color,
            debug=args.debug,
        )
