#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
import os
from xdsl.dialects import func, linalg
from xdsl_aux import parse_xdsl_module
from MlirModule import RawMlirModule
from MlirNodeImplementer import MlirNodeImplementer
from MlirGraphImplementer import MlirGraphImplementer
from MlirCompiler import MlirCompiler


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
        help="The prefix for LLVM/MLIR tools, or autodetected.",
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
    impl_module = RawMlirModule(source)
    compiler = MlirCompiler(
        mlir_module=impl_module,
        mlir_install_dir=args.llvm_dir,
    )
    print_source = args.print_source_ir or not (
        args.print_transformed_ir or args.print_lowered_ir or args.print_assembly
    )
    e = compiler.compile(
        print_source_ir=print_source,
        print_transformed_ir=args.print_transformed_ir,
        print_lowered_ir=args.print_lowered_ir,
        print_assembly=args.print_assembly,
        color=args.color,
        debug=args.debug,
    )
