#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
import os
from xtc.backends.mlir.MlirProgram import RawMlirProgram
from xtc.backends.mlir.MlirCompiler import MlirProgramCompiler


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
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow colors.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print debug messages."
    )

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        parser.error(f"{args.filename} does not exist.")
    with open(args.filename, "r") as f:
        source = f.read()
    mlir_program = RawMlirProgram(source)
    print_source = args.print_source_ir or not any(
        [
            args.print_transformed_ir,
            args.print_lowered_ir,
            args.print_assembly,
        ]
    )
    compiler = MlirProgramCompiler(
        mlir_program=mlir_program,
        mlir_install_dir=args.llvm_dir,
        print_source_ir=print_source,
        print_transformed_ir=args.print_transformed_ir,
        print_lowered_ir=args.print_lowered_ir,
        print_assembly=args.print_assembly,
        color=args.color,
        debug=args.debug,
    )
    compiler.compile()
