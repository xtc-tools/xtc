#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
import os
from xdsl.dialects import func, builtin
from xdsl.ir import (
    Operation,
)
from xdsl_aux import parse_xdsl_module
from MlirModule import MlirModule
from MlirNodeImplementer import MlirNodeImplementer
from MlirGraphImplementer import MlirGraphImplementer
from MlirCompiler import MlirCompiler


def select_xdsl_payload(module: builtin.ModuleOp) -> func.FuncOp:
    myfunc = None
    for o in module.walk():
        if isinstance(o, func.FuncOp):
            myfunc = o
            break
    assert myfunc
    return myfunc


def operations_to_schedule(myfunc: func.FuncOp) -> list[Operation]:
    annotated_operations = []
    for o in myfunc.walk():
        for attr_name in o.attributes:
            if "loop." in attr_name:
                annotated_operations.append(o)
                break
    return annotated_operations


def extract_string_list_from_attr(o: Operation, attr_name: str) -> list[str]:
    extracted_list = []
    if attr_name in o.attributes:
        raw_list = o.attributes[attr_name]
        assert isinstance(raw_list, builtin.ArrayAttr)
        for d in raw_list.data:
            assert isinstance(d, builtin.StringAttr)
            extracted_list.append(d.data)
    return extracted_list


def extract_string_int_dict_from_attr(o: Operation, attr_name: str) -> dict[str, int]:
    extracted_dict = {}
    if attr_name in o.attributes:
        raw_dict = o.attributes[attr_name].data
        for string, integer in raw_dict.items():
            assert isinstance(string, str) and isinstance(integer, builtin.IntegerAttr)
            extracted_dict[string] = integer.value.data
    return extracted_dict


def schedule_operation(
    o: Operation,
    implementer_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
    evaluate: bool,
):
    parsed_id = None
    for attr_name in o.attributes:
        if attr_name.startswith("__"):
            assert parsed_id is None
            parsed_id = attr_name

    # Parse the initial specification
    dims: dict[str, int | None] = {}
    if "loop.dims" in o.attributes:
        for k, v in extract_string_int_dict_from_attr(o, "loop.dims").items():
            dims[k] = v
    elif "loop.tiles_names" in o.attributes:
        for dim, ts in o.attributes["loop.tiles_names"].data.items():
            dims[dim] = None
    else:
        assert False
    assert dims
    parallel_dims = extract_string_list_from_attr(o, "loop.parallel_dims")
    reduction_dims = extract_string_list_from_attr(o, "loop.reduction_dims")
    loop_stamps = extract_string_list_from_attr(o, "loop.add_attributes")

    impl = MlirNodeImplementer(
        source_op=o,
        dims=dims,
        parallel_dims=parallel_dims,
        reduction_dims=reduction_dims,
        always_vectorize=always_vectorize,
        payload_name=implementer_name,
        concluding_passes=concluding_passes,
        loop_stamps=loop_stamps,
        no_alias=no_alias,
        id=parsed_id,
    )

    # Parse and process the tiling declarations
    tiles_sizes = extract_string_int_dict_from_attr(o, "loop.tiles_sizes")
    if "loop.tiles_names" in o.attributes:
        for dim, ts in o.attributes["loop.tiles_names"].data.items():
            tiles_on_dim = {}
            for t in ts:
                size = tiles_sizes[t.data]
                tiles_on_dim[t.data] = size
            impl.tile(dim, tiles_on_dim)

    # Parse the scheduling attributes
    interchange = extract_string_list_from_attr(o, "loop.interchange")
    vectorize = extract_string_list_from_attr(o, "loop.vectorize")
    if evaluate:
        parallelize = []
    else:
        parallelize = extract_string_list_from_attr(o, "loop.parallelize")
    unroll = extract_string_int_dict_from_attr(o, "loop.unroll")

    # Feed the scheduler
    if interchange:
        impl.interchange(interchange)
    impl.vectorize(vectorize)
    impl.parallelize(parallelize)
    impl.unroll(unroll)

    return impl


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
        "--concluding-passes",
        metavar="N",
        type=str,
        nargs="*",
        default=[],
        help="Conclude the transform script with MLIR arbitrary passes.",
    )
    parser.add_argument(
        "--always-vectorize",
        action="store_true",
        help="Vectorize even if no vectorization dimension has been specified..",
    )
    parser.add_argument(
        "--print-source-ir",
        action="store_true",
        default=False,
        help="Print the source IR.",
    )
    parser.add_argument(
        "--no-alias", action="store_true", help="All tensors are considered alias-free."
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
    myfunc = select_xdsl_payload(module)
    annotated_operations = operations_to_schedule(myfunc)
    if len(annotated_operations) > 0:
        # Build the transform script
        count = 0
        impls = []
        for o in annotated_operations:
            implementer_name = f"v{count}"
            count += 1
            impl = schedule_operation(
                o,
                implementer_name,
                always_vectorize=args.always_vectorize,
                concluding_passes=args.concluding_passes,
                no_alias=args.no_alias,
                evaluate=args.evaluate,
            )
            impls.append(impl)

        impl_module = MlirGraphImplementer(
            always_vectorize=args.always_vectorize,
            xdsl_func=myfunc,
            nodes=impls,
            concluding_passes=args.concluding_passes,
            no_alias=args.no_alias,
        )
    else:
        impl_module = MlirModule(xdsl_func=myfunc, no_alias=args.no_alias)

    if args.evaluate:
        impl_module.measure_execution_time()

    # Apply the transform script
    impl_module.implement()
    compiler = MlirCompiler(
        mlir_module=impl_module,
        mlir_install_dir=args.llvm_dir,
        to_disassemble=impl_module.payload_name,
    )
    if args.evaluate:
        e = compiler.evaluate(
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
        e = compiler.compile(
            print_source_ir=print_source,
            print_transformed_ir=args.print_transformed_ir,
            print_lowered_ir=args.print_lowered_ir,
            print_assembly=args.print_assembly,
            color=args.color,
            debug=args.debug,
        )
