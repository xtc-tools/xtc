#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
from pathlib import Path
from xdsl.dialects import func, builtin
from xdsl.ir import (
    Operation,
    Data,
)

from xtc.utils.xdsl_aux import parse_xdsl_module
from xtc.backends.mlir.MlirNodeImplementer import MlirNodeImplementer
from xtc.backends.mlir.MlirGraphImplementer import MlirGraphImplementer


def remove_attr(o: Operation, attr_name: str):
    if attr_name in o.attributes:
        del o.attributes[attr_name]


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
        raw_dict = o.attributes[attr_name]
        assert isinstance(raw_dict, builtin.DictionaryAttr)
        for string, integer in raw_dict.data.items():
            assert isinstance(string, str) and isinstance(integer, builtin.IntegerAttr)
            extracted_dict[string] = integer.value.data
    return extracted_dict


def schedule_operation(
    o: Operation,
    node_name: str,
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
    assert "loop.dims" in o.attributes
    dims = extract_string_list_from_attr(o, "loop.dims")
    assert dims
    remove_attr(o, "loop.dims")
    loop_stamps = extract_string_list_from_attr(o, "loop.add_attributes")

    impl = MlirNodeImplementer(
        source_op=o,
        dims=dims,
        always_vectorize=always_vectorize,
        payload_name=node_name,
        concluding_passes=concluding_passes,
        loop_stamps=loop_stamps,
        no_alias=no_alias,
        id=parsed_id,
    )

    sched = impl.get_scheduler()

    # Parse and process the tiling declarations
    tiles_sizes = extract_string_int_dict_from_attr(o, "loop.tiles_sizes")
    remove_attr(o, "loop.tiles_sizes")
    if "loop.tiles_names" in o.attributes:
        assert isinstance(o.attributes["loop.tiles_names"], builtin.DictionaryAttr)
        for dim, ts in o.attributes["loop.tiles_names"].data.items():
            tiles_on_dim = {}
            assert isinstance(ts, builtin.ArrayAttr)
            for t in ts:
                size = tiles_sizes[t.data]
                tiles_on_dim[t.data] = size
            sched.tile(dim, tiles_on_dim)
    remove_attr(o, "loop.tiles_names")
    # Parse the scheduling attributes
    interchange = extract_string_list_from_attr(o, "loop.interchange")
    vectorize = extract_string_list_from_attr(o, "loop.vectorize")
    parallelize = extract_string_list_from_attr(o, "loop.parallelize")
    unroll = extract_string_int_dict_from_attr(o, "loop.unroll")
    remove_attr(o, "loop.interchange")
    remove_attr(o, "loop.vectorize")
    remove_attr(o, "loop.parallelize")
    remove_attr(o, "loop.unroll")

    # Feed the scheduler
    if interchange:
        sched.interchange(interchange)
    sched.vectorize(vectorize)
    sched.parallelize(parallelize)
    sched.unroll(unroll)

    return sched


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
        "--arch",
        type=str,
        default="native",
        help="The target architecture.",
    )
    parser.add_argument(
        "--microarch",
        type=str,
        default="native",
        help="The target microarchitecture.",
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
    parser.add_argument("--color", action="store_true", help="Allow colors.")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print debug messages."
    )

    args = parser.parse_args()

    if not Path(args.filename).exists():
        parser.error(f"{args.filename} does not exist.")
    with open(args.filename, "r") as f:
        source = f.read()
    module = parse_xdsl_module(source)
    myfunc = select_xdsl_payload(module)
    annotated_operations = operations_to_schedule(myfunc)

    # Build the transform script
    count = 0
    nodes_scheds = []
    for op in annotated_operations:
        node_name = f"v{count}"
        count += 1
        sched = schedule_operation(
            op,
            node_name,
            always_vectorize=args.always_vectorize,
            concluding_passes=args.concluding_passes,
            no_alias=args.no_alias,
            evaluate=args.evaluate,
        )
        nodes_scheds.append(sched)

    impl_graph = MlirGraphImplementer(
        always_vectorize=args.always_vectorize,
        xdsl_func=myfunc,
        nodes=[sched.backend for sched in nodes_scheds],
        concluding_passes=args.concluding_passes,
        no_alias=args.no_alias,
    )
    impl_scheduler = impl_graph.get_scheduler(nodes_schedulers=nodes_scheds)
    impl_schedule = impl_scheduler.schedule()

    print_source = args.print_source_ir or (
        not args.evaluate
        and not (
            args.print_transformed_ir or args.print_lowered_ir or args.print_assembly
        )
    )

    dump_file = Path(args.filename).stem
    compiler_args = dict(
        mlir_install_dir=args.llvm_dir,
        to_disassemble=impl_graph.payload_name,
        print_source_ir=print_source,
        print_transformed_ir=args.print_transformed_ir,
        print_lowered_ir=args.print_lowered_ir,
        print_assembly=args.print_assembly,
        color=args.color,
        debug=args.debug,
        dump_file=dump_file,
        arch=args.arch,
        microarch=args.microarch,
    )
    if args.evaluate:
        compiler_args.update(dict(shared_lib=True))

    compiler = impl_graph.get_compiler(**compiler_args)
    module = compiler.compile(impl_schedule)

    if args.evaluate:
        evaluator = module.get_evaluator()
        res, code, err = evaluator.evaluate()
        assert code == 0, f"evaluation error: {err}"
        print(min(res))
