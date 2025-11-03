#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from xdsl.dialects import func, builtin
from xdsl.ir import Operation

from xtc.itf.schd.scheduler import Scheduler
from xtc.schedules.descript import descript_scheduler
from xtc.utils.xdsl_aux import parse_xdsl_module
from xtc.backends.mlir.MlirNodeBackend import MlirNodeBackend
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend
from xtc.runtimes.types.ndarray import NDArray


def main():
    args = parse_args()

    with open(args.filename, "r") as f:
        source = f.read()

    module = parse_xdsl_module(source)
    # Extract the only function of the module (or fail)
    myfunc = None
    for o in module.walk():
        if isinstance(o, func.FuncOp):
            assert myfunc is None
            myfunc = o
    assert myfunc
    # Identify the scheduled operations
    ops_to_schedule = [
        op for op in myfunc.walk() if any("loop." in attr for attr in op.attributes)
    ]

    # Build the transform script
    schedulers = []
    for idx, op in enumerate(ops_to_schedule):
        parsed_id = next((k for k in op.attributes if k.startswith("__")), None)
        node_name = parsed_id if parsed_id else f"__node{idx}__"
        sched = build_node_scheduler(
            op,
            node_name,
            always_vectorize=args.always_vectorize,
            concluding_passes=args.concluding_passes,
            no_alias=not args.alias,
        )
        schedulers.append(sched)

    graph_backend = MlirGraphBackend(
        always_vectorize=args.always_vectorize,
        xdsl_func=myfunc,
        nodes=[sched.backend for sched in schedulers],
        concluding_passes=args.concluding_passes,
        no_alias=not args.alias,
    )
    graph_scheduler = graph_backend.get_scheduler(nodes_schedulers=schedulers)
    final_schedule = graph_scheduler.schedule()

    print_source = args.print_source_ir or not any(
        [
            args.evaluate,
            args.print_transformed_ir,
            args.print_lowered_ir,
            args.print_assembly,
        ]
    )

    with TemporaryDirectory() as tdir:
        dump_file = str(Path(tdir) / Path(args.filename).stem)

        compiler_args = {
            "mlir_install_dir": args.llvm_dir,
            "print_source_ir": print_source,
            "print_transformed_ir": args.print_transformed_ir,
            "print_lowered_ir": args.print_lowered_ir,
            "print_assembly": args.print_assembly,
            "visualize_jumps": not args.hide_jumps,
            "color": args.color,
            "debug": args.debug,
            "dump_file": dump_file,
            "arch": args.arch,
            "cpu": args.cpu,
            "vectors_size": args.vectors_size,
        }

        if args.evaluate:
            compiler_args["shared_lib"] = True

        compiler = graph_backend.get_compiler(**compiler_args)
        module = compiler.compile(final_schedule)

        if args.evaluate:
            if args.huge_pages:
                NDArray.set_alloc_alignment(2 * 1024 * 1024)
            else:
                NDArray.set_alloc_alignment(256)
            evaluator = module.get_evaluator(
                init_zero=args.init_zero,
                min_repeat_ms=100,
            )
            res, code, err = evaluator.evaluate()
            if code != 0:
                raise RuntimeError(f"Evaluation failed: {err}")
            print(min(res))


def build_node_scheduler(
    op: Operation,
    node_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
) -> Scheduler:
    backend = build_mlir_node_backend(
        op=op,
        node_name=node_name,
        always_vectorize=always_vectorize,
        concluding_passes=concluding_passes,
        no_alias=no_alias,
    )

    scheduler = backend.get_scheduler()
    assert isinstance(scheduler.backend, MlirNodeBackend)

    if "loop.schedule" in op.attributes:
        schedule_attribute = op.attributes.get("loop.schedule")
        assert isinstance(schedule_attribute, builtin.DictionaryAttr)
        normal_schedule = normalize_schedule(schedule_attribute)
        descript_scheduler(
            scheduler=scheduler,
            node_name=node_name,
            abstract_axis=scheduler.backend.dims,
            spec=normal_schedule,
        )
        op.attributes.pop("loop.schedule", None)

    return scheduler


def normalize_schedule(
    raw_schedule: builtin.DictionaryAttr,
) -> dict[str, dict]:
    schedule: dict[str, Any] = {}
    for declaration, val in raw_schedule.data.items():
        assert isinstance(declaration, str)
        if ":" in declaration:
            if not isinstance(val, builtin.DictionaryAttr):
                raise Exception(
                    f"The schedule within a split should be a dictionnary or void but got {declaration}"
                )

            assert isinstance(val, builtin.DictionaryAttr)
            inner_schedule = normalize_schedule(val)
            schedule[str(declaration)] = inner_schedule
        else:
            annotations: dict[str, int | None] = {}
            if isinstance(val, builtin.DictionaryAttr):
                for instr, param in val.data.items():
                    assert isinstance(instr, str)
                    if isinstance(param, builtin.UnitAttr):
                        annotations[instr] = None
                    elif isinstance(param, builtin.IntegerAttr):
                        annotations[instr] = param.value.data
                    else:
                        raise Exception(f"Annotation parameter should be void or int.")

            elif not isinstance(val, builtin.UnitAttr):
                raise Exception(
                    f"Annotation parameter should be a dict or void but got {type(val)}"
                )

            schedule[declaration] = annotations
    return schedule


def build_mlir_node_backend(
    op: Operation,
    node_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
) -> MlirNodeBackend:
    # Dims
    dims = get_string_list_attribute(op, "loop.dims")
    if not dims:
        raise Exception("Missing loop.dims attribute")
    op.attributes.pop("loop.dims", None)
    # Additional attributes
    loop_stamps = get_string_list_attribute(op, "loop.add_attributes")

    return MlirNodeBackend(
        source_op=op,
        dims=dims,
        always_vectorize=always_vectorize,
        payload_name=node_name,
        concluding_passes=concluding_passes,
        loop_stamps=loop_stamps,
        no_alias=no_alias,
        id=node_name,
    )


def get_string_list_attribute(op: Operation, attr_name: str) -> list[str]:
    attr = op.attributes.get(attr_name)
    if not attr:
        return []
    assert isinstance(attr, builtin.ArrayAttr)
    return [elem.data for elem in attr.data if isinstance(elem, builtin.StringAttr)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate code from MLIR kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "--cpu",
        type=str,
        default="native",
        help="The target CPU.",
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Vectorize even if no vectorization dimension has been specified..",
    )
    parser.add_argument(
        "--vectors-size",
        type=int,
        default=None,
        help="The default vectors size.",
    )
    parser.add_argument(
        "--print-source-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the source IR.",
    )
    parser.add_argument(
        "--alias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tensors are not considered alias-free.",
    )
    parser.add_argument(
        "--print-transformed-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the IR after application of the transform dialect.",
    )
    parser.add_argument(
        "--print-lowered-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the IR at LLVM level.",
    )
    parser.add_argument(
        "--print-assembly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the generated assembly.",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the generated code.",
    )
    parser.add_argument(
        "--huge-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="alloc at huge page boundaries",
    )
    parser.add_argument(
        "--init-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Init the output with zeros before measurement.",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow colors.",
    )
    parser.add_argument(
        "--hide-jumps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hide assembly visualization of control flow.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print debug messages.",
    )

    args = parser.parse_args()

    if not Path(args.filename).exists():
        parser.error(f"{args.filename} does not exist.")

    return args
