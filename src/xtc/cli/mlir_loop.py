#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

import argparse
from pathlib import Path
from xdsl.dialects import func, builtin
from xdsl.ir import Attribute, Operation

from xtc.itf.schd.scheduler import Scheduler
from xtc.utils.xdsl_aux import parse_xdsl_module
from xtc.backends.mlir.MlirNodeBackend import MlirNodeBackend
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend
from xtc.runtimes.types.ndarray import NDArray


def main():
    args = parse_args()

    with open(args.filename, "r") as f:
        source = f.read()

    module = parse_xdsl_module(source)
    myfunc = get_unique_function(module)
    ops_to_schedule = get_annotated_operations(myfunc)

    # Build the transform script
    get_scheduler = parse_scheduler_legacy if args.old_syntax else parse_scheduler
    assert get_scheduler
    schedulers = []
    for idx, op in enumerate(ops_to_schedule):
        node_name = f"v{idx}"
        sched = get_scheduler(
            op,
            node_name,
            always_vectorize=args.always_vectorize,
            concluding_passes=args.concluding_passes,
            no_alias=args.no_alias,
        )
        schedulers.append(sched)

    graph_backend = MlirGraphBackend(
        always_vectorize=args.always_vectorize,
        xdsl_func=myfunc,
        nodes=[sched.backend for sched in schedulers],
        concluding_passes=args.concluding_passes,
        no_alias=args.no_alias,
    )
    graph_scheduler = graph_backend.get_scheduler(nodes_schedulers=schedulers)
    final_schedule = graph_scheduler.schedule()

    dump_file = Path(args.filename).stem
    print_source = args.print_source_ir or not any(
        [
            args.evaluate,
            args.print_transformed_ir,
            args.print_lowered_ir,
            args.print_assembly,
        ]
    )

    compiler_args = {
        "mlir_install_dir": args.llvm_dir,
        "to_disassemble": graph_backend.payload_name,
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
            NDArray.set_alloc_alignment(
                2 * 1024 * 1024
            )  # 2MB to catch Huge Pages if THB is one
        else:
            NDArray.set_alloc_alignment(256)  # default align to 256 bytes as DLPack
        evaluator = module.get_evaluator(
            init_zero=args.init_zero,
            min_repeat_ms=100,
        )
        res, code, err = evaluator.evaluate()
        if code != 0:
            raise RuntimeError(f"Evaluation failed: {err}")
        print(min(res))


def parse_scheduler(
    op: Operation,
    node_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
):
    backend = parse_mlir_node_backend(
        op=op,
        node_name=node_name,
        always_vectorize=always_vectorize,
        concluding_passes=concluding_passes,
        no_alias=no_alias,
    )

    scheduler = backend.get_scheduler()

    if "loop.schedule" in op.attributes:
        schedule_attribute = op.attributes.get("loop.schedule")
        assert isinstance(schedule_attribute, builtin.DictionaryAttr)
        parse_schedule(scheduler, schedule_attribute)
        remove_attribute(op, "loop.schedule")

    return scheduler


def parse_schedule(scheduler: Scheduler, schedule: builtin.DictionaryAttr):
    assert isinstance(scheduler.backend, MlirNodeBackend)

    sizes: dict[str, int | None] = {}
    interchange: list[str] = []
    tiles: dict[str, dict[str, int]] = {d: {} for d in scheduler.backend.dims}
    unroll: dict[str, int] = {}
    vecto: list[str] = []
    parallel: list[str] = []
    for declaration, val in schedule.data.items():
        # Tiles
        if "#" in declaration:
            dim_name, tile_size = declaration.split("#")
            loop_size = int(tile_size)
            tile_num = len(tiles[dim_name])
            loop_name = f"{dim_name}{tile_num}"
            tiles[dim_name][loop_name] = loop_size
        # Initial dimensions
        elif declaration in scheduler.backend.dims:
            loop_name = declaration
            loop_size = None
        else:
            raise Exception(f"Unknown declaration: {declaration}")
        sizes[loop_name] = loop_size
        # Build the interchange
        interchange.append(loop_name)
        # Annotations
        if isinstance(val, builtin.DictionaryAttr):
            for annotation, param in val.data.items():
                match annotation:
                    case "unroll":
                        if isinstance(param, builtin.UnitAttr):
                            loop_size = sizes[loop_name]
                            assert loop_size
                            unroll_factor = loop_size
                        elif isinstance(param, builtin.IntegerAttr):
                            unroll_factor = param.value.data
                        else:
                            raise Exception(f"Unknown unroll factor for {loop_name}")
                        unroll[loop_name] = unroll_factor
                    case "vectorize":
                        vecto.append(loop_name)
                    case "parallelize":
                        parallel.append(loop_name)
                    case _:
                        raise Exception(
                            f"Unknown annotation on {loop_name}: {annotation}"
                        )
        elif isinstance(val, builtin.UnitAttr):
            pass
        else:
            raise Exception(f"The annotation on {loop_name} must be a dict")

    for dim in tiles:
        scheduler.tile(dim, tiles[dim])
    scheduler.interchange(interchange)
    scheduler.vectorize(vecto)
    scheduler.parallelize(parallel)
    scheduler.unroll(unroll)
    return scheduler


def parse_scheduler_legacy(
    op: Operation,
    node_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
):
    backend = parse_mlir_node_backend(
        op=op,
        node_name=node_name,
        always_vectorize=always_vectorize,
        concluding_passes=concluding_passes,
        no_alias=no_alias,
    )

    sched = backend.get_scheduler()

    # Tiling
    tile_attr = op.attributes.get("loop.tiles")
    if tile_attr:
        assert isinstance(tile_attr, builtin.DictionaryAttr)
        for dim, tile_dict in tile_attr.data.items():
            assert isinstance(tile_dict, builtin.DictionaryAttr)
            sched.tile(
                dim,
                {
                    k: v.value.data
                    for k, v in tile_dict.data.items()
                    if isinstance(k, str) and isinstance(v, builtin.IntegerAttr)
                },
            )
        remove_attribute(op, "loop.tiles")

    # Feed the scheduler
    if "loop.interchange" in op.attributes:
        sched.interchange(get_string_list_attribute(op, "loop.interchange"))
    sched.vectorize(get_string_list_attribute(op, "loop.vectorize"))
    sched.parallelize(get_string_list_attribute(op, "loop.parallelize"))
    sched.unroll(get_string_int_dict_attribute(op, "loop.unroll"))

    for a in ["interchange", "vectorize", "parallelize", "unroll"]:
        remove_attribute(op, f"loop.{a}")

    return sched


def parse_mlir_node_backend(
    op: Operation,
    node_name: str,
    always_vectorize: bool,
    concluding_passes: list[str],
    no_alias: bool,
) -> MlirNodeBackend:
    # ID
    parsed_id = next((k for k in op.attributes if k.startswith("__")), None)
    # Dims
    dims = get_string_list_attribute(op, "loop.dims")
    if not dims:
        raise ValueError("Missing loop.dims attribute")
    remove_attribute(op, "loop.dims")
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
        id=parsed_id,
    )


def remove_attribute(operation: Operation, attr_name: str):
    operation.attributes.pop(attr_name, None)


def get_unique_function(module: builtin.ModuleOp) -> func.FuncOp:
    myfunc = None
    for o in module.walk():
        if isinstance(o, func.FuncOp):
            assert myfunc is None
            myfunc = o
    assert myfunc
    return myfunc


def get_annotated_operations(myfunc: func.FuncOp) -> list[Operation]:
    return [
        op for op in myfunc.walk() if any("loop." in attr for attr in op.attributes)
    ]


def get_string_list_attribute(op: Operation, attr_name: str) -> list[str]:
    attr = op.attributes.get(attr_name)
    if not attr:
        return []
    assert isinstance(attr, builtin.ArrayAttr)
    return [elem.data for elem in attr.data if isinstance(elem, builtin.StringAttr)]


def get_string_int_dict_attribute(op: Operation, attr_name: str) -> dict[str, int]:
    attr = op.attributes.get(attr_name)
    if not attr:
        return {}
    assert isinstance(attr, builtin.DictionaryAttr)
    return {
        key: val.value.data
        for key, val in attr.data.items()
        if isinstance(key, str) and isinstance(val, builtin.IntegerAttr)
    }


def parse_args() -> argparse.Namespace:
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
        action="store_true",
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
        "--huge-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="alloc at huge page boundaries",
    )
    parser.add_argument(
        "--init-zero",
        action="store_true",
        default=False,
        help="Init the output with zeros before measurement.",
    )
    parser.add_argument("--color", action="store_true", help="Allow colors.")
    parser.add_argument(
        "--hide-jumps",
        action="store_true",
        help="Hide assembly visualization of control flow.",
    )
    parser.add_argument(
        "--old-syntax",
        action="store_true",
        help="Parse the old version of the attributes dialect.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print debug messages."
    )

    args = parser.parse_args()

    if not Path(args.filename).exists():
        parser.error(f"{args.filename} does not exist.")

    return args
