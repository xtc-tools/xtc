#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Command-line entry point for loop exploration.

The exploration API lives in :mod:`xtc.search.explore`.
"""

import sys
import argparse
import logging
import subprocess
from typing import Sequence

from xtc.search.explore import logger, ExplorationConfig, Exploration


def launch_child(argv: Sequence[str], args: argparse.Namespace):
    env = {}
    if "tvm" in args.backends:
        # Force number of threads for TVM
        env.update({"TVM_NUM_THREADS": str(args.threads)})
    env_args = [
        "env",
        *(f"{k}={v}" for k, v in env.items()),
    ]
    setarch_args: list[str] = []
    if sys.platform.startswith("linux"):
        setarch_args = ["setarch", "-R", "--"]
    cmd = [
        *env_args,
        *setarch_args,
        argv[0],
        "--child",
        *argv[1:],
    ]
    logger.debug("Executing child command: %s", " ".join(cmd))
    proc = subprocess.run(
        args=cmd,
    )
    if proc.returncode != 0:
        logger.debug(
            f"ERROR: running subprocess: exit code: %s, command: %s",
            proc.returncode,
            " ".join(cmd),
        )
    raise SystemExit(proc.returncode)


def main():
    defaults = ExplorationConfig()
    parser = argparse.ArgumentParser(
        description="Autotune Operator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=defaults.operator,
        help="operator to optimize, use --operator-list for available operators",
    )
    parser.add_argument(
        "--operator-list",
        action="store_true",
        help="print available operators",
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Input gaph serialized yaml file",
    )
    parser.add_argument(
        "--op-name",
        type=str,
        help="operation to optimize, use --op-name-list for available operations",
    )
    parser.add_argument(
        "--op-name-list",
        action="store_true",
        help="print available operations names for the given operator",
    )
    parser.add_argument(
        "--func-name", type=str, help="function name to generate, default to operator"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=defaults.strategy,
        help=f"tile strategy to use, use --strategy-list for list of strategies",
    )
    parser.add_argument(
        "--strategy-list",
        action="store_true",
        help=f"print available strategies",
    )
    parser.add_argument(
        "--descript",
        type=str,
        help="path to a descript yaml specification. Ignores --strategy if used.",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=["random", "exhaustive", "data", "iterative"],
        default=defaults.search,
        help="search strategy",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        choices=["mlir", "tvm", "jir"],
        default=defaults.backends,
        help="backends to use",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=defaults.optimizer,
        help=f"optimizer to use, use --optimizer-list for list.",
    )
    parser.add_argument(
        "--optimizer-list",
        action="store_true",
        help=f"print list of optimizers",
    )
    parser.add_argument(
        "--data", type=str, help="data CSV file for input to data search"
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, help="dimensions, default to operators's default"
    )
    parser.add_argument(
        "--huge-pages",
        action=argparse.BooleanOptionalAction,
        default=defaults.huge_pages,
        help="alloc at huge page boundaries",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        type=int,
        default=defaults.test,
        help="test this input only",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=defaults.opt_level,
        help="opt level, 0-3 one-shot, 4 search",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=defaults.dtype,
        choices=["float32", "float64"],
        help="data type, default to operator's default",
    )
    parser.add_argument(
        "--trials", type=int, default=defaults.trials, help="num trials"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=defaults.threads,
        help="number of execution threads",
    )
    parser.add_argument(
        "--max-unroll",
        type=int,
        help="max unroll in tiling strategies, or strategy default",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed, help="seed")
    parser.add_argument(
        "--output", type=str, default=defaults.output, help="output csv file for search"
    )
    parser.add_argument(
        "--db-file",
        type=str,
        help="output json db, for instance: xtc-graphs-db.json",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=defaults.resume,
        help="resume from an existing output file and skip already recorded samples",
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=defaults.append,
        help="append new results to output file without deduplication",
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["eval"],
        default=defaults.eval,
        help="evaluation method",
    )
    parser.add_argument(
        "--repeat", type=int, default=defaults.repeat, help="evaluation repeat"
    )
    parser.add_argument(
        "--number", type=int, default=defaults.number, help="evaluation number"
    )
    parser.add_argument(
        "--min-repeat-ms",
        type=int,
        default=defaults.min_repeat_ms,
        help="evaluation min repeat ms",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=defaults.validate,
        help="validate results",
    )
    parser.add_argument(
        "--save-temps",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_temps,
        help="save temps to save temps dir",
    )
    parser.add_argument(
        "--save-temps-dir",
        type=str,
        default=defaults.save_temps_dir,
        help="save temps dir",
    )
    parser.add_argument(
        "--explore-dir",
        type=str,
        default=defaults.explore_dir,
        help="exploration results .so dir",
    )
    parser.add_argument(
        "--optimizer-config",
        type=str,
        default=defaults.optimizer_config,
        help="config yaml file for optimizer",
    )
    parser.add_argument(
        "--child",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="internal flag for marking child execution (obsolete)",
    )
    parser.add_argument(
        "--bare-ptr",
        action=argparse.BooleanOptionalAction,
        default=defaults.bare_ptr,
        help="use bare pointer interface (for TVM backend)",
    )
    parser.add_argument(
        "--jobs", type=int, default=defaults.jobs, help="parallel compile jobs"
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=defaults.execute,
        help="do not execute, only compile",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        help="machine peak flops (flop/sec) for the dtype, or estimated",
    )
    parser.add_argument(
        "--mlir-prefix", type=str, help="MLIR install prefix, defaults to mlir package"
    )
    parser.add_argument(
        "--use-tensors",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_tensors,
        help="use tensors instead of memref for the mlir backend",
    )
    parser.add_argument(
        "--batch", type=int, default=defaults.batch, help="batch size for optimizer"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="debug mode",
    )
    parser.add_argument(
        "--debug-compile",
        action=argparse.BooleanOptionalAction,
        default=defaults.debug_compile,
        help="debug compile commands",
    )
    parser.add_argument(
        "--debug-xtc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="debug xtc modules",
    )
    parser.add_argument(
        "--debug-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="debug optimizer",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=defaults.quiet,
        help="quiet optional output and progress bar",
    )
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=defaults.dump,
        help="dump IR while generating",
    )
    args = parser.parse_args()

    if args.resume and args.append:
        parser.error("--resume and --append cannot be used together")

    logging.basicConfig()
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.debug_xtc:
        logging.getLogger("xtc").setLevel(logging.DEBUG)
    if args.debug_optimizer:
        logging.getLogger("xtc.search.optimizers").setLevel(logging.INFO)

    if not args.child:
        launch_child(sys.argv, args)

    if args.operator_list:
        Exploration.list_operators()
        raise SystemExit()

    if args.op_name_list:
        assert args.operator is not None
        Exploration.list_operations_dims(args.operator)
        raise SystemExit()

    if args.strategy_list:
        Exploration.list_strategies()
        raise SystemExit()

    if args.optimizer_list:
        Exploration.list_optimizers()
        raise SystemExit()

    config = ExplorationConfig.from_args(args)
    exploration = Exploration(config)
    exploration.run()


if __name__ == "__main__":
    main()
