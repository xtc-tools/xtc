#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
Explore tilings for a matmult

Matmult is defined as :
- dimensions in order: i, j, k
- C[i, j] += A[i,k] * B[k,j]
- k is the reduction axis
- j is the vectorizable axis

Different tiling strategies are available:
- tile3d: 3D input vector for 1-level tiling of the three axes
  - filter only inner 3D tile with number of vector ops <= 1024
  - axes order i, k, j, i1, k1, j1
  - paralleliwe outer axe
  - unroll i1 and k1
  - vectorize j1

- tile4d: 4D input vector for 1-level tiling + ordering of the three axes
  - filter only inner 3D tile with number of vector ops <= 1024
  - axes order i, j, k + order(i1, k1, j1)
  - paralleliwe outer axes i and j
  - unroll inner axes i1, j1, k1
  - vectorize j1 if inner

- tile4dv: 4D input vector for 1-level tiling + ordering of the three axes + vectorization
  - same as tile4d and filter only orders were j1 is inner and >= 16 elts

- tile7d: 7D input vector with fixed ordering strategy inspired from Ansor sketches
  - tile all parallel axes at 4 levels
  - tile all reduction axes at 2 levels
  - order as PPRPRP where P are parallel axes and R reduction axes
  - parallelize the outer P level
  - vectorize the inner axis
  - Always unroll the inner RP levels

"""

import os
import argparse
import logging
import itertools
import csv
import random
import numpy as np
from tqdm import tqdm

from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import TensorType, f32, f64
from xdsl.ir import Block

import utils
import Implementer as mlir_impl
import TVMImplementer as tvm_impl

logger = logging.getLogger(__name__)


def mlir_matmul(i, j, k, dtype):
    operands_types = [
        TensorType(MTYPES_MAP[dtype], shape) for shape in [[i, k], [k, j], [i, j]]
    ]
    op_block = Block(arg_types=operands_types)
    op_matmul = linalg.MatmulOp(
        (op_block.args[0], op_block.args[1]), (op_block.args[2],)
    )
    return op_matmul


def mlir_matmul_sched(i, j, k, dtype):
    op_matmul = mlir_matmul(i, j, k, dtype)
    sched = mlir_impl.Implementer(
        mlir_install_dir=f"{HOME}/bin/llvm-xdsl",
        source_op=op_matmul,
        dims=dict(i=i, j=j, k=k),
        parallel_dims=["i", "j"],
    )
    return sched, op_matmul


def tvm_matmul(i, j, k, dtype):
    operation = tvm_impl.Operation(
        tvm_impl.Operators.matmul, (i, j, k, DTYPES_MAP[dtype])
    )
    return operation


def tvm_matmul_sched(i, j, k, dtype):
    op_matmul = tvm_matmul(i, j, k, dtype)
    sched = tvm_impl.Implementer(
        source_op=op_matmul,
        dims=dict(i=i, j=j, k=k),
        parallel_dims=["i", "j"],
    )
    return sched, op_matmul


def tile_strategy_3d(impl, op_args, in_x):
    # TODO: generalize: no need to be matmul specific as soon as
    # we have axes names
    i, j, k, dtype = op_args
    tiles_i = utils.factors_to_sizes(in_x[0:1])
    tiles_j = utils.factors_to_sizes(in_x[1:2])
    tiles_k = utils.factors_to_sizes(in_x[2:3])
    tiles_i_dict = {f"i{i + 1}": v for i, v in enumerate(tiles_i)}
    tiles_j_dict = {f"j{i + 1}": v for i, v in enumerate(tiles_j)}
    tiles_k_dict = {f"k{i + 1}": v for i, v in enumerate(tiles_k)}
    axes_order = ["i", "k", "j", "i1", "k1", "j1"]
    vector_axes = axes_order[-1:]
    parallel_axes = None
    if THREADS > 1:
        parallel_axes = axes_order[:1]
    unroll_axes = {"k1": tiles_k[0], "i1": tiles_i[0]}
    logger.debug(
        "input: %s: tile i: %s, j: %s, k: %s, order: %s, vector: %s, parallel: %s, unroll: %s",
        in_x,
        tiles_i_dict,
        tiles_j_dict,
        tiles_k_dict,
        axes_order,
        vector_axes,
        parallel_axes,
        unroll_axes,
    )
    impl.tile("i", tiles_i_dict)
    impl.tile("j", tiles_j_dict)
    impl.tile("k", tiles_k_dict)
    impl.interchange(axes_order)
    if parallel_axes is not None:
        impl.parallelize(parallel_axes)
    impl.vectorize(vector_axes)
    impl.unroll(unroll_axes)


def tile_generator_3d(op_args, size=None):
    i, j, k, dtype = op_args
    tiles_i = [t[0] for t in utils.factors_enumeration(i, 1)]
    tiles_j = [t[0] for t in utils.factors_enumeration(j, 1)]
    tiles_k = [t[0] for t in utils.factors_enumeration(k, 1)]
    all_tiles = [tiles_i, tiles_j, tiles_k]
    all_in_x = list(itertools.product(*all_tiles))
    logger.debug(f"Total space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    # Filter out last level if > 1024 vector elems
    all_in_x = [x for x in all_in_x if utils.mulall(x) / min(x[1], 16) <= 1024]
    logger.debug(f"Filtered space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    return np.array(all_in_x)


_tile_strategy_4d_permutations = None


def tile_strategy_4d(impl, op_args, in_x):
    # TODO: generalize: no need to be matmul specific when we have axes names
    global _tile_strategy_4d_permutations
    if _tile_strategy_4d_permutations is None:
        _tile_strategy_4d_permutations = list(
            itertools.permutations(["i1", "j1", "k1"])
        )
    i, j, k, dtype = op_args
    ti = in_x[0]
    tj = in_x[1]
    tk = in_x[2]
    order = in_x[3]
    permutations = list(_tile_strategy_4d_permutations[order])
    tiles = {"i1": ti, "j1": tj, "k1": tk}
    axes_order = ["i", "j", "k"] + permutations
    vector_axes = [axes_order[-1]] if axes_order[-1] == "j1" else None
    parallel_axes = None
    if THREADS > 1:
        parallel_axes = [axes_order[0]] if axes_order[0] in ["i", "j"] else None
        if parallel_axes is not None:
            parallel_axes += [axes_order[1]] if axes_order[1] in ["i", "j"] else []
    unroll_axes = {
        axis: tiles[axis]
        for axis in (permutations if vector_axes is None else permutations[:-1])[::-1]
    }
    logger.debug(
        "input: %s: tiles: %s, order: %s, vector: %s, parallel: %s, unroll: %s",
        in_x,
        tiles,
        axes_order,
        vector_axes,
        parallel_axes,
        unroll_axes,
    )
    impl.tile("i", {"i1": ti})
    impl.tile("j", {"j1": tj})
    impl.tile("k", {"k1": tk})
    impl.interchange(axes_order)
    if parallel_axes is not None:
        impl.parallelize(parallel_axes)
    if vector_axes is not None:
        impl.vectorize(vector_axes)
    impl.unroll(unroll_axes)


def tile_generator_4d(op_args, size=None):
    i, j, k, dtype = op_args
    tiles_i = [t[0] for t in utils.factors_enumeration(i, 1)]
    tiles_j = [t[0] for t in utils.factors_enumeration(j, 1)]
    tiles_k = [t[0] for t in utils.factors_enumeration(k, 1)]
    orders = list(range(6))  # 6 permutations for 3 axes
    all_tiles = [tiles_i, tiles_j, tiles_k, orders]
    all_in_x = list(itertools.product(*all_tiles))
    logger.debug(f"Total space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    # Filter out last level if > 1024 vector elems.
    # x[1] is the tile size for the vector axis, x[-1] in [1, 4] is true when the vector axis is inner
    all_in_x = [
        x
        for x in all_in_x
        if utils.mulall(x[:-1]) / min(x[1], max((x[-1] in [1, 4]) * 16, 1)) <= 1024
    ]
    logger.debug(f"Filtered space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    return np.array(all_in_x)


def tile_generator_4dv(op_args, size=None):
    i, j, k, dtype = op_args
    all_in_x = tile_generator_4d(op_args)
    # Keep only vectorized dims, i.e. x[-1] in [1, 4] and tile j >= 16
    all_in_x = [x for x in all_in_x if (x[-1] in [1, 4] and x[1] >= 16)]
    logger.debug(f"Filtered space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    return np.array(all_in_x)


def tile_strategy_7d(impl, op_args, in_x):
    # TODO: generalize: no need to be matmul specific as soon as
    # we have axes names
    i, j, k, dtype = op_args
    tiles_i = utils.factors_to_sizes(in_x[0:3])
    tiles_j = utils.factors_to_sizes(in_x[3:6])
    tiles_k = utils.factors_to_sizes(in_x[6:7])
    tiles_i_dict = {f"i{i + 1}": v for i, v in enumerate(tiles_i)}
    tiles_j_dict = {f"j{i + 1}": v for i, v in enumerate(tiles_j)}
    tiles_k_dict = {f"k{i + 1}": v for i, v in enumerate(tiles_k)}
    axes_order = ["i", "j", "i1", "j1", "k", "i2", "j2", "k1", "i3", "j3"]
    vector_axes = axes_order[-1:]
    parallel_axes = None
    if THREADS > 1:
        parallel_axes = axes_order[:2]
    unroll_axes = {"i3": tiles_i[-1], "k1": tiles_k[-1]}
    logger.debug(
        "input: %s: tile i: %s, j: %s, k: %s, order: %s, vector: %s, parallel: %s, unroll: %s",
        in_x,
        tiles_i_dict,
        tiles_j_dict,
        tiles_k_dict,
        axes_order,
        vector_axes,
        parallel_axes,
        unroll_axes,
    )
    impl.tile("i", tiles_i_dict)
    impl.tile("j", tiles_j_dict)
    impl.tile("k", tiles_k_dict)
    impl.interchange(axes_order)
    if parallel_axes is not None:
        impl.parallelize(parallel_axes)
    impl.vectorize(vector_axes)
    impl.unroll(unroll_axes)


def tile_generator_7d(op_args, size=None):
    i, j, k, dtype = op_args
    tiles_i = utils.factors_enumeration(i, 3)
    tiles_j = utils.factors_enumeration(j, 3)
    tiles_k = utils.factors_enumeration(k, 1)
    all_tiles = tiles_i + tiles_j + tiles_k
    space_size = len(tiles_i) * len(tiles_j) * len(tiles_k)
    logger.debug(f"Raw space size: {space_size} for problem dims: {i}x{j}x{k}")

    def in_space(x):
        # Filter out last level if > 1024 vector elems
        return x[2] * x[5] * x[6] / min(x[5], 16) <= 1024

    if size is None or size >= space_size:
        logger.debug(f"Generate exhaustive")
        # Exhaustive
        for ti in tiles_i:
            for tj in tiles_j:
                for tk in tiles_k:
                    x = ti + tj + tk
                    if in_space(x):
                        yield np.array(x)
    else:
        logger.debug(f"Generate random {size}")
        seen = set()
        noop_num = 0
        while len(seen) < size and noop_num < 10000:
            noop_num += 1
            xi = random.randint(0, len(tiles_i) - 1)
            xj = random.randint(0, len(tiles_j) - 1)
            xk = random.randint(0, len(tiles_k) - 1)
            x = tiles_i[xi] + tiles_j[xj] + tiles_k[xk]
            if not in_space(x):
                continue
            tx = tuple(x)
            if tx in seen:
                continue
            noop_num = 0
            seen.add(tx)
            yield np.array(x)


def evaluate_one(scheduler, tile_strategy, op_args, in_x, args, callback=None):
    assert isinstance(in_x, list), f"X not a list: {in_x} ({type(in_x)})"
    logger.debug(f"Evaluate: {in_x}...")
    impl, op = scheduler(*op_args)
    tile_strategy(impl, op_args, in_x)
    eval_args = {}
    if args.dump:
        eval_args.update(
            dict(
                print_source_ir=True,
                print_transformed_ir=True,
                print_assembly=True,
            )
        )
    stdout = impl.evaluate(**eval_args)
    logger.debug("STDOUT: %s", stdout)
    error = 0
    try:
        # TODO: for now we detect errors when trying to parse the result
        time = float(stdout)
    except ValueError:
        time = 0
        error = 1

    if error == 0:
        logger.debug(f"Schedule: {in_x}: time: {time * 1000:.2f} msecs")
    else:
        logger.error(f"Error evaluating: {in_x}")

    result = (in_x, error, time)
    if callback:
        callback(result)
    return result


def evaluate_exhaustive(
    scheduler, tile_strategy, tile_generator, op_args, args, callback=None
):
    gen_size = args.trials if args.search == "random" else None
    all_in_x = tile_generator(op_args, size=gen_size)
    all_in_x = np.array(list(all_in_x))  # convert list or generator to np.array
    if args.search == "random":
        if len(all_in_x) > args.trials:
            idxs = np.random.choice(
                np.arange(len(all_in_x)), size=args.trials, replace=False
            )
            all_in_x = all_in_x[idxs]
    results = []
    for in_x in tqdm(all_in_x, smoothing=0):
        result = evaluate_one(
            scheduler, tile_strategy, op_args, in_x.tolist(), args, callback
        )
        results.append(result)
    return results


def evaluate_data(scheduler, tile_strategy, X, op_args, args, callback=None):
    size = len(X)
    logger.debug(f"Search space size: {size}")
    results = []
    for in_x in tqdm(X, smoothing=0):
        result = evaluate_one(
            scheduler, tile_strategy, op_args, in_x.tolist(), args, callback
        )
        results.append(result)
    return results


def read_input(fname, args):
    X = []
    with open(fname, newline="") as infile:
        reader = csv.reader(infile, delimiter=";")
        for idx, row in enumerate(reader):
            if idx == 0:
                X_idx = row.index("X")
            else:
                X.append(eval(row[X_idx], {}, {}))
    return np.array(X)


def peak_time(args):
    return utils.cpu_peak_time(
        utils.mulall(args.dims), DTYPES_MAP[args.dtype], args.threads
    )


def search_some(scheduler, tile_strategy, tile_generator, op_args, args):
    # Search depends on search strategy
    ptime = peak_time(args)
    with open(args.output, "w", newline="") as outf:
        writer = csv.writer(outf, delimiter=";")
        writer.writerow(("X", "time", "peak"))
        outf.flush()

        def result_callback(result):
            x, error, time = result
            if error != 0:
                logger.debug(f"Skip recording error for: {x}")
                return
            peak = ptime / time
            row = [x, time, peak]
            logger.debug(f"Record row: {row}")
            writer.writerow(row)
            outf.flush()

        if args.search in ["exhaustive", "random"]:
            results = evaluate_exhaustive(
                scheduler,
                tile_strategy,
                tile_generator,
                op_args,
                args,
                callback=result_callback,
            )
        elif args.search == "data":
            assert args.data is not None
            X = read_input(args.data, args)
            results = evaluate_data(
                scheduler, tile_strategy, X, op_args, args, callback=result_callback
            )


def optimize(args):
    dims = args.dims
    dtype = args.dtype
    op_args = (*dims, dtype)
    scheduler = OPERATORS[args.operator]["backends"][args.backend]["scheduler"]
    tile_strategy = STRATEGIES[args.strategy]["strategy"]
    if args.test:
        ptime = peak_time(args)
        in_x, error, time = evaluate_one(
            scheduler, tile_strategy, op_args, args.test, args
        )
        if error == 0:
            logger.info(
                f"Schedule: {in_x}: time: {time * 1000:.2f} msecs, peak perf: {ptime / time * 100:.2f}%"
            )
    else:
        tile_generator = STRATEGIES[args.strategy]["generator"]
        search_some(scheduler, tile_strategy, tile_generator, op_args, args)


HOME = os.environ.get("HOME", "")

THREADS = None

MTYPES_MAP = {
    "f32": f32,
    "f64": f64,
}

DTYPES_MAP = {
    "f32": "float32",
    "f64": "float64",
}

OPERATORS = {
    "matmul": {
        "dims": ["i", "j", "k"],
        "default_dims": [512, 1024, 128],
        "default_type": "f32",
        "backends": {
            "mlir": {
                "operation": mlir_matmul,
                "scheduler": mlir_matmul_sched,
            },
            "tvm": {
                "operation": tvm_matmul,
                "scheduler": tvm_matmul_sched,
            },
        },
    },
}

STRATEGIES = {
    "tile3d": {
        "strategy": tile_strategy_3d,
        "generator": tile_generator_3d,
    },
    "tile4d": {
        "strategy": tile_strategy_4d,
        "generator": tile_generator_4d,
    },
    "tile4dv": {
        "strategy": tile_strategy_4d,
        "generator": tile_generator_4dv,
    },
    "tile7d": {
        "strategy": tile_strategy_7d,
        "generator": tile_generator_7d,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Autotune Matmult",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operator",
        type=str,
        choices=list(OPERATORS.keys()),
        default="matmul",
        help="operator to optimize",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGIES.keys()),
        default="tile3d",
        help="tile strategy to use",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=["random", "exhaustive", "data"],
        default="random",
        help="search strategy",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mlir", "tvm"],
        default="mlir",
        help="backend to use",
    )
    parser.add_argument(
        "--data", type=str, help="data CSV file for input to data search"
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=OPERATORS["matmul"]["default_dims"],
        help="dimensions",
    )
    parser.add_argument(
        "--test", nargs="+", type=int, default=[], help="test this input only"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(MTYPES_MAP.keys()),
        default=OPERATORS["matmul"]["default_type"],
        help="data type",
    )
    parser.add_argument("--trials", type=int, default=10000, help="num trials")
    parser.add_argument("--threads", type=int, default=1, help="number of threads")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--output", type=str, default="results.csv", help="output csv file for search"
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="debug mode"
    )
    parser.add_argument(
        "--dump", action=argparse.BooleanOptionalAction, help="dump IR while generating"
    )
    args = parser.parse_args()

    logging.basicConfig()
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)

    global THREADS
    THREADS = args.threads

    if args.backend == "tvm":
        # Force number of threads for TVM
        os.environ["TVM_NUM_THREADS"] = str(args.threads)

    optimize(args)


if __name__ == "__main__":
    main()
