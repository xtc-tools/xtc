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

import sys
import os
import argparse
import logging
import itertools
import csv
import random
import numpy as np
from tqdm import tqdm
import subprocess

import utils
from ndarray import NDArray

logger = logging.getLogger(__name__)


def xdsl_matmul(i, j, k, ftype):
    from xdsl.dialects import func, linalg
    from xdsl.dialects.builtin import TensorType, f32, f64
    from xdsl.ir import Block

    elt_type = {"f32": f32, "f64": f64}[ftype]
    operands_types = [TensorType(elt_type, shape) for shape in [[i, k], [k, j], [i, j]]]
    block0 = Block(arg_types=operands_types)
    matmul = linalg.MatmulOp(
        (block0.args[0], block0.args[1]),
        (block0.args[2],),
    )
    return matmul


def xdsl_matmul_sched(i, j, k, ftype, args):
    from XdslImplementer import XdslImplementer as impl

    op_matmul = xdsl_matmul(i, j, k, ftype)
    sched = impl(
        mlir_install_dir=f"{HOME}/bin/llvm-xdsl",
        source_op=op_matmul,
        dims={"i": i, "j": j, "k": k},
        parallel_dims=["i", "j"],
        reduction_dims=["k"],
    )
    return sched, op_matmul, "xdsl"


def mlir_matmul_sched(i, j, k, ftype, args):
    from MlirImplementer import MlirImplementer as impl

    op_matmul = xdsl_matmul(i, j, k, ftype)
    sched = impl(
        mlir_install_dir=f"{HOME}/bin/llvm-xdsl",
        source_op=op_matmul,
        dims={"i": i, "j": j, "k": k},
        parallel_dims=["i", "j"],
        reduction_dims=["k"],
    )
    return sched, op_matmul, "mlir"


def tvm_matmul(i, j, k, ftype):
    import TVMImplementer as impl

    operation = impl.Operation(impl.Operators.matmul, (i, j, k, DTYPES_MAP[ftype]))
    return operation


def tvm_matmul_sched(i, j, k, ftype, args):
    import TVMImplementer as impl

    op_matmul = tvm_matmul(i, j, k, ftype)
    sched = impl.Implementer(
        source_op=op_matmul,
        dims=dict(i=i, j=j, k=k),
        parallel_dims=["i", "j"],
    )
    return sched, op_matmul, "tvm"


def jir_matmul(i, j, k, ftype):
    import JIRImplementer as impl

    return impl.Operation(impl.Operators.matmul, (i, j, k, DTYPES_MAP[ftype]))


def jir_matmul_sched(i, j, k, ftype, args):
    import JIRImplementer as impl

    op = jir_matmul(i, j, k, ftype)
    dims = dict(i=i, j=j, k=k)
    dtype = op.args[3]
    jir_install_dir = f"{HOME}/bin/llvm-jir"
    geist_install_dir = f"{HOME}/bin/llvm-geist"
    sched = impl.Implementer(
        source_op=op,
        dims=dims,
        jir_install_dir=jir_install_dir,
        geist_install_dir=geist_install_dir,
        save_temps=args.save_temps,
        save_temps_dir=args.save_temps_dir,
    )
    # sched.save_temps = True
    return sched, op, "jir"


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


def get_eval_parameters(args):
    NDArray.set_alloc_alignment(
        2 * 1024 * 1024
    )  # 2MB to catch Huge Pages if THB is one
    dims_names = OPERATORS[args.operator]["dims"]
    dims_map = {k: v for k, v in zip(dims_names, args.dims)}
    dtype = DTYPES_MAP[args.dtype]
    inputs = OPERATORS[args.operator]["inputs"]
    outputs = OPERATORS[args.operator]["outputs"]
    inputs_spec = [
        {
            "shape": tuple([dims_map[x] for x in shape]),
            "dtype": dtype,
        }
        for shape in inputs
    ]
    outputs_spec = [
        {
            "shape": tuple([dims_map[x] for x in shape]),
            "dtype": dtype,
        }
        for shape in outputs
    ]
    nd_inputs = [NDArray(utils.np_init(**spec)) for spec in inputs_spec]
    nd_outputs = [NDArray(np.empty(**spec)) for spec in outputs_spec]
    return (nd_inputs, nd_outputs)


def evaluate_one(scheduler, tile_strategy, op_args, in_x, args, callback=None):
    assert isinstance(in_x, list), f"X not a list: {in_x} ({type(in_x)})"
    impl, op, backend = scheduler(*op_args, args)
    logger.debug(f"Evaluate: {backend}: {in_x}...")
    tile_strategy(impl, op_args, in_x)
    eval_args = {}
    if args.dump:
        eval_args.update(
            dict(
                print_source_ir=True,
                print_transformed_ir=True,
                print_lowered_ir=True,
                print_assembly=True,
                color=False,
            )
        )
    if args.eval == "jit":
        stdout = impl.evaluate(**eval_args)
    elif args.eval == "exe":
        stdout = impl.compile_and_evaluate(**eval_args)
    else:
        assert args.eval == "eval"
        impl.compile(**eval_args, shared_lib=True, dump_file="explore-payload")
        stdout = impl.load_and_evaluate(
            "explore-payload.so",
            impl.payload_name,
            repeat=args.repeat,
            number=args.number,
            min_repeat_ms=args.min_repeat_ms,
            validate=args.validate,
            parameters=args.eval_parameters,
        )
    logger.debug("timing: %s", stdout)
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

    result = (in_x, error, time, backend)
    if callback:
        callback(result)
    return result


def evaluate_exhaustive(tile_strategy, tile_generator, op_args, args, callback):
    gen_size = args.trials if args.search == "random" else None
    all_in_x = tile_generator(op_args, size=gen_size)
    all_in_x = np.array(list(all_in_x))  # convert list or generator to np.array
    if args.search == "random":
        if len(all_in_x) > args.trials:
            idxs = np.random.choice(
                np.arange(len(all_in_x)), size=args.trials, replace=False
            )
            all_in_x = all_in_x[idxs]
    for in_x in tqdm(all_in_x, smoothing=0):
        evaluate_one_backends(tile_strategy, op_args, in_x.tolist(), args, callback)


def evaluate_data(tile_strategy, X, op_args, args, callback):
    size = len(X)
    logger.debug(f"Search space size: {size}")
    for in_x in tqdm(X, smoothing=0):
        evaluate_one_backends(
            scheduler, tile_strategy, op_args, in_x.tolist(), args, callback
        )


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


def search_some(tile_strategy, tile_generator, op_args, args):
    # Search depends on search strategy
    ptime = peak_time(args)
    with open(args.output, "w", newline="") as outf:
        writer = csv.writer(outf, delimiter=";")
        writer.writerow(("X", "time", "peak", "backend"))
        outf.flush()

        def result_callback(result):
            x, error, time, backend = result
            if error != 0:
                logger.debug(f"Skip recording error for: {backend}: {x}")
                return
            peak = ptime / time
            row = [x, time, peak, backend]
            logger.debug(f"Record row: {row}")
            writer.writerow(row)
            outf.flush()

        if args.search in ["exhaustive", "random"]:
            evaluate_exhaustive(
                tile_strategy, tile_generator, op_args, args, callback=result_callback
            )
        elif args.search == "data":
            assert args.data is not None
            X = read_input(args.data, args)
            evaluate_data(tile_strategy, X, op_args, args, callback=result_callback)


def evaluate_one_backends(tile_strategy, op_args, in_x, args, callback):
    for backend in args.backends:
        scheduler = OPERATORS[args.operator]["backends"][backend]["scheduler"]
        evaluate_one(scheduler, tile_strategy, op_args, in_x, args, callback=callback)


def optimize(args):
    dims = args.dims
    dtype = args.dtype
    op_args = (*dims, dtype)
    tile_strategy = STRATEGIES[args.strategy]["strategy"]
    if args.test:
        ptime = peak_time(args)

        def output_one(results):
            in_x, error, time, backend = results
            if error == 0:
                logger.info(
                    f"Schedule: {backend}: {in_x}: time: {time * 1000:.2f} msecs, peak perf: {ptime / time * 100:.2f}%"
                )

        evaluate_one_backends(
            tile_strategy, op_args, args.test, args, callback=output_one
        )
    else:
        tile_generator = STRATEGIES[args.strategy]["generator"]
        search_some(tile_strategy, tile_generator, op_args, args)


HOME = os.environ.get("HOME", "")

THREADS = None

DTYPES_MAP = {
    "f32": "float32",
    "f64": "float64",
}

OPERATORS = {
    "matmul": {
        "dims": ["i", "j", "k"],
        "default_dims": [512, 1024, 128],
        "default_type": "f32",
        "inputs": [["i", "k"], ["k", "j"]],
        "outputs": [["i", "j"]],
        "backends": {
            "mlir": {
                "operation": xdsl_matmul,
                "scheduler": mlir_matmul_sched,
            },
            "xdsl": {
                "operation": xdsl_matmul,
                "scheduler": xdsl_matmul_sched,
            },
            "tvm": {
                "operation": tvm_matmul,
                "scheduler": tvm_matmul_sched,
            },
            "jir": {
                "operation": jir_matmul,
                "scheduler": jir_matmul_sched,
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


def launch_child(argv, args):
    env = {}
    if "tvm" in args.backends:
        # Force number of threads for TVM
        env.update({"TVM_NUM_THREADS": str(args.threads)})
    cmd = [
        "env",
        *(f"{k}={v}" for k, v in env.items()),
        "setarch",
        "-R",
        "--",
        argv[0],
        "--child",
        *argv[1:],
    ]
    code = 0
    logger.debug("Executing child command: %s", " ".join(cmd))
    try:
        subprocess.run(
            args=cmd,
            check=True,
        )
    except:
        print(f"ERROR: executing child command: {' '.join(cmd)}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


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
        choices=["mlir", "tvm", "xdsl", "jir"],
        default="mlir",
        help="backend to use",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        choices=["mlir", "tvm", "xdsl", "jir"],
        default=["mlir"],
        help="backends to use",
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
        choices=list(DTYPES_MAP.keys()),
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
        "--eval",
        type=str,
        choices=["jit", "exe", "eval"],
        default="eval",
        help="evaluation method",
    )
    parser.add_argument("--repeat", type=int, default=5, help="evaluation repeat")
    parser.add_argument("--number", type=int, default=1, help="evaluation number")
    parser.add_argument(
        "--min-repeat-ms", type=int, default=100, help="evaluation min repeat ms"
    )
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, help="validate results"
    )
    parser.add_argument(
        "--save-temps",
        action=argparse.BooleanOptionalAction,
        help="save temps to save temps dir",
    )
    parser.add_argument(
        "--save-temps-dir", type=str, help="save temps dir, default to ./save_temps_dir"
    )
    parser.add_argument(
        "--child",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="internal flag for marking child execution",
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

    if not args.child:
        launch_child(sys.argv, args)

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)

    global THREADS
    THREADS = args.threads

    if args.eval == "eval":
        args.eval_parameters = get_eval_parameters(args)

    # Workaround to ensure that TVM backend is after MLIR backends,
    # otherwise the import of tvm breaks the MLIR python bindings
    args.backends = sorted(args.backends)

    optimize(args)


if __name__ == "__main__":
    main()
