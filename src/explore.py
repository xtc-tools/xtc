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
  - same as tile4d and filter only orders were j1 is inner and >= VEC_SIZE elts

- tile7d: 7D input vector with fixed ordering strategy inspired from Ansor sketches
  - tile all parallel axes at 4 levels
  - tile all reduction axes at 2 levels
  - order as PPRPRP where P are parallel axes and R reduction axes
  - parallelize the outer P level
  - vectorize the inner axis
  - Always unroll the inner RP levels

- tile4dv: 7D input vector with fixed ordering strategy inspired from Ansor sketches + vectorization
  - same as tile7d where j inner >= VEC_SIZE elts

"""

import sys
import os
import argparse
import logging
import itertools
import functools
import csv
import random
import numpy as np
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
from pathlib import Path

import utils
from ndarray import NDArray
import runtime

logger = logging.getLogger(__name__)


def reference_matmul(a, b, c):
    np.matmul(a, b, out=c)


def xdsl_matmul_graph(i, j, k, ftype, name="matmul"):
    from xdsl.dialects import func, linalg, arith, builtin
    from xdsl.dialects.builtin import MemRefType, f32, f64, UnitAttr
    from xdsl.ir import Block, Region
    from xdsl.builder import ImplicitBuilder

    elt_type = {"f32": f32, "f64": f64}[ftype]
    ops_types = [MemRefType(elt_type, shape) for shape in [[i, k], [k, j], [i, j]]]
    block = Block(arg_types=ops_types)
    with ImplicitBuilder(block):
        cst0 = arith.ConstantOp(builtin.FloatAttr(0, 32))
        fill = linalg.FillOp(
            res=(),
            inputs=(cst0.results[0],),
            outputs=(block.args[2],),
        )
        matmul = linalg.MatmulOp(
            res=(),
            inputs=(block.args[0], block.args[1]),
            outputs=(block.args[2],),
        )
        func.ReturnOp()
    fill.attributes["__xtc_id_fill__"] = UnitAttr()
    matmul.attributes["__xtc_id_matmul__"] = UnitAttr()
    region = Region([block])
    payload = func.FuncOp.from_region(
        name=name,
        input_types=ops_types,
        return_types=[],
        region=region,
    )
    args = {f"arg{i}": arg for i, arg in enumerate(block.args)}
    graph = {
        "args": args,
        "inps": ["arg0", "arg1"],
        "outs": ["arg2"],
        "payload": payload,
        "nodes": {
            "fill": {
                "args": ["arg2"],
                "inps": [],
                "outs": ["arg2"],
                "dims": {"i": i, "j": j},
                "parallel_dims": ["i", "j"],
                "reduction_dims": [],
                "op": fill,
            },
            "matmul": {
                "args": ["arg0", "arg1", "arg2"],
                "inps": ["arg0", "arg1", "arg2"],
                "outs": ["arg2"],
                "dims": {"i": i, "j": j, "k": k},
                "parallel_dims": ["i", "j"],
                "reduction_dims": ["k"],
                "op": matmul,
            },
        },
    }
    return graph


def mlir_matmul_impl(i, j, k, ftype, graph, args):
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import FunctionType, MemRefType, f32, f64
    from xdsl.dialects import func
    from MlirGraphImplementer import MlirGraphImplementer
    from MlirCompiler import MlirCompiler
    from MlirNodeImplementer import MlirNodeImplementer

    mlir_nodes = {
        ident: MlirNodeImplementer(
            payload_name=ident,
            source_op=node["op"],
            dims=node["dims"],
            parallel_dims=node["parallel_dims"],
            reduction_dims=node["reduction_dims"],
            no_alias=True,
            always_vectorize=True,
            id=f"__xtc_id_{ident}__",
        )
        for ident, node in graph["nodes"].items()
    }
    impl = MlirGraphImplementer(
        xdsl_func=graph["payload"],
        nodes=list(mlir_nodes.values()),
    )
    compiler = MlirCompiler(
        mlir_module=impl,
        to_disassemble=impl.payload_name,
    )
    node_scheduler = mlir_nodes["matmul"].get_scheduler()
    scheduler = impl.get_scheduler()
    source_op = mlir_nodes["matmul"].source_op
    return compiler, scheduler, node_scheduler, source_op, "mlir"


def tvm_matmul_graph(i, j, k, ftype, name="matmul"):
    # Note that mlir, tvm import order causes issues
    import tvm, tvm.te
    import TVMImplementer

    matmul = TVMImplementer.Operation(
        TVMImplementer.Operators.matmul,
        (i, j, k, DTYPES_MAP[ftype]),
        name=name,
    )
    return {
        "nodes": {
            "matmul": {
                "dims": {"i": i, "j": j, "k": k},
                "parallel_dims": ["i", "j"],
                "reduction_dims": ["k"],
                "op": matmul,
            }
        }
    }


def tvm_matmul_impl(i, j, k, ftype, graph, args):
    import TVMImplementer

    node = graph["nodes"]["matmul"]
    impl = TVMImplementer.Implementer(
        source_op=node["op"],
        dims=node["dims"],
        parallel_dims=node["parallel_dims"],
    )
    compiler = impl
    node_scheduler = impl.get_scheduler()
    return compiler, node_scheduler, node_scheduler, node["op"], "tvm"


def jir_matmul_graph(i, j, k, ftype, name="matmul"):
    import JIRImplementer

    matmul = JIRImplementer.Operation(
        JIRImplementer.Operators.matmul,
        (i, j, k, DTYPES_MAP[ftype]),
        name=name,
    )
    return {
        "nodes": {
            "matmul": {
                "dims": {"i": i, "j": j, "k": k},
                "parallel_dims": ["i", "j"],
                "reduction_dims": ["k"],
                "op": matmul,
            }
        }
    }


def jir_matmul_impl(i, j, k, ftype, graph, args):
    import JIRImplementer

    node = graph["nodes"]["matmul"]
    # TODO: MLIR JIR install dirs
    jir_install_dir = f"{HOME}/bin/llvm-jir"
    geist_install_dir = f"{HOME}/bin/llvm-geist"
    impl = JIRImplementer.Implementer(
        source_op=node["op"],
        dims=node["dims"],
        jir_install_dir=jir_install_dir,
        geist_install_dir=geist_install_dir,
    )
    compiler = impl
    node_scheduler = impl.get_scheduler()
    return compiler, node_scheduler, node_scheduler, node["op"], "jir"


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
    unroll_axes = {"j1": tiles_j[0], "k1": tiles_k[0], "i1": tiles_i[0]}
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


def tile_schedule_default_3d(opt_level, op_args):
    i, j, k, dtype = op_args
    default_schedule = [1, 1, 1]
    if opt_level >= 3:
        jtile = VEC_SIZE
        itile = 2  # todo IPC
        ktile = 1
        idiv = i >= itile and i % itile == 0
        jdiv = j >= jtile and j % jtile == 0
        kdiv = k >= ktile and k % ktile == 0
        if idiv and jdiv and kdiv:
            default_schedule = [itile, jtile, ktile]
    return default_schedule


def tile_generator_3d(op_args, size=None):
    i, j, k, dtype = op_args
    tiles_i = [t[0] for t in utils.factors_enumeration(i, 1)]
    tiles_j = [t[0] for t in utils.factors_enumeration(j, 1)]
    tiles_k = [t[0] for t in utils.factors_enumeration(k, 1)]
    all_tiles = [tiles_i, tiles_j, tiles_k]
    all_in_x = list(itertools.product(*all_tiles))
    logger.debug(f"Total space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    # Filter out last level if > 1024 vector elems
    all_in_x = [
        x for x in all_in_x if utils.mulall(x) / min(x[1], VEC_SIZE) <= MAX_UNROLL
    ]
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
    unroll_axes = {axis: tiles[axis] for axis in permutations[::-1]}
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


def tile_schedule_default_4d(opt_level, op_args):
    i, j, k, dtype = op_args
    default_schedule = [1, 1, 1, 0]
    if opt_level >= 2:
        default_schedule = [1, 1, 1, 1]
    if opt_level >= 3:
        jtile = VEC_SIZE
        itile = 2  # todo IPC
        ktile = 1
        idiv = i >= itile and i % itile == 0
        jdiv = j >= jtile and j % jtile == 0
        kdiv = k >= ktile and k % ktile == 0
        if idiv and jdiv and kdiv:
            default_schedule = [itile, jtile, ktile, 1]
    return default_schedule


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
        if utils.mulall(x[:-1]) / min(x[1], max((x[-1] in [1, 4]) * VEC_SIZE, 1))
        <= MAX_UNROLL
    ]
    logger.debug(f"Filtered space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    return np.array(all_in_x)


def tile_generator_4dv(op_args, size=None):
    i, j, k, dtype = op_args
    all_in_x = tile_generator_4d(op_args)
    # Keep only vectorized dims, i.e. x[-1] in [1, 4] and tile j >= VEC_SIZE
    all_in_x = [x for x in all_in_x if (x[-1] in [1, 4] and x[1] >= VEC_SIZE)]
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
    unroll_axes = {"j3": tiles_j[-1], "i3": tiles_i[-1], "k1": tiles_k[-1]}
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


def tile_schedule_default_7d(opt_level, op_args):
    i, j, k, dtype = op_args
    default_schedule = [1, 1, 1, 1, 1, 1, 1]
    if opt_level >= 3:
        jtile = VEC_SIZE
        itile = 2  # todo IPC
        ktile = 1
        idiv = i >= itile and i % itile == 0
        jdiv = j >= jtile and j % jtile == 0
        kdiv = k >= ktile and k % ktile == 0
        if idiv and jdiv and kdiv:
            default_schedule = [1, 1, itile, 1, 1, jtile, ktile]
    return default_schedule


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
        return x[2] * x[5] * x[6] / min(x[5], VEC_SIZE) <= MAX_UNROLL

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


def tile_generator_7dv(op_args, size=None):
    i, j, k, dtype = op_args
    all_in_x = tile_generator_7d(op_args)
    # Keep only vectorized dims, i.e. x[5] (inner j) >= VEC_SIZE
    all_in_x = [x for x in all_in_x if x[5] >= VEC_SIZE]
    logger.debug(f"Filtered space size: {len(all_in_x)} for problem dims: {i}x{j}x{k}")
    return np.array(all_in_x)


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


def get_all_impls(op_args, args):
    return [
        (backend, OPERATORS[args.operator]["backends"][backend]["operation"](*op_args))
        for backend in args.backends
    ]


def compile_one_impls(ident, impls, tile_strategy, op_args, in_x, args, callbacks):
    compiled = []
    for name, impl in impls:
        task_ident = f"{args.operator}_{name}_{ident}"
        compiled.append(
            compile_one(
                task_ident,
                name,
                impl,
                tile_strategy,
                op_args,
                in_x,
                args,
                callbacks=callbacks,
            )
        )
    return compiled


def compile_one(
    ident,
    backend,
    operation,
    tile_strategy,
    op_args,
    in_x,
    args,
    callbacks=None,
    dump_file=None,
):
    assert isinstance(in_x, list), f"X not a list: {in_x} ({type(in_x)})"
    logger.debug("Compile: %s: %s: %s...", ident, backend, in_x)
    implementer = OPERATORS[args.operator]["backends"][backend]["implementer"]
    compiler, scheduler, node_scheduler, source_op, backend_name = implementer(
        *op_args, operation, args
    )
    assert backend_name == backend
    tile_strategy(node_scheduler, op_args, in_x)
    schedule = scheduler.implement()
    if dump_file is None:
        dump_file = f"{args.explore_dir}/payload_{ident}"
    compile_args = dict(
        schedule=schedule,
        shared_lib=True,
        dump_file=dump_file,
        no_entry=True,
    )
    if args.dump:
        compile_args.update(
            dict(
                print_source_ir=True,
                print_transformed_ir=True,
                print_lowered_ir=True,
                print_assembly=True,
                color=False,
            )
        )
    if args.save_temps:
        compile_args.update(
            dict(
                save_temps=True,
                save_temps_dir=f"{args.save_temps_dir}/{ident}",
            )
        )
    assert args.eval == "eval"
    compiler.compile(**compile_args)
    logger.debug("  Compile done: %s: %s.", ident, in_x)
    return (ident, backend, compiler, dump_file, in_x)


def load_and_evaluate_one(
    ident, backend, compiler, dump_file, in_x, args, callbacks=None
):
    logger.debug("Evaluate: %s: %s...", ident, in_x)
    payload_lib = f"{dump_file}.so"
    payload_name = compiler.payload_name
    stdout = compiler.load_and_evaluate(
        payload_lib,
        payload_name,
        repeat=args.repeat,
        number=args.number,
        min_repeat_ms=args.min_repeat_ms,
        validate=args.validate,
        parameters=args.eval_parameters,
        reference=reference_matmul,
    )
    if not args.save_temps:
        Path(payload_lib).unlink()
    error = 0
    try:
        # TODO: for now we detect errors when trying to parse the result
        time = float(stdout)
    except ValueError:
        time = 0
        error = 1

    if error == 0:
        logger.debug("  Evaluated: %s: %s: time: %.2f msecs", ident, in_x, time * 1000)
    else:
        logger.error("Error evaluating: %s: %s", ident, in_x)

    result = (in_x, error, time, backend)
    if callbacks and "result" in callbacks:
        callbacks["result"](result)
    return result


def evaluate_all_parallel(tile_strategy, all_in_x, impls, op_args, args, callbacks):
    jobs = args.jobs

    def do_compile(idx, in_x):
        return idx, compile_one_impls(
            ident=f"{idx:04}",
            impls=impls,
            in_x=in_x,
            tile_strategy=tile_strategy,
            op_args=op_args,
            args=args,
            callbacks=None,
        )

    nback = len(args.backends)
    ntasks = len(all_in_x) * nback
    tqdm_args = dict(
        total=ntasks, miniters=0, mininterval=0, smoothing=0, disable=args.quiet
    )
    allbar = tqdm(desc="Overall", colour="red", position=0, **tqdm_args)
    compbar = tqdm(desc="Compile", colour="blue", position=1, **tqdm_args)
    if args.execute:
        evalbar = tqdm(desc="Execute", colour="green", position=2, **tqdm_args)
    try:
        for job_idx, job_in_x in enumerate(
            np.array_split(all_in_x, np.ceil(len(all_in_x) / jobs), axis=0)
        ):
            compbar.unpause()
            job_compiled = []
            if jobs == 1:
                job_compiled.append(
                    do_compile(
                        idx=job_idx,
                        in_x=job_in_x[0].tolist(),
                    )
                )
                compbar.update(nback)
                if not args.execute:
                    allbar.update(nback)
            else:

                def future_callback(future):
                    job_compiled.append(future.result())
                    compbar.update(nback)
                    if not args.execute:
                        allbar.update(nback)

                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    futures = [
                        executor.submit(
                            do_compile,
                            idx=job_idx * jobs + idx,
                            in_x=in_x.tolist(),
                        )
                        for idx, in_x in enumerate(job_in_x)
                    ]
                    for future in futures:
                        future.add_done_callback(future_callback)
            compbar.update(0)
            allbar.update(0)

            if args.execute:
                compiled_results = [
                    x[1] for x in sorted(job_compiled, key=lambda x: x[0])
                ]
                evalbar.unpause()
                for compiled_list in compiled_results:
                    for compiled in compiled_list:
                        ident, backend, impl, dump_file, in_x = compiled
                        load_and_evaluate_one(
                            ident,
                            backend,
                            impl,
                            dump_file,
                            in_x,
                            args,
                            callbacks=callbacks,
                        )
                        evalbar.update(1)
                        allbar.update(1)
    finally:
        compbar.unpause()
        if args.execute:
            evalbar.unpause()
        allbar.close()
        compbar.close()
        if args.execute:
            evalbar.close()


def evaluate_generate(tile_strategy, tile_generator, impls, op_args, args, callbacks):
    gen_size = args.trials if args.search == "random" else None
    all_in_x = tile_generator(op_args, size=gen_size)
    all_in_x = np.array(list(all_in_x))  # convert list or generator to np.array
    if args.search == "random":
        if len(all_in_x) > args.trials:
            idxs = np.random.choice(
                np.arange(len(all_in_x)), size=args.trials, replace=False
            )
            all_in_x = all_in_x[idxs]
    evaluate_all_parallel(tile_strategy, all_in_x, impls, op_args, args, callbacks)


def evaluate_data(tile_strategy, X, impls, op_args, args, callbacks):
    size = len(X)
    logger.debug(f"Search space size: {size}")
    evaluate_all_parallel(tile_strategy, X, impls, op_args, args, callbacks)


def evaluate_one(tile_strategy, in_x, impls, op_args, args, callbacks):
    evaluate_all_parallel(
        tile_strategy, np.array([in_x]), impls, op_args, args, callbacks
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
    if not args.execute:
        return 0
    dtype = DTYPES_MAP[args.dtype]
    flops = runtime.evaluate_flops(dtype)
    assert flops != 0, f"unable to evaluate machine flops for type {dtype}"
    flop = utils.mulall(args.dims)
    time = flop / flops / args.threads
    return time


def search_some(tile_strategy, tile_generator, impls, op_args, args):
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
            evaluate_generate(
                tile_strategy,
                tile_generator,
                impls,
                op_args,
                args,
                callbacks={"result": result_callback},
            )
        elif args.search == "data":
            assert args.data is not None
            X = read_input(args.data, args)
            evaluate_data(
                tile_strategy,
                X,
                impls,
                op_args,
                args,
                callbacks={"result": result_callback},
            )


def optimize(args):
    dims = args.dims
    dtype = args.dtype
    op_args = (*dims, dtype)
    tile_strategy = STRATEGIES[args.strategy]["strategy"]
    impls = get_all_impls(op_args, args)
    if args.test:
        all_results = []

        def output_one(results):
            all_results.append(results)

        callbacks = {"result": output_one}
        evaluate_one(
            tile_strategy, args.test, impls, op_args, args, callbacks=callbacks
        )
        ptime = peak_time(args)
        for results in all_results:
            in_x, error, time, backend = results
            if error == 0:
                tqdm.write(
                    f"Schedule: {backend}: {in_x}: time: {time * 1000:.2f} msecs, peak perf: {ptime / time * 100:.2f}%"
                )
    else:
        tile_generator = STRATEGIES[args.strategy]["generator"]
        search_some(tile_strategy, tile_generator, impls, op_args, args)


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
                "operation": xdsl_matmul_graph,
                "implementer": mlir_matmul_impl,
            },
            "tvm": {
                "operation": tvm_matmul_graph,
                "implementer": tvm_matmul_impl,
            },
            "jir": {
                "operation": jir_matmul_graph,
                "implementer": jir_matmul_impl,
            },
        },
    },
}

STRATEGIES = {
    "tile3d": {
        "strategy": tile_strategy_3d,
        "generator": tile_generator_3d,
        "schedule": tile_schedule_default_3d,
    },
    "tile4d": {
        "strategy": tile_strategy_4d,
        "generator": tile_generator_4d,
        "schedule": tile_schedule_default_4d,
    },
    "tile4dv": {
        "strategy": tile_strategy_4d,
        "generator": tile_generator_4dv,
        "schedule": tile_schedule_default_4d,
    },
    "tile7d": {
        "strategy": tile_strategy_7d,
        "generator": tile_generator_7d,
        "schedule": tile_schedule_default_7d,
    },
    "tile7dv": {
        "strategy": tile_strategy_7d,
        "generator": tile_generator_7dv,
        "schedule": tile_schedule_default_7d,
    },
}


def setup_args(args):
    global THREADS
    global MAX_UNROLL
    global VEC_SIZE
    THREADS = args.threads
    MAX_UNROLL = args.max_unroll
    VEC_SIZE = 16

    if args.eval == "eval" and args.execute:
        args.eval_parameters = get_eval_parameters(args)

    # Workaround to ensure that TVM backend is after MLIR backends,
    # otherwise the import of tvm breaks the MLIR python bindings
    args.backends = sorted(args.backends)


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
    logger.debug("Executing child command: %s", " ".join(cmd))
    proc = subprocess.run(
        args=cmd,
    )
    if proc.returncode != 0:
        print(
            f"ERROR: running subprocess: exit code: {proc.returncode}: {' '.join(cmd)}",
            file=sys.stderr,
        )
    raise SystemExit(proc.returncode)


def main():
    default_jobs = max(1, multiprocessing.cpu_count() // 2)
    default_unroll = 512
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
    parser.add_argument(
        "--max-unroll",
        type=int,
        default=default_unroll,
        help="max unroll in tiling strategies",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--output", type=str, default="results.csv", help="output csv file for search"
    )
    parser.add_argument(
        "--eval", type=str, choices=["eval"], default="eval", help="evaluation method"
    )
    parser.add_argument("--repeat", type=int, default=1, help="evaluation repeat")
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
        "--save-temps-dir", type=str, default="./save_temps_dir", help="save temps dir"
    )
    parser.add_argument(
        "--explore-dir", type=str, default=".", help="exploration results .so dir"
    )
    parser.add_argument(
        "--child",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="internal flag for marking child execution",
    )
    parser.add_argument(
        "--jobs", type=int, default=default_jobs, help="parallel compile jobs"
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do not execute, only compile",
    )
    parser.add_argument(
        "--mlir-prefix", type=str, help="MLIR install prefix, defaults to mlir package"
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="debug mode"
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        help="quiet optionnal output and progress bar",
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

    setup_args(args)

    optimize(args)


if __name__ == "__main__":
    main()
