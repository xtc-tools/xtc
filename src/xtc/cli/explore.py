#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
Explore tilings for some operators.

Refer to xtc.search.strategies.py for the available scheduling strategies.

Currently, depending on backend ans strategies, the combinations of
operator x strategies is limited.

Though most strategies are supported for all backends for matmult.

"""

import sys
import os
import argparse
from argparse import Namespace as NS
import logging
import itertools
from functools import partial
import csv
import random
import numpy as np
import numpy.typing
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing
import time
from pathlib import Path
from collections.abc import Sequence, Iterator, Mapping
from typing import Any, Type, TypeAlias, cast
from typing_extensions import override

from xtc.itf.back import Backend
from xtc.itf.graph import Graph
from xtc.itf.schd import Scheduler
from xtc.itf.comp import Module
from xtc.itf.search import Strategy, Sample

from xtc.search.strategies import Strategies

from xtc.utils.numpy import (
    np_init,
)
from xtc.utils.math import (
    mulall,
    factors_to_sizes,
    factors_enumeration,
)
from xtc.runtimes.types.ndarray import NDArray
import xtc.runtimes.host.runtime as runtime

logger = logging.getLogger(__name__)

NPSamples: TypeAlias = numpy.typing.NDArray[np.int64]
CallBacks: TypeAlias = Mapping[str, Any]


def xtc_matmul_graph(i: int, j: int, k: int, ftype: str, name: str = "matmul") -> Graph:
    import xtc.graphs.xtc.op as O

    dtype = DTYPES_MAP[ftype]
    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")
    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")
    return gb.graph


def xtc_relu_graph(i: int, ftype: str, name: str = "relu") -> Graph:
    import xtc.graphs.xtc.op as O

    dtype = DTYPES_MAP[ftype]
    inp = O.tensor((i,), dtype, name="I")
    with O.graph(name=name) as gb:
        O.relu(inp, threshold=0, name="O")
    return gb.graph


def tvm_matmul_impl(graph: Graph) -> tuple[Backend, str]:
    from xtc.backends.tvm import TVMBackend

    impl = TVMBackend(graph)
    return impl, "tvm"


def tvm_relu_impl(graph: Graph) -> tuple[Backend, str]:
    from xtc.backends.tvm import TVMBackend

    impl = TVMBackend(graph)
    return impl, "tvm"


def jir_matmul_impl(graph: Graph) -> tuple[Backend, str]:
    from xtc.backends.jir import JIRBackend

    impl = JIRBackend(graph)
    return impl, "jir"


def mlir_matmul_impl(graph: Graph) -> tuple[Backend, str]:
    from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend

    impl = MlirGraphBackend(
        graph,
        no_alias=True,
        always_vectorize=False,
    )
    return impl, "mlir"


def get_eval_parameters(args: NS):
    if args.huge_pages:
        NDArray.set_alloc_alignment(
            2 * 1024 * 1024
        )  # 2MB to catch Huge Pages if THB is one
    else:
        NDArray.set_alloc_alignment(256)  # default align to 256 bytes as DLPack
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
    nd_inputs = [NDArray(np_init(**spec)) for spec in inputs_spec]
    nd_outputs = [NDArray(np.empty(**spec)) for spec in outputs_spec]
    return (nd_inputs, nd_outputs)


def compile_one_all_backends(
    ident: str,
    graph: Graph,
    strategy: Strategy,
    in_x: Sample,
    args: NS,
    callbacks: CallBacks = {},
):
    compiled = []
    for backend in args.backends:
        task_ident = f"{args.operator}_{backend}_{ident}"
        compiled.append(
            compile_one(
                task_ident, backend, graph, strategy, in_x, args, callbacks=callbacks
            )
        )
    return compiled


def compile_one(
    ident: str,
    backend: str,
    graph: Graph,
    strategy: Strategy,
    in_x: Sample,
    args: NS,
    callbacks: CallBacks = {},
    dump_file: str | None = None,
):
    assert isinstance(in_x, list), f"X not a list: {in_x} ({type(in_x)})"
    logger.debug("Compile: %s: %s: %s...", ident, backend, in_x)
    implementer = OPERATORS[args.operator]["backends"][backend]["implementer"]
    impl, backend_name = implementer(graph)
    assert backend_name == backend
    scheduler = impl.get_scheduler()
    node_scheduler = scheduler  # by default the output node is scheduled
    strategy.generate(node_scheduler, in_x)
    schedule = scheduler.schedule()
    logger.debug("  Schedule done: %s: %s.", ident, schedule)
    if dump_file is None:
        dump_file = f"{args.explore_dir}/payload_{ident}"
    compile_args = dict(
        shared_lib=True,
        dump_file=dump_file,
        no_entry=True,
        bare_ptr=args.bare_ptr,
        debug=args.debug_compile,
    )
    if args.dump:
        compile_args.update(
            dict(
                print_source_ir=True,
                print_transformed_ir=True,
                print_lowered_ir=True,
                print_assembly=True,
                color=False,
                to_disassemble=impl.payload_name,
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
    compiler = impl.get_compiler(**compile_args)
    module = compiler.compile(schedule=schedule)
    logger.debug("  Compile done: %s: %s.", ident, in_x)
    return (ident, backend, module, dump_file, in_x)


def load_and_evaluate_sample(
    ident: str,
    backend: str,
    module: Module,
    in_x: Sample,
    args: NS,
    callbacks: CallBacks = {},
):
    logger.debug("Evaluate: %s: %s...", ident, in_x)
    evaluator_args = dict(
        repeat=args.repeat,
        number=args.number,
        min_repeat_ms=args.min_repeat_ms,
        validate=args.validate,
        parameters=args.eval_parameters,
    )
    reference_impl = OPERATORS[args.operator]["reference_impl"]
    if reference_impl is not None:
        evaluator_args.update(
            dict(
                reference_impl=reference_impl,
            )
        )
    payload_lib = module.file_name
    evaluator = module.get_evaluator(**evaluator_args)
    evaluate = evaluator.evaluate
    results, code, error_msg = evaluate()
    if code == 0:
        time = min(results)
        logger.debug("  Evaluated: %s: %s: time: %.2f msecs", ident, in_x, time * 1000)
    else:
        time = 0
        logger.error("Error evaluating: %s: %s: %d: %s", ident, in_x, code, error_msg)

    if not args.save_temps:
        Path(payload_lib).unlink()

    result = (in_x, code, time, backend)
    if callbacks and "result" in callbacks:
        callbacks["result"](result)
    return result


class SearchProgress:
    def search_start(self, ntasks: int):
        pass

    def compile_batch_start(self):
        pass

    def compile_job_start(self):
        pass

    def compile_job_end(self):
        pass

    def compile_batch_end(self):
        pass

    def execute_batch_start(self):
        pass

    def execute_job_start(self):
        pass

    def execute_job_end(self):
        pass

    def execute_batch_end(self):
        pass

    def search_end(self):
        pass


class SearchProgressTQDM(SearchProgress):
    def __init__(
        self,
        ncomp_per_job: int = 1,
        nexec_per_job: int = 1,
        quiet: bool = False,
        prefix: str = "",
        position: int = 0,
    ):
        self.ncomp_per_job = ncomp_per_job
        self.nexec_per_job = nexec_per_job
        self.quiet = quiet
        self.prefix = prefix
        self.position = position
        self.allbar: Any = None
        self.compbar: Any = None
        self.evalbar: Any = None
        self.ntasks = None

    @override
    def search_start(self, ntasks: int):
        self.ntasks = ntasks
        tqdm_args = dict(
            total=self.ntasks,
            miniters=0,
            mininterval=0,
            smoothing=0,
            disable=self.quiet,
        )
        self.allbar = tqdm(
            desc=f"{self.prefix}evaluate".strip(),
            colour="red",
            position=self.position + 0,
            **tqdm_args,
        )  # type: ignore
        self.compbar = tqdm(
            desc=f"{self.prefix} compile".strip(),
            colour="blue",
            position=self.position + 1,
            **tqdm_args,
        )  # type: ignore
        if self.nexec_per_job > 0:
            self.evalbar = tqdm(
                desc=f"{self.prefix} execute".strip(),
                colour="green",
                position=self.position + 2,
                **tqdm_args,
            )  # type: ignore

    @override
    def compile_batch_start(self):
        self.compbar.unpause()

    @override
    def compile_job_start(self):
        pass

    @override
    def compile_job_end(self):
        self.compbar.update(self.ncomp_per_job)
        if self.nexec_per_job == 0:
            self.allbar.update(self.ncomp_per_job)

    @override
    def compile_batch_end(self):
        self.compbar.update(0)
        self.allbar.update(0)

    @override
    def execute_batch_start(self):
        if self.nexec_per_job > 0:
            self.evalbar.unpause()

    @override
    def execute_job_start(self):
        pass

    @override
    def execute_job_end(self):
        if self.nexec_per_job > 0:
            self.evalbar.update(self.nexec_per_job)
            self.allbar.update(self.nexec_per_job)

    @override
    def execute_batch_end(self):
        pass

    @override
    def search_end(self):
        self.compbar.unpause()
        if self.nexec_per_job > 0:
            self.evalbar.unpause()
        self.allbar.close()
        self.compbar.close()
        if self.nexec_per_job > 0:
            self.evalbar.close()


def evaluate_all_parallel(
    strategy: Strategy,
    all_in_x: NPSamples,
    graph: Graph,
    args: NS,
    callbacks: CallBacks,
):
    jobs = args.jobs

    def do_compile(idx: int, in_x: Sample):
        return idx, compile_one_all_backends(
            ident=f"{idx:04}",
            graph=graph,
            strategy=strategy,
            in_x=in_x,
            args=args,
            callbacks=callbacks,
        )

    ntasks = len(all_in_x) * len(args.backends)
    search_callback = cast(
        SearchProgress,
        callbacks["search"] if "search" in callbacks else SearchProgress(),
    )
    search_callback.search_start(ntasks)
    try:
        for job_idx, job_in_x in enumerate(
            np.array_split(all_in_x, np.ceil(len(all_in_x) / jobs), axis=0)
        ):
            search_callback.compile_batch_start()
            job_compiled = []
            if jobs == 1:
                search_callback.compile_job_start()
                job_compiled.append(
                    do_compile(
                        idx=job_idx,
                        in_x=job_in_x[0].tolist(),
                    )
                )
                search_callback.compile_job_end()
            else:

                def future_callback(future: Future):
                    job_compiled.append(future.result())
                    search_callback.compile_job_end()

                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    futures = []
                    for idx, in_x in enumerate(job_in_x):
                        search_callback.compile_job_start()
                        future = executor.submit(
                            do_compile,
                            idx=job_idx * jobs + idx,
                            in_x=in_x.tolist(),
                        )
                        future.add_done_callback(future_callback)
                        futures.append(future)
                if len(job_compiled) < len(job_in_x):
                    raise RuntimeError("compilation error in some compile job")
            search_callback.compile_batch_end()
            if args.execute:
                compiled_results = [
                    x[1] for x in sorted(job_compiled, key=lambda x: x[0])
                ]
                search_callback.execute_batch_start()
                for compiled_list in compiled_results:
                    for compiled in compiled_list:
                        search_callback.execute_job_start()
                        ident, backend, module, dump_file, in_x = compiled
                        load_and_evaluate_sample(
                            ident, backend, module, in_x, args, callbacks=callbacks
                        )
                        search_callback.execute_job_end()
                search_callback.execute_batch_end()
    finally:
        search_callback.search_end()


def evaluate_generate(strategy: Strategy, graph: Graph, args: NS, callbacks: CallBacks):
    assert args.search in ["exhaustive", "random"]
    all_in_x = strategy.exhaustive()
    if args.search == "random":
        # TODO: for now randomize from exhaustive
        all_in_x = np.array(list(all_in_x))
        trials = (
            args.trials
            if args.trials and len(all_in_x) > args.trials
            else len(all_in_x)
        )
        idxs = np.random.choice(np.arange(len(all_in_x)), size=trials, replace=False)
        all_in_x = all_in_x[idxs]
    elif args.trials:
        all_in_x = np.array(list(itertools.islice(all_in_x, args.trials)))
    else:
        all_in_x = np.array(list(all_in_x))
    evaluate_all_parallel(strategy, all_in_x, graph, args, callbacks)


def evaluate_data(
    strategy: Strategy, X: NPSamples, graph: Graph, args: NS, callbacks: CallBacks
):
    size = len(X)
    logger.debug(f"Search space size: {size}")
    evaluate_all_parallel(strategy, X, graph, args, callbacks)


def evaluate_sample(
    strategy: Strategy, in_x: Sample, graph: Graph, args: NS, callbacks: CallBacks
):
    evaluate_all_parallel(strategy, np.array([in_x]), graph, args, callbacks)


def read_input(fname: str, args: NS) -> NPSamples:
    X = []
    with open(fname, newline="") as infile:
        reader = csv.reader(infile, delimiter=";")
        X_idx = 0
        for idx, row in enumerate(reader):
            if idx == 0:
                X_idx = row.index("X")
                continue
            X.append(eval(row[X_idx], {}, {}))
    return np.array(X)


def peak_time(args: NS) -> float:
    if not args.execute:
        return 0
    dtype = DTYPES_MAP[args.dtype]
    flops = runtime.evaluate_flops(dtype)
    assert flops != 0, f"unable to evaluate machine flops for type {dtype}"
    flop = mulall(args.dims)
    time = flop / flops / args.threads
    return time


class CSVCallback:
    def __init__(self, fname: str, peak_time: float) -> None:
        self._fname = fname
        self._peak_time = peak_time
        self._outf = open(fname, "w", newline="")
        self._header = ("X", "time", "peak", "backend")
        self._writer = csv.writer(self._outf, delimiter=";")
        self._write_header()
        self._results = []
        self._rows = []

    def _write_header(self) -> None:
        self._writer.writerow(self._header)
        self._outf.flush()

    def _write_row(self, row: Sequence) -> None:
        self._rows.append(row)
        self._writer.writerow(row)
        self._outf.flush()

    def _write_result(self, result: Sequence) -> None:
        self._results.append(result)
        x, error, time, backend = result
        if error != 0:
            logger.debug(f"Skip recording error for: {backend}: {x}")
            return
        peak = self._peak_time / time
        row = [x, time, peak, backend]
        logger.debug(f"Record row: {row}")
        self._write_row(row)

    def __call__(self, result: Sequence) -> None:
        self._write_result(result)

    def __del__(self) -> None:
        self._outf.close()


def search_some(strategy: Strategy, graph: Graph, args: NS):
    # Search depends on search strategy
    ncomp_per_job = len(args.backends)
    nexec_per_job = 1 if args.execute else 0
    search_callback = SearchProgressTQDM(
        ncomp_per_job,
        nexec_per_job,
        args.quiet,
        args.operator,
    )
    ptime = peak_time(args)
    result_callback = CSVCallback(args.output, ptime)
    callbacks = {
        "result": result_callback,
        "search": search_callback,
    }
    if args.search in ["exhaustive", "random"]:
        evaluate_generate(
            strategy,
            graph,
            args,
            callbacks=callbacks,
        )
    elif args.search == "data":
        assert args.data is not None
        X = read_input(args.data, args)
        evaluate_data(
            strategy,
            X,
            graph,
            args,
            callbacks=callbacks,
        )


def optimize(args: NS):
    dims = args.dims
    dtype = args.dtype
    op_args = (*dims, dtype)
    graph = OPERATORS[args.operator]["operation"](*op_args, name=args.func_name)
    strategy = get_strategy(graph, args)
    if args.test or args.opt_level in [0, 1, 2, 3]:
        schedule = args.test
        if not schedule:
            schedule = strategy.default_schedule(args.opt_level)
        ncomp_per_job = len(args.backends)
        nexec_per_job = 1 if args.execute else 0
        search_callback = SearchProgressTQDM(
            ncomp_per_job,
            nexec_per_job,
            args.quiet,
            args.operator,
        )
        ptime = peak_time(args)
        result_callback = CSVCallback(args.output, ptime)
        callbacks = {
            "result": result_callback,
            "search": search_callback,
        }
        evaluate_sample(strategy, schedule, graph, args, callbacks=callbacks)
        for row in result_callback._rows:
            in_x, time, peak, backend = row
            tqdm.write(
                f"Schedule: {backend}: {in_x}: time: {time * 1000:.2f} msecs, peak perf: {peak * 100:.2f}%"
            )
    else:
        search_some(strategy, graph, args)


def get_strategy(graph: Graph, args: NS) -> Strategy:
    def strat_name(name: str) -> str:
        alias = STRATEGIES_ALIASES.get(name)
        if alias is None:
            return name
        return strat_name(alias)

    name = strat_name(args.strategy)
    cls = Strategies.from_name(name)
    return cls(
        graph, threads=args.threads, max_unroll=args.max_unroll, vec_size=VEC_SIZE
    )


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
        "reference_impl": None,  # defaults to graph evaluation
        "operation": xtc_matmul_graph,
        "backends": {
            "mlir": {
                "implementer": mlir_matmul_impl,
            },
            "tvm": {
                "implementer": tvm_matmul_impl,
            },
            "jir": {
                "implementer": jir_matmul_impl,
            },
        },
        "default_strategy": "tile_t1",
    },
    "relu": {
        "dims": ["i"],
        "default_dims": [512 * 1024],
        "default_type": "f32",
        "inputs": [["i"]],
        "outputs": [["i"]],
        "reference_impl": None,  # defaults to graph evaluation
        "operation": xtc_relu_graph,
        "backends": {
            "tvm": {
                "implementer": tvm_relu_impl,
            },
        },
        "default_strategy": "tile_t1",
    },
}

STRATEGIES_ALIASES = {
    # legacy: same as t1 for 1 dimensional kernels
    "tile1d": "tile_t1",
    # legacy: same as t1 for matmul kernel
    "tile3d": "tile_t1",
    # legacy: tile4d* for matmul is same as tile_p1*
    "tile4d": "tile_p1",
    "tile4dv": "tile_p1_v",
    # legacy: tile7d* for matmul is same as tile_pprprp*
    "tile7d": "tile_pprprp",
    "tile7dv": "tile_pprprp_v",
    "tile7dvr": "tile_pprprp_vr",
    # legacy: tile8d* for matmul is same as tile_ppwrprp*
    "tile8d": "tile_ppwrprp",
    "tile8dv": "tile_ppwrprp_v",
    "tile8dvr": "tile_ppwrprp_vr",
}


def setup_args(args: NS):
    global THREADS
    global MAX_UNROLL
    global VEC_SIZE
    THREADS = args.threads
    MAX_UNROLL = args.max_unroll
    VEC_SIZE = 16

    if "tvm" in args.backends:
        os.environ["TVM_NUM_THREADS"] = str(args.threads)

    if args.eval == "eval" and args.execute:
        args.eval_parameters = get_eval_parameters(args)

    # Workaround to ensure that TVM backend is after MLIR backends,
    # otherwise the import of tvm breaks the MLIR python bindings
    args.backends = sorted(args.backends)


def launch_child(argv: Sequence[str], args: NS):
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
        logger.debug(
            f"ERROR: running subprocess: exit code: %s, command: %s",
            proc.returncode,
            " ".join(cmd),
        )
    raise SystemExit(proc.returncode)


def main():
    default_op = "matmul"
    default_dims = OPERATORS[default_op]["default_dims"]
    default_dtype = OPERATORS[default_op]["default_type"]
    default_jobs = max(1, multiprocessing.cpu_count() // 2)
    default_unroll = 512
    choice_strategies = list(Strategies.names()) + list(STRATEGIES_ALIASES.keys())
    parser = argparse.ArgumentParser(
        description="Autotune Matmult",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operator",
        type=str,
        choices=list(OPERATORS.keys()),
        default=default_op,
        help="operator to optimize",
    )
    parser.add_argument(
        "--func-name", type=str, help="function name to generate, default to operator"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=choice_strategies,
        help="tile strategy to use, default to operator's default",
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
        choices=["mlir", "tvm", "jir"],
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
        default=default_dims,
        help="dimensions, default to operators's default",
    )
    parser.add_argument(
        "--huge-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="alloc at huge page boundaries",
    )
    parser.add_argument(
        "--test", nargs="+", type=int, default=[], help="test this input only"
    )
    parser.add_argument(
        "--opt-level", type=int, default=4, help="opt level, 0-3 one-shot, 4 search"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=default_dtype,
        choices=list(DTYPES_MAP.keys()),
        help="data type, default to operator's default",
    )
    parser.add_argument("--trials", type=int, default=100, help="num trials")
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
        default=True,
        help="internal flag for marking child execution (obsolete)",
    )
    parser.add_argument(
        "--bare-ptr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use bare pointer interface (for TVM backend)",
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
        "--debug-compile",
        action=argparse.BooleanOptionalAction,
        help="debug compile commands",
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

    if not args.func_name:
        args.func_name = args.operator
    if not args.strategy:
        args.strategy = OPERATORS[args.operator]["default_strategy"]
    if not args.dims:
        args.dims = OPERATORS[args.operator]["default_dims"]
    if not args.dtype:
        args.dtype = OPERATORS[args.operator]["default_type"]

    for backend in args.backends:
        assert backend in OPERATORS[args.operator]["backends"], (
            f"backend {backend} not available for operator {args.operator}"
        )

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)

    setup_args(args)

    optimize(args)


if __name__ == "__main__":
    main()
