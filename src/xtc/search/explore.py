#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
Explore tilings for some operators.

Refer to xtc.search.strategies.py for the available scheduling strategies.

Currently, depending on backend and strategies, the combinations of
operator x strategies is limited.

Though most strategies are supported for all backends for matmult.

"""

from dataclasses import dataclass, field
import sys
import os
from argparse import Namespace as NS
import logging
import itertools
import csv
import json
import random
from datetime import datetime, timezone
import numpy as np
import numpy.typing
import subprocess
import multiprocessing
from pathlib import Path
from collections.abc import Sequence, Mapping
from typing import Any, TypeAlias, cast
from importlib import import_module

from xtc.itf.back import Backend
from xtc.itf.graph import Graph
from xtc.itf.comp import Module
from xtc.itf.search import Strategy, Sample

from xtc.graphs.xtc.graph import XTCGraph

from xtc.search.strategies import Strategies

from xtc.utils.numpy import (
    np_init,
)
from xtc.runtimes.types.ndarray import NDArray
from xtc.runtimes.host import HostRuntime
from xtc.artifacts import get_operation, list_operations
from xtc.search.optimizers import Optimizers

from .progress import SearchProgress, SearchProgressTQDM, SearchProgressMO
from .callback import ResultCallBack, CSVCallback, DBCallback, MemoryCallback
from .pipeline import CompileExecutePipeline

logger = logging.getLogger(__name__)

NPSamples: TypeAlias = numpy.typing.NDArray[np.int64]
CallBacks: TypeAlias = Mapping[str, Any]

__all__ = ["ExplorationConfig", "Exploration"]


@dataclass
class ExplorationConfig:
    operator: str | None = "matmul"
    graph_file: str | None = None
    op_name: str | None = None
    func_name: str | None = None
    strategy: str = "tile_oo"
    search: str = "random"
    backends: list[str] = field(default_factory=lambda: ["mlir"])
    optimizer: str = "random-forest-default"
    data: str | None = None
    dims: list[int] | None = None
    huge_pages: bool = True
    test: list[int] = field(default_factory=list)
    opt_level: int = 4
    dtype: str = "float32"
    trials: int = 100
    threads: int = 1
    max_unroll: int | None = None
    seed: int = 0
    output: str | None = "results.csv"
    db_file: str | None = None
    resume: bool = False
    append: bool = False
    eval: str = "eval"
    repeat: int = 1
    number: int = 1
    min_repeat_ms: int = 100
    validate: bool = False
    save_temps: bool = False
    save_temps_dir: str = "./save_temps_dir"
    explore_dir: str = "."
    optimizer_config: str = ""
    bare_ptr: bool = False
    jobs: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() // 2))
    execute: bool = True
    peak_flops: float | None = None
    mlir_prefix: str | None = None
    batch: int = 1
    debug_compile: bool = False
    quiet: bool = False
    dump: bool = False
    eval_parameters: Any = None
    memory_callback: Any = None
    csv_callback: Any = None
    db_callback: Any = None
    results: list[Sequence] = field(default_factory=list)
    descript: str | None = None
    use_tensors: bool = False
    tir_schedule: bool = False
    progress_cls: str = "tqdm"

    def __post_init__(self):
        if self.graph_file is not None:
            self.operator = None
            self.func_name = None

        # Workaround to ensure that TVM backend is after MLIR backends,
        # otherwise the import of tvm breaks the MLIR python bindings
        self.backends = sorted(self.backends)

        if self.operator:
            if not self.func_name:
                self.func_name = self.operator
            for backend in self.backends:
                assert backend in cast(list, OPERATORS[self.operator]["backends"]), (
                    f"backend {backend} not available for operator {self.operator}"
                )

    @staticmethod
    def from_args(
        args: NS | None = None,
        **overrides: Any,
    ):
        config = ExplorationConfig()
        if args is not None:
            for key, value in vars(args).items():
                if hasattr(config, key):
                    setattr(config, key, value)
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"unknown exploration configuration option: {key}")
            setattr(config, key, value)
        config.__post_init__()
        return config


OPERATORS = {
    "matmul": {
        "dims": ["i", "j", "k"],
        "default_dims": [512, 1024, 128],
        "default_type": "float32",
        "inputs": [["i", "k"], ["k", "j"]],
        "outputs": [["i", "j"]],
        "reference_impl": None,  # defaults to graph evaluation
        "operation": "xtc_matmul_graph",
        "backends": {
            "mlir": {},
            "tvm": {},
            "jir": {},
        },
        "default_strategy": "tile_oo",
    },
    "conv2d": {
        "dims": ["n", "h", "w", "f", "r", "s", "c", "SH", "SW"],
        "default_dims": [1, 112, 112, 64, 7, 7, 3, 2, 2],
        "default_type": "float32",
        "inputs": [
            ["n", "h * SH + r - 1", "w * SW + s - 1", "c"],
            ["r", "s", "c", "f"],
        ],
        "outputs": [["n", "h", "w", "f"]],
        "reference_impl": None,  # defaults to graph evaluation
        "operation": "xtc_conv2d_graph",
        "backends": {
            "mlir": {},
            "tvm": {},
        },
        "default_strategy": "tile_oo",
    },
    "relu": {
        "dims": ["i"],
        "default_dims": [512 * 1024],
        "default_type": "float32",
        "inputs": [["i"]],
        "outputs": [["i"]],
        "reference_impl": None,  # defaults to graph evaluation
        "operation": "xtc_relu_graph",
        "backends": {
            "tvm": {},
        },
        "default_strategy": "tile_oo",
    },
}


class Exploration:
    """Callable exploration API.

    Instantiate with an :class:`ExplorationConfig` (or compatible object), call it,
    then inspect the structured results through the :attr:`results` field.
    """

    def __init__(self, config: ExplorationConfig | None = None):
        self.config = config if config is not None else ExplorationConfig()
        self.results: list[Sequence] = []

    @staticmethod
    def xtc_load_graph(graph_file: str) -> Graph:
        import xtc.graphs.xtc.op as O

        with O.graph() as gb:
            gb.load(graph_file)
        return gb.graph

    @staticmethod
    def xtc_matmul_graph(
        i: int, j: int, k: int, dtype: str, name: str = "matmul"
    ) -> Graph:
        import xtc.graphs.xtc.op as O

        a = O.tensor((i, k), dtype, name="A")
        b = O.tensor((k, j), dtype, name="B")
        with O.graph(name=name) as gb:
            O.matmul(a, b, name="C")
        return gb.graph

    @staticmethod
    def xtc_relu_graph(i: int, dtype: str, name: str = "relu") -> Graph:
        import xtc.graphs.xtc.op as O

        inp = O.tensor((i,), dtype, name="I")
        with O.graph(name=name) as gb:
            O.relu(inp, threshold=0, name="O")
        return gb.graph

    @staticmethod
    def xtc_conv2d_graph(
        n: int,
        h: int,
        w: int,
        f: int,
        r: int,
        s: int,
        c: int,
        SH: int,
        SW: int,
        dtype: str,
        name: str = "conv2d",
    ) -> Graph:
        import xtc.graphs.xtc.op as O

        a = O.tensor((n, h * SH + r - 1, w * SW + s - 1, c), dtype, name="A")
        b = O.tensor((r, s, c, f), dtype, name="B")
        with O.graph(name=name) as gb:
            O.conv2d(a, b, stride=(SH, SW), name="O")
        return gb.graph

    @staticmethod
    def graph_implementer(
        graph: Graph, backend: str, **kwargs: Any
    ) -> tuple[Backend, str]:
        module = import_module(f"xtc.backends.{backend}")
        impl = module.Backend(graph, **kwargs)
        return impl, backend

    def get_dims(self) -> dict[str, int]:
        args = self.config
        if not args.operator:
            return {}
        dims_names = cast(list[str], OPERATORS[args.operator]["dims"])
        if args.op_name:
            dims: list[int] = Exploration.get_operation_dims(
                args.operator, args.op_name
            )
        elif args.dims is None:
            dims = cast(list[int], OPERATORS[args.operator]["default_dims"])
        else:
            dims = args.dims
        dims_map = {k: v for k, v in zip(dims_names, dims)}
        return dims_map

    def get_dtype(self) -> str:
        args = self.config
        if not args.operator:
            return ""
        return args.dtype

    def init_eval_parameters(self):
        args = self.config
        assert args.operator is not None
        if args.huge_pages:
            NDArray.set_alloc_alignment(2 * 1024 * 1024)
        else:
            NDArray.set_alloc_alignment(256)
        dims_map = self.get_dims()
        dtype = self.get_dtype()
        inputs = cast(list[list[str]], OPERATORS[args.operator]["inputs"])
        outputs = cast(list[list[str]], OPERATORS[args.operator]["outputs"])
        inputs_spec = [
            {"shape": tuple([eval(x, {}, dims_map) for x in shape]), "dtype": dtype}
            for shape in inputs
        ]
        outputs_spec = [
            {"shape": tuple([eval(x, {}, dims_map) for x in shape]), "dtype": dtype}
            for shape in outputs
        ]
        nd_inputs = [NDArray(np_init(**spec)) for spec in inputs_spec]  # type: ignore
        nd_outputs = [NDArray(np.empty(**spec)) for spec in outputs_spec]  # type: ignore
        return (nd_inputs, nd_outputs)

    def compile_one_all_backends(
        self,
        ident: str,
        graph: Graph,
        strategy: Strategy,
        in_x: Sample,
        callbacks: CallBacks = {},
    ):
        args = self.config
        compiled = []
        for backend in args.backends:
            task_ident = f"{graph.name}_{backend}_{ident}"
            compiled.append(
                self.compile_one(
                    task_ident,
                    backend,
                    graph,
                    strategy,
                    in_x,
                    callbacks=callbacks,
                )
            )
        return compiled

    def compile_one(
        self,
        ident: str,
        backend: str,
        graph: Graph,
        strategy: Strategy,
        in_x: Sample,
        callbacks: CallBacks = {},
        dump_file: str | None = None,
    ):
        args = self.config
        assert isinstance(in_x, list), f"X not a list: {in_x} ({type(in_x)})"
        logger.debug("Compile: %s: %s: %s...", ident, backend, in_x)
        kwargs = {}
        if backend == "tvm":
            kwargs.update({"tir_schedule": args.tir_schedule})
        if backend == "mlir":
            kwargs.update({"use_tensor_dialect": args.use_tensors})
        impl, backend_name = self.graph_implementer(
            graph,
            backend,
            **kwargs,
        )
        assert backend_name == backend
        scheduler = impl.get_scheduler()
        node_scheduler = scheduler
        strategy.generate(node_scheduler, in_x)
        schedule = scheduler.schedule()
        logger.debug("  Schedule done: %s: %s.", ident, schedule)
        if dump_file is None:
            dump_file = f"{args.explore_dir}/payload_{ident}"
        compile_args = dict(
            shared_lib=True,
            dump_file=dump_file,
            bare_ptr=args.bare_ptr,
            debug=args.debug_compile,
        )
        if args.dump:
            compile_args.update(
                dict(
                    print_source_ir=True,
                    print_transformed_ir=True,
                    print_bufferization_ir=args.use_tensors,
                    print_lowered_ir=True,
                    print_assembly=True,
                    color=False,
                    to_disassemble=impl.graph.name,
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
        self,
        ident: str,
        backend: str,
        module: Module,
        in_x: Sample,
        callbacks: CallBacks = {},
    ):
        args = self.config
        logger.debug("Evaluate: %s: %s...", ident, in_x)
        evaluator_args = dict(
            repeat=args.repeat,
            number=args.number,
            min_repeat_ms=args.min_repeat_ms,
            validate=args.validate,
            parameters=args.eval_parameters,
        )
        if args.operator:
            reference_impl = OPERATORS[args.operator]["reference_impl"]
            if reference_impl is not None:
                evaluator_args.update(dict(reference_impl=reference_impl))
        payload_lib = module.file_name
        evaluator = module.get_evaluator(**evaluator_args)
        results, code, error_msg = evaluator.evaluate()
        if code == 0:
            time = min(results)
            logger.debug(
                "  Evaluated: %s: %s: time: %.2f msecs", ident, in_x, time * 1000
            )
        else:
            time = 0
            logger.error(
                "Error evaluating: %s: %s: %d: %s", ident, in_x, code, error_msg
            )

        if not args.save_temps:
            Path(payload_lib).unlink()

        result = (in_x, code, time, backend)
        if callbacks and "result" in callbacks:
            for callback in callbacks["result"]:
                callback(result)
        return result

    def evaluate_all_parallel(
        self,
        strategy: Strategy,
        all_in_x: NPSamples,
        graph: Graph,
        callbacks: CallBacks = {},
    ):
        jobs = self.config.jobs
        ntasks = len(all_in_x)
        batch_ntasks = ntasks * len(self.config.backends)

        def compile_func(idx_sample: Any) -> Any:
            idx, in_x = idx_sample
            if idx % jobs == 0:
                search_callback.compile_batch_start()
            search_callback.compile_job_start()
            res = self.compile_one_all_backends(
                ident=f"{idx:04}",
                graph=graph,
                strategy=strategy,
                in_x=in_x,
                callbacks=callbacks,
            )
            search_callback.compile_job_end()
            if idx == ntasks - 1 or idx + jobs - 1 % jobs == 0:
                search_callback.compile_batch_end()
            return idx, res

        def execute_func(idx_comp_result: Any) -> Any:
            if not self.config.execute:
                return None
            idx, comp_result = idx_comp_result
            if idx % jobs == 0:
                search_callback.execute_batch_start()
            search_callback.execute_job_start()
            exec_results = []
            for compiled in comp_result:
                search_callback.execute_job_start()
                ident, backend, module, dump_file, in_x = compiled
                exec_results.append(
                    self.load_and_evaluate_sample(
                        ident,
                        backend,
                        module,
                        in_x,
                        callbacks=callbacks,
                    )
                )
                search_callback.execute_job_end()
            if idx == ntasks - 1 or idx + jobs - 1 % jobs == 0:
                search_callback.execute_batch_end()
            return exec_results

        pipeline = CompileExecutePipeline(
            compile_func,
            execute_func,
            jobs,
        )
        search_callback = cast(
            SearchProgress,
            callbacks["search"] if "search" in callbacks else SearchProgress(),
        )
        search_callback.batch_start(batch_ntasks)
        results = pipeline.run(enumerate(x.tolist() for x in all_in_x))
        search_callback.batch_end()
        if self.config.execute:
            exec_results = [
                x
                for x in itertools.chain(*[res.exec_result for res in results])
                if x is not None
            ]
        else:
            exec_results = []
        return exec_results

    def evaluate_iterative(
        self,
        strategy: Strategy,
        graph: Graph,
        callbacks: CallBacks = {},
        peak_time: float = 0,
    ):
        args = self.config
        optimizer = Optimizers.from_name(args.optimizer)
        opt = optimizer(strategy.sample, args.batch, args.seed, args.optimizer_config)
        all_results = []
        progress = callbacks["search"]
        progress.search_start(args.trials * len(args.backends))
        try:
            for step in range(0, args.trials, args.batch):
                size = min(args.batch, args.trials - step)
                in_x = opt.suggest()[:size]
                results = self.evaluate_all_parallel(
                    strategy, np.array(in_x), graph, callbacks=callbacks
                )
                all_results.extend(results)
                peaks = [peak_time / res[-2] for res in results]
                opt.observe(in_x, peaks)
        finally:
            progress.search_end()
        opt.finished()
        return all_results

    def evaluate_generate(
        self,
        strategy: Strategy,
        graph: Graph,
        callbacks: CallBacks = {},
    ):
        args = self.config
        assert args.search in ["exhaustive", "random"]
        if args.search == "random":
            assert args.trials > 0
            sampled_x = strategy.sample(args.trials, args.seed)
            all_in_x = np.array(list(sampled_x))
        else:
            all_ex_x = strategy.exhaustive()
            if args.trials:
                all_in_x = np.array(list(itertools.islice(all_ex_x, args.trials)))
            else:
                all_in_x = np.array(list(all_ex_x))
        progress = callbacks["search"]
        progress.search_start(len(all_in_x) * len(args.backends))
        try:
            res = self.evaluate_all_parallel(
                strategy, all_in_x, graph, callbacks=callbacks
            )
        finally:
            progress.search_end()
        return res

    def evaluate_data(
        self,
        strategy: Strategy,
        X: NPSamples,
        graph: Graph,
        callbacks: CallBacks = {},
    ):
        args = self.config
        size = len(X)
        logger.debug(f"Search space size: {size}")
        progress = callbacks["search"]
        progress.search_start(size * len(args.backends))
        try:
            res = self.evaluate_all_parallel(strategy, X, graph, callbacks=callbacks)
        finally:
            progress.search_end()
        return res

    def evaluate_sample(
        self,
        strategy: Strategy,
        in_x: Sample,
        graph: Graph,
        callbacks: CallBacks = {},
    ):
        args = self.config
        progress = callbacks["search"]
        progress.search_start(1 * len(args.backends))
        try:
            res = self.evaluate_all_parallel(
                strategy, np.array([in_x]), graph, callbacks=callbacks
            )
        finally:
            progress.search_end()
        return res

    def read_input(self, fname: str) -> NPSamples:
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

    def peak_time(self, graph: Graph) -> float:
        args = self.config
        if not args.execute:
            return 0
        assert args.peak_flops is not None
        ops_count = graph.ops_count()
        return ops_count / args.peak_flops / args.threads

    @staticmethod
    def _args_to_metadata(args: NS | ExplorationConfig) -> dict[str, Any]:
        metadata_args: dict[str, Any] = {}
        for key, value in vars(args).items():
            if key == "eval_parameters":
                continue
            if isinstance(value, Path):
                metadata_args[key] = str(value)
            elif (
                isinstance(value, (str, int, float, bool, list, dict)) or value is None
            ):
                metadata_args[key] = value
            else:
                metadata_args[key] = str(value)
        return metadata_args

    @staticmethod
    def _git_commit_hash() -> str | None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def write_run_manifest(
        self,
        strategy: Strategy,
        args: NS | ExplorationConfig | None = None,
    ) -> None:
        args = cast(NS, self.config if args is None else args)
        output_path = Path(args.output)
        metadata_path = Path(f"{args.output}.meta.json")
        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "output": str(output_path),
            "resume": args.resume,
            "append": args.append,
            "sampleNames": strategy.sample_names,
            "gitCommit": self._git_commit_hash(),
            "python": {"version": sys.version, "executable": sys.executable},
            "platform": {"platform": sys.platform, "cwd": str(Path.cwd())},
            "args": self._args_to_metadata(args),
        }
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(payload, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")

    def get_result_callbacks(
        self,
        strategy: Strategy,
        graph: Graph,
    ) -> list[ResultCallBack]:
        args = self.config
        callbacks: list[ResultCallBack] = []
        ptime = self.peak_time(graph)
        sample_names = strategy.sample_names
        args.memory_callback = MemoryCallback()
        callbacks.append(args.memory_callback)
        if args.output:
            args.csv_callback = CSVCallback(
                args.output,
                ptime,
                sample_names,
                resume=args.resume,
                append=args.append,
            )
            callbacks.append(args.csv_callback)
        if args.db_file:
            args.db_callback = DBCallback(
                args.db_file,
                "native",
                args.threads,
                self.get_strategy_name(args.strategy),
            )
            args.db_callback.set_graph(graph)
            callbacks.append(args.db_callback)
        return callbacks

    def search_some(
        self,
        strategy: Strategy,
        graph: Graph,
    ):
        args = self.config
        ncomp_per_job = len(args.backends)
        nexec_per_job = 1 if args.execute else 0
        search_callback = self.progress(
            ncomp_per_job,
            nexec_per_job,
            args.quiet,
            graph.name,
            progress_cls=args.progress_cls,
        )
        result_callbacks = self.get_result_callbacks(strategy, graph)
        callbacks = {"result": result_callbacks, "search": search_callback}
        if args.search == "iterative":
            ptime = self.peak_time(graph)
            return self.evaluate_iterative(
                strategy, graph, callbacks=callbacks, peak_time=ptime
            )
        if args.search in ["exhaustive", "random"]:
            return self.evaluate_generate(strategy, graph, callbacks=callbacks)
        if args.search == "data":
            assert args.data is not None
            X = self.read_input(args.data)
            return self.evaluate_data(strategy, X, graph, callbacks=callbacks)
        return []

    def initialize_context(self):
        args = self.config
        if "tvm" in args.backends:
            os.environ["TVM_NUM_THREADS"] = str(args.threads)

        if args.operator and args.eval == "eval" and args.execute:
            self.init_eval_parameters()
        if args.execute and args.peak_flops is None:
            # TODO: get dtype from graph
            args.peak_flops = HostRuntime.get().evaluate_flops(args.dtype)
            assert args.peak_flops != 0, (
                f"unable to evaluate machine flops for type {args.dtype}"
            )
            logger.debug(f"Estimated peak flops: %g", args.peak_flops)

    @property
    def progress(self):
        cls = self.config.progress_cls
        if cls == "tqdm":
            return SearchProgressTQDM
        elif cls == "mo":
            return SearchProgressMO
        assert False, f"unknown progress class: {cls}"

    def optimize(self):
        args = self.config
        self.initialize_context()
        if args.operator:
            op_args = (
                *self.get_dims().values(),
                self.get_dtype(),
            )
            graph = getattr(self, cast(str, OPERATORS[args.operator]["operation"]))(
                *op_args, name=args.func_name
            )
        else:
            assert args.graph_file is not None
            graph = self.xtc_load_graph(args.graph_file)
        strategy = self.get_strategy(graph, args)
        self.write_run_manifest(strategy, args)
        if args.save_temps:
            cast(XTCGraph, graph).dump(
                Path(args.save_temps_dir) / f"{graph.name}.graph.yaml"
            )
        if args.test or args.opt_level in [0, 1, 2, 3]:
            schedule = args.test
            if not schedule:
                schedule = strategy.default_schedule(args.opt_level)
            ncomp_per_job = len(args.backends)
            nexec_per_job = 1 if args.execute else 0
            search_callback = self.progress(
                ncomp_per_job,
                nexec_per_job,
                args.quiet,
                graph.name,
            )
            result_callbacks = self.get_result_callbacks(strategy, graph)
            callbacks = {"result": result_callbacks, "search": search_callback}
            results = self.evaluate_sample(
                strategy, schedule, graph, callbacks=callbacks
            )
        else:
            results = self.search_some(strategy, graph)
        args.results = list(results or [])
        csv_callback = getattr(args, "csv_callback", None)
        if not args.quiet and csv_callback is not None and len(csv_callback._rows) > 0:
            ordered = sorted(csv_callback._rows, key=lambda x: x[-3])
            in_x, time, peak, backend = ordered[0][-4:]
            print(
                f"Schedule: {backend}: {in_x}: time: {time * 1000:.2f} msecs, peak perf: {peak * 100:.2f}%"
            )
        return args.results

    @staticmethod
    def get_strategy_name(strategy: str) -> str:
        return Strategies.resolve_name(strategy)

    def get_strategy(
        self, graph: Graph, args: NS | ExplorationConfig | None = None
    ) -> Strategy:
        args = cast(NS, self.config if args is None else args)
        if args.descript:
            from xtc.search.strategies import Strategy_Descript_Explore

            with open(args.descript, "r") as f:
                spec = f.read()
                return Strategy_Descript_Explore(graph=graph, spec=spec)
        strat_name = args.strategy
        strat_args = strat_name.split(":")
        name = self.get_strategy_name(strat_args[0])
        options = dict(
            threads=args.threads,
            **(dict(max_unroll=args.max_unroll) if args.max_unroll is not None else {}),
        )
        return Strategies.create(name, graph, *strat_args[1:], **options)

    @staticmethod
    def get_operation_dims(operator: str, name: str) -> list[int]:
        op = get_operation(operator, name)
        dims = [*op["dims"].values(), *op["params"].values()]
        return dims

    @staticmethod
    def list_operations_dims(operator: str):
        ops = list_operations(operator)
        for _, name in ops:
            op = get_operation(operator, name)
            print(f"{name}: {op['dims']}, {op['params']}")

    @staticmethod
    def list_operators():
        for name in OPERATORS:
            print(f"{name}")

    @staticmethod
    def list_strategies():
        for name in Strategies.names(include_aliases=True):
            print(f"{name}")

    @staticmethod
    def list_optimizers():
        for name in Optimizers.names():
            print(f"{name}")

    def run(self) -> list[Sequence]:
        args = self.config
        if args.seed >= 0:
            np.random.seed(args.seed)
            random.seed(args.seed)
        self.results = self.optimize()
        return self.results

    def __call__(self) -> list[Sequence]:
        return self.run()
