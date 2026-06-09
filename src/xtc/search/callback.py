#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
from collections.abc import Sequence
from typing import Any
import json
import csv
from pathlib import Path
import logging
import os

from xtc.itf.graph import Graph
from xtc.itf.search import Sample
from xtc.cli.query_results import ResultsDB

logger = logging.getLogger(__name__)


class ResultCallBack(ABC):
    @abstractmethod
    def __call__(self, result: Sequence) -> None: ...


class DBCallback(ResultCallBack):
    def __init__(
        self,
        dbfile: str,
        target: str,
        threads: int,
        strategy: str,
    ) -> None:
        self._dbfile = dbfile
        self._target = target
        self._threads = threads
        self._version = ResultsDB.get_version()
        self._platform = ResultsDB.get_native_platform()
        self._operator: list[Any] | None = None
        self._strategy = ResultsDB.get_strategy(strategy)

    def set_graph(self, graph: Graph):
        # assert len(graph.nodes) == 1, f"Only support recording of single node graph"
        # self._operator = ["xtc.operator", *signature]
        self._operator = ResultsDB.get_operator(graph)

    def _write_result(self, result: Sequence) -> None:
        x, code, time, backend = result
        if code != 0:
            time = 0
        compiler = ResultsDB.get_compiler(self._target, self._threads, backend)
        log = dict(
            version=self._version,
            platform=self._platform,
            compiler=compiler,
            operator=self._operator,
            strategy=self._strategy,
            schedule=list(x),
            results=[int(code), [float(time)]],
        )
        log_json = json.dumps(log)
        with open(self._dbfile, "a") as outf:
            print(log_json, flush=True, file=outf)

    @override
    def __call__(self, result: Sequence) -> None:
        self._write_result(result)


class MemoryCallback(ResultCallBack):
    def __init__(self) -> None:
        self.results: list[Sequence] = []

    @override
    def __call__(self, result: Sequence) -> None:
        self.results.append(result)


class CSVCallback(ResultCallBack):
    def __init__(
        self,
        fname: str,
        peak_time: float,
        sample_names: list[str],
        *,
        resume: bool = False,
        append: bool = False,
    ) -> None:
        self._fname = fname
        self._peak_time = peak_time
        self._sample_names = sample_names
        self._header = sample_names + ["X", "time", "peak", "backend"]
        self._results: list[Sequence] = []
        self._rows: list[Sequence] = []
        self._seen_keys: set[tuple[str, tuple[int, ...]]] = set()
        self._resume = resume
        self._append = append

        out_path = Path(fname)
        has_existing_file = out_path.exists() and out_path.stat().st_size > 0
        if resume:
            self._load_existing_rows()
            mode = "a"
        elif append:
            mode = "a"
        else:
            mode = "w"

        self._outf = open(fname, mode, newline="")
        self._writer = csv.writer(self._outf, delimiter=",")

        should_write_header = (not has_existing_file) or (mode == "w")
        if should_write_header:
            self._write_header()

    def _load_existing_rows(self) -> None:
        in_path = Path(self._fname)
        if not in_path.exists() or in_path.stat().st_size == 0:
            return
        with open(in_path, newline="") as infile:
            reader = csv.DictReader(infile, delimiter=",")
            for row in reader:
                backend = row.get("backend")
                if backend is None:
                    continue
                try:
                    sample = tuple(int(row[name]) for name in self._sample_names)
                except (TypeError, ValueError, KeyError):
                    continue
                self._seen_keys.add((backend, sample))

    def _sample_key(self, x: Sample, backend: str) -> tuple[str, tuple[int, ...]]:
        return backend, tuple(int(v) for v in x)

    def _write_header(self) -> None:
        self._writer.writerow(self._header)
        self._outf.flush()

    def _write_row(self, row: Sequence) -> None:
        self._rows.append(row)
        self._writer.writerow(row)
        self._outf.flush()
        try:
            os.fsync(self._outf.fileno())
        except OSError:
            logger.debug("Unable to fsync output file %s", self._fname)

    def _write_result(self, result: Sequence) -> None:
        self._results.append(result)
        x, error, time, backend = result
        if error != 0:
            logger.debug(f"Skip recording error for: {backend}: {x}")
            return
        key = self._sample_key(x, backend)
        if self._resume and key in self._seen_keys:
            logger.debug("Skip already recorded sample for resume mode: %s", key)
            return
        peak = self._peak_time / time
        s = str(x).replace(",", ";")
        row = [s, time, peak, backend]
        row = x + row
        logger.debug(f"Record row: {row}")
        self._write_row(row)
        self._seen_keys.add(key)

    @override
    def __call__(self, result: Sequence) -> None:
        self._write_result(result)

    def __del__(self) -> None:
        self._outf.close()
