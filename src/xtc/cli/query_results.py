#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
from typing import TypeAlias, Any
import importlib.metadata
import json
import argparse
import sys
import logging
import hashlib
import platform
from collections.abc import Generator, Iterable

from xtc.itf.graph import Graph
from xtc.itf.search import Strategy
from xtc.graphs.xtc.graph import XTCGraph

logger = logging.getLogger(__name__)

DBEntry: TypeAlias = dict[str, Any]


class ResultsDB(ABC):
    VERSION = "v0.2"

    def __init__(
        self,
        db_file: str,
        db_version: str = VERSION,
        node_target: str = "native",
        node: str | None = None,
    ):
        self._db_file = db_file
        self._version = [db_version]
        self._node = node
        self._node_target = node_target

    def _read_results(self) -> Generator[DBEntry, None, None]:
        with open(self._db_file) as inf:
            for jsonl in inf.readlines():
                yield json.loads(jsonl)

    def _default_match(
        self,
        results: Iterable[DBEntry],
        graph: Graph,
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
    ) -> Generator[DBEntry, None, None]:
        version = self.get_version()
        if self._node_target == "native":
            platform = self.get_native_platform()
        else:
            assert self._node is not None, (
                f"node must be specified for non native target"
            )
            platform = self.get_node_platform(self._node, self._node_target)
        compiler = self.get_compiler(target, threads, backend)
        operator = self.get_operator(graph)
        logger.debug("MATCH: version: %s", version)
        logger.debug("MATCH: platform: %s", platform)
        logger.debug("MATCH: compiler: %s", compiler)
        logger.debug("MATCH: operator: %s", operator)
        operator = operator[:2]
        for log in results:
            if (
                version == log["version"][: len(version)]
                and operator == log["operator"][: len(operator)]
                and platform == log["platform"][: len(platform)]
                and compiler == log["compiler"][: len(compiler)]
            ):
                yield log

    @classmethod
    def get_xtc_version(cls) -> str:
        return "v" + importlib.metadata.version("xtc")

    @classmethod
    def get_version(cls) -> list[Any]:
        return [cls.VERSION]

    @classmethod
    def get_compiler(
        cls, target: str = "native", threads: int = 1, backend: str | None = None
    ) -> list[Any]:
        compiler = ["xtc", cls.get_xtc_version(), target, threads]
        if backend is not None:
            compiler.append(backend)
        return compiler

    @classmethod
    def get_native_platform(cls) -> list[Any]:
        node = platform.node().split(".")[0]
        return [node, platform.system(), platform.machine()]

    @classmethod
    def get_node_platform(cls, node: str, target: str) -> list[Any]:
        return [node, "Linux", target]

    @classmethod
    def get_operator(cls, graph: Graph | str) -> list[Any]:
        obj: Any
        if not isinstance(graph, str):
            assert isinstance(graph, XTCGraph)
            obj = graph.to_dict()
            if "name" in obj:
                del obj["name"]
        else:
            obj = graph
        digest = hashlib.sha1(json.dumps(obj).encode()).hexdigest()
        operator = ["xtc.graph", digest, obj]
        return operator

    @classmethod
    def get_strategy(cls, strategy: Strategy | str) -> list[Any]:
        assert isinstance(strategy, str), f"TODO strategy type not yet supported"
        obj: Any = strategy
        digest = hashlib.sha1(json.dumps(obj).encode()).hexdigest()
        operator = ["xtc.strategy", digest, obj]
        return operator

    def get_results(
        self,
        graph: Graph,
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
        allow_errors: bool = False,
    ) -> list[DBEntry]:
        results = []
        for log in self._default_match(
            self._read_results(),
            graph=graph,
            target=target,
            threads=threads,
            backend=backend,
        ):
            if not allow_errors and log["results"][0] != 0:
                continue
            results.append(log)
        return results

    def get_best(
        self,
        graph: Graph,
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
    ) -> DBEntry | None:
        logs = self.get_results(
            graph,
            target,
            threads,
            backend,
            allow_errors=False,
        )
        ordered = sorted(logs, key=lambda x: min(x["results"][1]))
        return ordered[0] if len(ordered) > 0 else None


def main():
    default_db_file = "xtc-graphs-db.json"
    default_target = "native"
    default_threads = 1

    parser = argparse.ArgumentParser(
        description="Report DB results for a graph file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", required=True, type=str, help="graph yaml file")
    parser.add_argument("--target", type=str, default=default_target, help="target")
    parser.add_argument(
        "--threads", type=int, default=default_threads, help="threads for target"
    )
    parser.add_argument("--backend", type=str, help="optional backend for filter")
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="dump all matched results",
    )
    parser.add_argument(
        "--best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="get best result",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="quiet mode, only results on output",
    )
    parser.add_argument(
        "--db-file", type=str, default=default_db_file, help="results json db"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="debug mode",
    )
    args = parser.parse_args()

    logging.basicConfig()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    db = ResultsDB(db_file=args.db_file)

    import xtc.graphs.xtc.op as O

    with O.graph() as gb:
        gb.load(args.graph)
    graph = gb.graph

    if args.dump:
        results = db.get_results(
            graph=graph,
            target=args.target,
            threads=args.threads,
        )
        num = len(results)
        for idx, entry in enumerate(results):
            print(
                f"result {idx + 1}/{num}: compiler:",
                entry["compiler"],
                "results:",
                entry["results"][1],
                "strategy:",
                entry["strategy"][2],
                "schedule:",
                entry["schedule"],
            )
    if args.best:
        log = db.get_best(
            graph=graph,
            target=args.target,
            threads=args.threads,
        )
        if log is not None:
            time = min(log["results"][1])
            print(
                "best:",
                time,
                "strategy:",
                log["strategy"],
                "schedule:",
                log["schedule"],
                "compiler:",
                log["compiler"],
            )
        elif not args.quiet:
            print("warning: no match found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
