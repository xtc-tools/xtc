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


from xtc.graphs.xtc.operators import XTCOperator

logger = logging.getLogger(__name__)

VERSION = "v0.1"

DBEntry: TypeAlias = dict[str, Any]


def get_native_platform():
    import platform

    node = platform.node().split(".")[0]
    return [node, platform.system(), platform.machine()]


def get_node_platform(node: str, target: str):
    return [node, "Linux", target]


class ResultsDB(ABC):
    def __init__(
        self,
        db_file: str,
        db_version: str = VERSION,
        target: str = "native",
        node: str = "",
    ):
        self._db_file = db_file
        self._version = [db_version]
        if target == "native":
            self._platform = get_native_platform()
        else:
            assert node != "", f"node must be specified for non native target"
            self._platform = get_node_platform(node, target)
        self._xtc_version = "v" + importlib.metadata.version("xtc")
        self._results = []
        self._reload()

    def _reload(self):
        self._results = []
        with open(self._db_file) as inf:
            for jsonl in inf.readlines():
                log = json.loads(jsonl)
                self._results.append(log)
        self._results

    def _default_match(
        self,
        log: DBEntry,
        operator: list[Any],
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
    ) -> bool:
        compiler = ["xtc", self._xtc_version, target, threads]
        if backend is not None:
            compiler.append(backend)
        return (
            self._version == log["version"][: len(self._version)]
            and operator == log["operator"][: len(operator)]
            and self._platform == log["platform"][: len(self._platform)]
            and compiler == log["compiler"][: len(compiler)]
        )

    def get_operation_results(
        self,
        xtc_op_signature: list[Any],
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
        errors: bool = False,
    ) -> list[DBEntry]:
        operator = ["xtc.operator", *xtc_op_signature]
        results = []
        for log in self._results:
            if not self._default_match(
                log,
                operator=operator,
                target=target,
                threads=threads,
                backend=backend,
            ):
                continue
            if not errors and log["results"][0] != 0:
                continue
            results.append(log)
        return results

    def get_operation_best(
        self,
        xtc_op_signature: list[Any],
        target: str = "native",
        threads: int = 1,
        backend: str | None = None,
    ) -> DBEntry | None:
        best_time = float("+inf")
        best_log = None
        logs = self.get_operation_results(
            xtc_op_signature,
            target,
            threads,
            backend,
            errors=False,
        )
        for log in logs:
            time = min(log["results"][1])
            if time < best_time:
                best_log = log
                best_time = time
        return best_log


def get_signature_from_args(op_type: str, *spec: Any) -> list[Any]:
    match op_type:
        case "conv2d":
            n, h, w, f, r, s, c, SH, SW, dtype = spec
            signature = XTCOperator.get_op_signature(
                "conv2d",
                n,
                h,
                w,
                f,
                r,
                s,
                c,
                dtype,
                stride=(SH, SW),
            )
        case _:
            signature = XTCOperator.get_op_signature(op_type, *spec)
    logger.debug("matching for signature: %s", signature)
    return signature


def main():
    default_dtype = "float32"
    default_db_file = "xtc-operators-db.json"
    default_target = "native"
    default_threads = 1

    parser = argparse.ArgumentParser(
        description="Report DB results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--operator", required=True, type=str, help="operator to query")
    parser.add_argument("--dims", nargs="+", type=int, required=True, help="dimensions")
    parser.add_argument("--dtype", type=str, default=default_dtype, help="data type")
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

    spec = [*args.dims, args.dtype]
    signature = get_signature_from_args(args.operator, *spec)

    db = ResultsDB(db_file=args.db_file)

    if args.dump:
        results = db.get_operation_results(
            xtc_op_signature=signature,
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
            )
    if args.best:
        log = db.get_operation_best(
            xtc_op_signature=signature,
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
