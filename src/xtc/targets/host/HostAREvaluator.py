#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import tempfile
from pathlib import Path
import subprocess
import shlex
import shutil
import sys

import xtc.itf as itf
import xtc.targets.host as host


__all__ = [
    "HostAREvaluator",
    "HostARExecutor",
]


class HostAREvaluator(itf.exec.Evaluator):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        assert module.file_type == "arlib", (
            "must pass a arlib module to a HostAREvaluator"
        )
        self._module = module
        self._build_shlib_module()
        self._shlib_evaluator = host.HostEvaluator(self._shlib_module, **kwargs)

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        return self._shlib_evaluator.evaluate()

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module

    def _compile_to_shlib(self, shlib_base: str):
        cwd_dir = Path(shlib_base).parent
        shlib_name = Path(shlib_base).stem
        opts = "-O3 -march=native -mtune=native"
        sh_opts = "--shared -fPIC"
        arlibs = [
            str(Path(fname).absolute())
            for fname in [self._module.file_name] + self._module.arlibs
        ]
        opt_whole, opt_no_whole = "-Wl,--whole-archive", "-Wl,--no-whole-archive"
        ext = ".so"
        if sys.platform == "darwin":
            sh_opts += " -undefined dynamic_lookup"
            opt_whole, opt_no_whole = "-Wl,-all_load", ""
            ext = ".dylib"
        cmd = (
            f"cc {sh_opts} {opts} "
            f"{opt_whole}  "
            f"{' '.join(arlibs)} "
            f"{opt_no_whole}  "
            f"-o {shlib_name}{ext}"
        )
        p = subprocess.run(
            shlex.split(cmd), text=True, capture_output=True, cwd=cwd_dir
        )
        if p.returncode != 0:
            raise RuntimeError(f"Failed command {cmd}:\n{p.stdout}\n{p.stderr}\n")

    def _build_shlib_module(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        c_stem = Path(self._module.file_name).stem
        shlib_base = str(Path(self.tmp_dir.name) / f"{c_stem}_eval")
        self._compile_to_shlib(shlib_base)
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        self._shlib_module: host.HostModule = host.HostModule(
            shlib_base,
            self._module.payload_name,
            f"{shlib_base}{ext}",
            "shlib",
            bare_ptr=self._module._bare_ptr,
            graph=self._module._graph,
        )

    def __del__(self):
        shutil.rmtree(self.tmp_dir.name, ignore_errors=True)


class HostARExecutor(itf.exec.Executor):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        self._evaluator = HostAREvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=1,
            **kwargs,
        )

    @override
    def execute(self) -> int:
        _, code, _ = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
