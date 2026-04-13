#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import shutil
from pathlib import Path
import subprocess
import tempfile


def get_mlir_prefix(prefix: Path | str | None = None) -> Path:
    """
    Tentatively return the mlir compiler prefix where
    {prefix}/bin/mlir-opt can be found.
    Raise on error.
    Defined in order as:
    - passed prefix if not None
    - env var XTC_MLIR_PREFIX
    - mlir python package prefix if installed
    - mlir-opt binary prefix in PATH
    """
    how = None
    if prefix is None:
        prefix_var = os.environ.get("XTC_MLIR_PREFIX")
        if prefix_var:
            how, prefix = "XTC_MLIR_PREFIX envvar", Path(prefix_var)
        else:
            try:
                import mlir

                how, prefix = "mlir package", Path(mlir.__path__[0])
            except:
                mlir_exe = shutil.which("mlir-opt")
                if mlir_exe:
                    how, prefix = "mlir-opt PATH", Path(mlir_exe).parents[1]
    else:
        how, prefix = "explicit prefix", Path(prefix)
    if prefix is None:
        raise RuntimeError("could not find MLIR installation")
    if not prefix.exists():
        raise RuntimeError(f"could not find MLIR prefix at: {prefix}, method; {how}")
    mlir_opt = prefix / "bin" / "mlir-opt"
    if not mlir_opt.exists():
        if how == "mlir package":
            # Try to find prefix from MLIR standard python package install
            prefix2 = prefix.parents[2].resolve()
            mlir_opt2 = prefix2 / "bin" / "mlir-opt"
            raise RuntimeError(
                f"could not find mlir-opt at: {mlir_opt}, nor: {mlir_opt2}, method: {how}"
            )
        else:
            raise RuntimeError(f"could not find mlir-opt at: {mlir_opt}, method: {how}")
    return prefix


def get_llvm_prefix(prefix: Path | str | None = None) -> Path:
    """
    Tentatively return the llvm compiler prefix where
    {prefix}/bin/opt can be found.
    Raise on error.
    Defined in order as:
    - passed prefix if not None
    - env var XTC_LLVM_PREFIX
    - get_mlir_prefix() if successfull
    - llvm python package prefix if installed
    - opt binary prefix in PATH
    """
    how = None
    if prefix is None:
        prefix_var = os.environ.get("XTC_LLVM_PREFIX")
        if prefix_var:
            how, prefix = "XTC_LLVM_PREFIX envvar", Path(prefix_var)
        else:
            try:
                mlir_prefix = get_mlir_prefix()
                if not (mlir_prefix / "bin" / "opt").exists():
                    raise RuntimeError()
                how, prefix = "mlir prefix", mlir_prefix
            except RuntimeError:
                try:
                    import llvm

                    how, prefix = "llvm package", Path(llvm.__path__[0])
                except:
                    opt_exe = shutil.which("opt")
                    if opt_exe:
                        how, prefix = "opt PATH", Path(opt_exe).parents[1]
    else:
        how, prefix = "explicit prefix", Path(prefix)
    if prefix is None:
        raise RuntimeError("could not find LLVM installation")
    if not prefix.exists():
        raise RuntimeError(f"could not find LLVM prefix at: {prefix}, method; {how}")
    llvm_opt = prefix / "bin" / "opt"
    if not llvm_opt.exists():
        raise RuntimeError(f"could not find opt at: {llvm_opt}, method: {how}")
    return prefix


def get_geist_prefix(prefix: Path | str | None = None) -> Path:
    """
    Tentatively return the mlir polygeist prefix where
    {prefix}/bin/cgeist can be found.
    Raise on error.
    Defined in order as:
    - passed prefix if not None
    - env var XTC_GEIST_PREFIX
    - polygeist python package prefix if installed
    - cgeist binary prefix in PATH
    """
    if prefix is None:
        prefix_var = os.environ.get("XTC_GEIST_PREFIX")
        if prefix_var:
            prefix = Path(prefix_var)
        else:
            try:
                import polygeist

                prefix = Path(polygeist.__path__[0])
            except:
                cgeist_exe = shutil.which("cgeist")
                if cgeist_exe:
                    prefix = Path(cgeist_exe).parents[1]
    else:
        prefix = Path(prefix)
    if prefix is None:
        raise RuntimeError("could not find Polygeist installation")
    if not prefix.exists():
        raise RuntimeError(f"could not find Polygeist prefix at: {prefix}")
    cgeist = prefix / "bin" / "cgeist"
    if not cgeist.exists():
        raise RuntimeError(f"could not find cgeist at: {cgeist}")
    return prefix


def get_cuda_prefix(prefix: Path | str | None = None) -> Path:
    """
    Tentatively return the cuda installation dir
    Raise on error.
    Defined in order as:
    - passed prefix if not None
    - env var CUDA_INSTALL_DIR
    - /usr/local/cuda
    """
    if prefix is None:
        prefix_var = os.environ.get("CUDA_INSTALL_DIR")
        if prefix_var:
            prefix = Path(prefix_var)
        else:
            prefix = Path("/usr/local/cuda")
    else:
        prefix = Path(prefix)
    if not prefix.exists():
        raise RuntimeError(f"could not find CUDA installation dir at: {prefix}")
    return prefix


def check_compile(code: str, libs: str | list[str] | None = None):
    """
    Attempt to compile (and link) a small C program.
    """
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        return False

    if isinstance(libs, str):
        libs = [libs]
    libs = libs or []

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.c")
        exe_path = os.path.join(tmpdir, "test")

        with open(src_path, "w") as f:
            f.write(code)

        cmd = [compiler, src_path, "-o", exe_path] + [f"-l{lib}" for lib in libs]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
