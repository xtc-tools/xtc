#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import shutil
from pathlib import Path


def get_mlir_prefix(prefix: Path | str | None = None):
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
    if prefix is None:
        prefix_var = os.environ.get("XTC_MLIR_PREFIX")
        if prefix_var:
            prefix = Path(prefix_var)
        else:
            try:
                import mlir

                prefix = Path(mlir.__path__[0])
            except:
                mlir_exe = shutil.which("mlir-opt")
                if mlir_exe:
                    prefix = Path(mlir_exe).parents[1]
    else:
        prefix = Path(prefix)
    if prefix is None:
        raise RuntimeError("could not find MLIR installation")
    if not prefix.exists():
        raise RuntimeError(f"could not find MLIR prefix at: {prefix}")
    mlir_opt = prefix / "bin" / "mlir-opt"
    if not mlir_opt.exists():
        prefix = prefix.parents[2].resolve()
    mlir_opt2 = prefix / "bin" / "mlir-opt"
    if not mlir_opt2.exists():
        raise RuntimeError(f"could not find mlir-opt at: {mlir_opt}")
    return prefix


def get_geist_prefix(prefix: Path | str | None = None):
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


def get_cuda_prefix(prefix: Path | str | None = None):
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
