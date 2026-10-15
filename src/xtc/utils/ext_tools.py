#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes.util
import os
import platform
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

from xtc.utils.tools import check_compile

homebrew_libomp_prefix = "/opt/homebrew/opt/libomp"


def get_library_path(libname: str) -> str:
    # Check Homebrew path for macOS first
    if platform.system() == "Darwin":
        homebrew_path = f"{homebrew_libomp_prefix}/lib/lib{libname}.dylib"
        if os.path.exists(homebrew_path):
            return homebrew_path

    libfile = ctypes.util.find_library(libname)
    assert libfile, (
        f"ctypes.util.find_library: can't find library: {libname}, please install corresponding package"
    )

    if platform.system() == "Darwin":
        return libfile

    libpath = Path(libfile)
    if libpath.is_absolute() and libpath.is_file():
        return str(libpath)

    # Nix and other non-FHS environments expose libraries through environment
    # search paths rather than the host's ldconfig cache.
    search_paths: list[Path] = []
    for variable in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        search_paths.extend(
            Path(path)
            for path in os.environ.get(variable, "").split(os.pathsep)
            if path
        )
    for directory in search_paths:
        candidate = directory / libfile
        if candidate.is_file():
            return str(candidate)

    ldconfig = shutil.which("ldconfig")
    if ldconfig is None and Path("/sbin/ldconfig").is_file():
        ldconfig = "/sbin/ldconfig"
    if ldconfig is not None:
        result = subprocess.run([ldconfig, "-p"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if libfile in line:
                match = re.search(r"=>\s+(\S+)", line)
                if match:
                    return match.group(1)

    searched = os.pathsep.join(str(path) for path in search_paths)
    raise RuntimeError(
        f"could not resolve {libfile} for {libname}; searched: {searched}"
    )


@cache
def get_openmp_cflags(compiler: str) -> list[str]:
    """
    Returns the flags enabling OpenMP for the given C compiler.
    Apple clang rejects -fopenmp in the driver, hence the -Xpreprocessor
    fallback with Homebrew libomp headers.
    Returns an empty list when OpenMP is unavailable, in which case
    omp pragmas are ignored and the code is compiled sequentially.
    """
    code = (
        "#include <omp.h>\n#ifndef _OPENMP\n#error\n#endif\nint main() { return 0; }\n"
    )
    candidates = [["-fopenmp"]]
    if platform.system() == "Darwin":
        candidates.append(
            [
                "-Xpreprocessor",
                "-fopenmp",
                f"-I{homebrew_libomp_prefix}/include",
            ]
        )
    for flags in candidates:
        if check_compile(code, flags=flags, link=False, compiler=compiler):
            return flags
    return []


def get_shlib_extension():
    if platform.system() == "Darwin":
        return "dylib"

    return "so"


transform_opts = [
    "transform-interpreter",
]

mlirtranslate_opts = ["--mlir-to-llvmir"]

xtctranslate_opts = ["--mlir-to-c"]

llc_opts = [
    "-O2",
    "-filetype=obj",
]

opt_opts = ["-O2", "--fp-contract=fast"]

target_cc_opts = ["-O3", "-ffp-contract=fast"]

cc_opts = ["-O3", "-march=native"]

shared_lib_opts = ["--shared", *cc_opts]

exe_opts = [*cc_opts]


runtime_libs = [
    f"libmlir_runner_utils.{get_shlib_extension()}",
    f"libmlir_c_runner_utils.{get_shlib_extension()}",
    f"libmlir_async_runtime.{get_shlib_extension()}",
]

cuda_runtime_lib = "libmlir_cuda_runtime.so"

system_libs = [get_library_path("omp")]

dump_file = "/tmp/dump"

mlirrunner_opts = [
    "-e",
    "entry",
    "--entry-point-result=void",
    "--O3",
]

target_cc_bin = "cc"

cc_bin = "cc"
