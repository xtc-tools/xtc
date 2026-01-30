#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes.util
import subprocess
import re
import platform

import os
import platform
import ctypes.util
import subprocess
import re


def get_library_path(libname: str) -> str:
    # Check Homebrew path for macOS first
    if platform.system() == "Darwin":
        homebrew_path = f"/opt/homebrew/opt/libomp/lib/lib{libname}.dylib"
        if os.path.exists(homebrew_path):
            return homebrew_path

    libfile = ctypes.util.find_library(libname)
    assert libfile, (
        f"ctypes.util.find_library: can't find library: {libname}, please install corresponding package"
    )

    if platform.system() == "Darwin":
        return libfile

    result = subprocess.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if libfile in line:
            match = re.search(r"=>\s+(\S+)", line)
            if match:
                return match.group(1)
    assert False


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
    "-O3",
    "-filetype=obj",
]

opt_opts = ["-O3", "--enable-unsafe-fp-math", "--fp-contract=fast"]

target_cc_opts = ["-O3", "-ffast-math", "--fp-contract=fast"]

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
