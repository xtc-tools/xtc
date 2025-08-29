#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes.util
import subprocess
import re


def get_library_path(libname: str) -> str:
    libfile = ctypes.util.find_library(libname)
    assert libfile

    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if libfile in line:
            match = re.search(r"=>\s+(\S+)", line)
            if match:
                return match.group(1)
    assert False


transform_opts = [
    "transform-interpreter",
]

lowering_llvm_opts = [
    "canonicalize",
    "cse",
    "sccp",
    # From complex control to the soup of basic blocks
    "expand-strided-metadata",
    "convert-linalg-to-loops",
    "lower-affine",
    "convert-vector-to-scf{full-unroll=true}",
    "scf-forall-to-parallel",
    "convert-scf-to-openmp",
    "canonicalize",
    "cse",
    "sccp",
    "convert-scf-to-cf",
    "canonicalize",
    "cse",
    "sccp",
    # Memory accesses to LLVM
    "buffer-results-to-out-params",
    "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}",
    "finalize-memref-to-llvm",
    "canonicalize",
    "cse",
    "sccp",
    # Data flow to LLVM
    "convert-vector-to-llvm{enable-x86vector=true}",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "canonicalize",
    "cse",
    "sccp",
    # Control flow to LLVM
    "convert-cf-to-llvm",
    "convert-openmp-to-llvm",
    "canonicalize",
    "cse",
    "sccp",
]

lowering_std_opts = [
    "canonicalize",
    "cse",
    "sccp",
    # From complex control to the soup of basic blocks
    "expand-strided-metadata",
    "convert-linalg-to-loops",
    "lower-affine",
    "convert-vector-to-scf{full-unroll=true}",
    "scf-forall-to-parallel",
    "convert-scf-to-openmp",
    "canonicalize",
    "cse",
    "sccp",
    # "convert-scf-to-cf",
    "canonicalize",
    "cse",
    "sccp",
    # Memory accesses
    "buffer-results-to-out-params",
    "canonicalize",
    "cse",
    "sccp",
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
    "libmlir_runner_utils.so",
    "libmlir_c_runner_utils.so",
    "libmlir_async_runtime.so",
]

system_libs = [get_library_path("omp")]

dump_file = "/tmp/dump"

mlirrunner_opts = [
    "-e",
    "entry",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = "x86_64-linux-gnu-objdump"

objdump_arm_bin = "aarch64-linux-gnu-objdump"

target_cc_bin = "cc"

cc_bin = "cc"

objdump_opts = [
    "-dr",
    "--no-addresses",
    "--no-show-raw-insn",
]

objdump_color_opts = [
    "--disassembler-color=on",
]
