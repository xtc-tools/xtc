#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
transform_opts = [
    "transform-interpreter",
]

lowering_opts = [
    "canonicalize",
    "cse",
    "sccp",
    # From complex control to the soup of basic blocks
    "expand-strided-metadata",
    "lower-affine",
    "convert-linalg-to-loops",
    "convert-vector-to-scf",
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

mlirtranslate_opts = ["--mlir-to-llvmir"]

llc_opts = ["-O3", "-filetype=obj", "--mcpu=native"]

opt_opts = ["-O3", "--march=native"]

cc_opts = ["-O3", "-march=native"]

shared_lib_opts = ["--shared", *cc_opts]

exe_opts = [*cc_opts]

runtime_libs = [
    "libmlir_runner_utils.so",
    "libmlir_c_runner_utils.so",
    "libmlir_async_runtime.so",
]

dump_file = "/tmp/dump"

mlirrunner_opts = [
    "-e",
    "entry",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = "objdump"

cc_bin = "cc"

objdump_opts = ["-dr", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]
