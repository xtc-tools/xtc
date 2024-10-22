#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
transform_opts = [
    "transform-interpreter",
    "canonicalize",
]

lowering_opts = [
    "lower-affine",
    "convert-vector-to-scf",
    "convert-linalg-to-loops",
    "loop-invariant-code-motion",
    "func.func(buffer-loop-hoisting)",
    "cse",
    "sccp",
    "canonicalize",
    "convert-scf-to-cf",
    "convert-vector-to-llvm{enable-x86vector=true}",
    "convert-math-to-llvm",
    "expand-strided-metadata",
    "lower-affine",
    "buffer-results-to-out-params",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}",
    "convert-index-to-llvm",
    "reconcile-unrealized-casts",
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
    "libmlir_c_runner_utils.so.18git",
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
