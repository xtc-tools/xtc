#!/usr/bin/env bash
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# Build the IREE runtime from sources and install a self-contained C SDK
# (headers + static archives) under a prefix consumed by
# xtc.utils.tools.get_iree_prefix() (default: ~/.cache/xtc/iree-runtime,
# override IREE_RUNTIME_DIR):
#   {prefix}/include  IREE runtime headers (source + generated + flatcc)
#   {prefix}/lib      IREE runtime static archives (*.a)
#
# The xtc_iree_shim shared library is not built here, so there is no CMake
# dependency inside the xtc runtime.
#
set -euo pipefail

# Keep in sync with iree_requirements.txt (iree-base-runtime==3.11.0).
IREE_VERSION="${IREE_VERSION:-v3.11.0}"
PREFIX="${IREE_RUNTIME_DIR:-$HOME/.cache/xtc/iree-runtime}"
WORK="${IREE_BUILD_DIR:-$HOME/.cache/xtc/iree-build}"
# Submodules required by a runtime-only build (compiler OFF, tests OFF). flatcc
# is the flatbuffer runtime; append more here if configure reports a missing one.
SUBMODULES="${IREE_SUBMODULES:-third_party/flatcc third_party/benchmark third_party/googletest third_party/printf}"

SRC="$WORK/iree"
BUILD="$WORK/build-runtime"
mkdir -p "$WORK"

if [ ! -e "$SRC/.git" ]; then
  echo ">> Cloning IREE $IREE_VERSION (no submodules yet)"
  git clone --depth 1 --branch "$IREE_VERSION" \
    https://github.com/iree-org/iree.git "$SRC"
fi

echo ">> Fetching runtime submodules: $SUBMODULES"
git -C "$SRC" submodule update --init --depth 1 $SUBMODULES

echo ">> Configuring IREE runtime (compiler OFF, local CPU drivers only)"
cmake -G Ninja -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_C_FLAGS="-fPIC" \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY=ON

# Building iree_runtime_unified pulls in the whole runtime link closure, so all
# the static archives xtc links the shim against get built as a side effect.
echo ">> Building the runtime static libraries"
cmake --build "$BUILD" --target iree_runtime_unified

echo ">> Installing the C SDK under: $PREFIX"
rm -rf "$PREFIX/include" "$PREFIX/lib"
mkdir -p "$PREFIX/include" "$PREFIX/lib"
# Headers: source tree, generated tree (flatbuffer schemas, config), and flatcc.
# IREE has no relocatable C-SDK install, so we copy the headers ourselves; only
# *.h is needed to compile the shim.
for hdrs in "$SRC/runtime/src/" "$BUILD/runtime/src/" "$SRC/third_party/flatcc/include/"; do
  rsync -am --include='*/' --include='*.h' --exclude='*' "$hdrs" "$PREFIX/include/"
done
# Archives: static libs are position-independent, so copying them out of the
# build tree is relocatable (unlike IREE's own SDK install).
find "$BUILD" -name '*.a' -exec cp {} "$PREFIX/lib/" \;

echo ">> Done. IREE C SDK installed under: $PREFIX"
echo "   include/ headers, lib/ $(ls "$PREFIX"/lib/*.a | wc -l | tr -d ' ') archives"
echo "   IREE runtime build tree kept at: $BUILD (re-runs are incremental)"
echo "   xtc builds xtc_iree_shim lazily from this SDK on first use."
