#!/usr/bin/env bash
#
# Requires: pip install pytest pytest-xdist
#
set -euo pipefail
dir="$(dirname "$0")"

jobs="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

# Reduce jobs to at max 8, as pytest jobs setup is actually costly
[ "$jobs" -le 8 ] || jobs=8

# Get to top dir as pytest does not discover pyproject.toml otherwise
cd "$dir"/../..

set -x
exec pytest -n "$jobs" "$@"
