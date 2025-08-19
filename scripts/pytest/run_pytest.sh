#!/usr/bin/env bash
#
# Requires: pip install pytest pytest-xdist
#
set -euo pipefail
jobs="$(nproc)"

set -x
exec pytest -n "$jobs" "$@"
