#!/usr/bin/env bash
set -euo pipefail

ruff="ruff"
ruff_opts="--force-exclude"
git ls-files -z '*.py' | xargs -0 "$ruff" format $ruff_opts "$@"
