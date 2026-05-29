#!/usr/bin/env bash
#
# Run python coverage over an arbitrary command line.
#
# Usage: ./coverage_run.sh <cmd...>
#
# Uses the coverage tool: `pip install coverage`.
#
# Will generate coverage file for each python process in
# in `$PWD/.coverage*`. The coverage files should then
# be combined with `coverage combine`.
#
# Example:
#     ./scripts/coverage/coverage_run.sh lit tests
#     coverage combine
#     coverage report
#
# Note:
# - we have to use a `sitecustomize.py` file present in `sitecustomize`,
#   hence if running python processes requiring sitecustomize this will
#   not work
# - coverage tool options are passed through environment,
#   hence the env variables defined below must be presereved
#   in subprocesses environment
#

set -euo pipefail

dir="$(dirname "$(readlink -f "$0")")"

# These vars must be preserved in subprocesses' environment
export PYTHONPATH="$dir/sitecustomize${PYTHONPATH+:$PYTHONPATH}"
export COVERAGE_PROCESS_START="$dir/../../pyproject.toml"
export COVERAGE_FILE="$PWD/.coverage"

exec "$@"
