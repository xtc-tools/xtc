#!/usr/bin/env bash
#
# Test notebooks in the docs/marimo dir
#
set -euo pipefail

dir="$(dirname "$0")"

MARIMO_DIR="$dir"/../../docs/marimo
(set -x; python "$MARIMO_DIR"/mlir_example_notebook.py)
(set -x; python "$MARIMO_DIR"/mlir_loop_notebook.py)
