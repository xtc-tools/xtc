#!/usr/bin/env bash
#
# Test explore_optimizers tutorial in several configurations
#
set -euo pipefail

dir="$(dirname "$0")"

MARIMO="$dir"/../../docs/tutorials/explore_optimizers.py
set -x
python "$MARIMO" --search-type random --trials 8
python "$MARIMO" --search-type iterative --batch 4 --trials 16 --opt-name random
python "$MARIMO" --search-type iterative --batch 4 --trials 16 --opt-name random-forest-default
python "$MARIMO" --search-type iterative --batch 4 --trials 16 --opt-name random-forest-explore
python "$MARIMO" --search-type iterative --batch 4 --trials 16 --opt-name random-forest-aggressive
