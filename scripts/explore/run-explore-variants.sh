#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"
BACKENDS="${BACKENDS:-mlir tvm jir}"
TRIALS="${TRIALS:-20}"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dv tile4dvr " \
          "tile7d tile7dv tile7dvr " \
          "tile8d tile8dv tile8dvr" \
          "}"

mkdir -p "$outdir"
rm -f "$outdir/*.csv"

t="$TRIALS"
for s in $STRATEGIES; do
    for b in $BACKENDS; do
        # Check whether the given strategy is supported by the backend
        echo "Testing whether backend $b supports strategy $s..."
        (set -x && loop-explore --validate --backends "$b" --opt-level 3 --execute --strategy "$s" --quiet 2>/dev/null) || continue
        echo "Testing backend $b with tiling strategy $s for $t trials..."
        (set -x && loop-explore --backends "$b" --trials "$t" --jobs 1 --strategy "$s" --output "$outdir/results.$b.$s.$t.csv")
    done
done

