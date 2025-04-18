#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"
BACKENDS="${BACKENDS:-mlir tvm jir}"
OPTS="1 2 3"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dv " \
          "tile7d tile7dv tile7dvr " \
          "}"

mkdir -p "$outdir"
rm -f "$outdir/*.csv"

for s in $STRATEGIES; do
    for b in $BACKENDS; do
        for o in $OPTS; do
            echo "Testing backend $b with tiling strategy $s for opt level $o..."
            (set -x && loop-explore --backends "$b" --opt-level "$o" --jobs 1 --strategy "$s" --output "$outdir/results.b$b.s$s.o$o.csv")
        done
    done
done

