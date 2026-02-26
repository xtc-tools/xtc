#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"
BACKENDS="${BACKENDS:-mlir tvm jir}"
TRIALS="${TRIALS:-20}"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dv " \
          "tile7d tile7dv tile7dvr " \
          "}"

mkdir -p "$outdir"
rm -f "$outdir/*.csv"

op="matmul"
dims="512 1024 128"

t="$TRIALS"
for s in $STRATEGIES; do
    for b in $BACKENDS; do
        echo "Testing backend $b with tiling strategy $s for $t trials..."
        (set -x && loop-explore --backends "$b" --operator $op --dims $dims --trials "$t" --jobs 1 --strategy "$s" --output "$outdir/results.b$b.s$s.t$t.csv" --db-file "$outdir/results.db")
    done
done
result="$(set -x && db-results --operator $op --dims $dims --db-file "$outdir/results.db")"
[ -n "$result" ] || { echo "ERROR: unexpected empty db" >&2 ; exit 1; }
echo "$result"
