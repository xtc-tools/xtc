#!/usr/bin/env bash
set -euo pipefail

dir="$(dirname "$0")"

outdir="${1-.}"
BACKENDS="${BACKENDS:-mlir tvm jir}"
TRIALS="${TRIALS:-20}"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dv " \
          "tile7d tile7dv tile7dvr " \
          "}"
OPS="matmul conv2d"

graph_dir="$dir"/../../tests/graphs

mkdir -p "$outdir"
rm -f "$outdir/*.csv"
rm -f "$outdir/*.json"

db_file="$outdir/results-variants-db.json"
t="$TRIALS"
for op in $OPS; do
    for s in $STRATEGIES; do
        for b in $BACKENDS; do
            echo "Testing backend $b with tiling strategy $s for $t trials..."
            (set -x && loop-explore --graph "$graph_dir"/"$op".graph.yaml --backends "$b" --trials "$t" --jobs 1 --strategy "$s" --output "$outdir/results.b$b.s$s.t$t.csv" --db-file "$db_file")
        done
    done
done
for op in $OPS; do
    (set -x && loop-results --graph "$graph_dir"/"$op".graph.yaml --db-file "$db_file")
done

