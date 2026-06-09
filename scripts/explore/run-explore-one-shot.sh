#!/usr/bin/env bash
set -euo pipefail

dir="$(dirname "$0")"

outdir="${1-.}"
BACKENDS="${BACKENDS:-mlir tvm jir}"
OPTS="${OPTS:-1 2 3}"
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
for op in $OPS; do
    for s in $STRATEGIES; do
        for b in $BACKENDS; do
            for o in $OPTS; do
                echo "Testing operator $op on backend $b with tiling strategy $s for opt level $o..."
                (set -x && loop-explore --graph "$graph_dir"/"$op".graph.yaml --backends "$b" --opt-level "$o" --jobs 1 --strategy "$s" --output "$outdir/results.$op.b$b.s$s.o$o.csv" --db-file "$db_file")
            done
        done
    done
done
for op in $OPS; do
    (set -x && loop-results --graph "$graph_dir"/"$op".graph.yaml --db-file "$db_file")
done
