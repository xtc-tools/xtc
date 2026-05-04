#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"
BACKENDS="${BACKENDS:-tvm}"
TRIALS="${TRIALS:-100}"
SEEDS="${SEEDS:- 1 2 3 4 5}"
STRATEGY="${STRATEGY:-tile_ppwrprp}"
OPTIMIZERS="${OPTIMIZERS:-random random-forest-aggressive}"
PROBLEM="${PROBLEM:-ResNet18_01}"
OPERATOR="${OPERATOR:-conv2d}"
JOBS="${JOBS:-1}"
BATCH="${BATCH:-1}"

mkdir -p "$outdir"
rm -f "$outdir/*.csv"

for b in $BACKENDS; do
    echo "using backend $b"
    for o in $OPTIMIZERS; do
        for s in $SEEDS; do
            echo "Testing optimizer $o on $PROBLEM with seed $s ..." 
            (
            set -x
            loop-explore \
              --backends "$b" \
              --search iterative \
              --optimizer "$o" \
              --operator "$OPERATOR" \
              --op-name "$PROBLEM" \
              --jobs "$JOBS" \
              --batch "$BATCH" \
              --seed $s \
              --strategy "$STRATEGY" \
              --output "$outdir/results.b$b.prob$PROBLEM.strat$STRATEGY.seed$s.csv"
            )
        done
    done
done
