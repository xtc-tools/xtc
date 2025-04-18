#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"

NOSHOW="${NOSHOW:-}"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dvr tile4dv " \
          "tile7d tile7dv tile7dvr " \
          "tile8d tile8dv tile8dvr" \
          "}"

opts=""
[ -z "$NOSHOW" ] || opts="$opts --no-show"



for s in $STRATEGIES; do
    # match file = results.b<backend>.s<strategy>.o<level>.csv -> file:backend:X:peak
    args="$(ls "$outdir"/results.b*.s$s.o*.csv 2>/dev/null | sed 's|/\(\([^.]*\)\.b\([^.]*\)\.s\([^.]*\)\.o\([^.]*\)\.csv\)|/\1:\3-O\5:X:peak|' || true)"
    [ -n "$args" ] || continue
    (set -x && loop-display $opts --no-cdf --title "Opt Level performance for strategy $s" --output "$outdir/results.$s.png" $args)
done
