#!/usr/bin/env bash
set -eu o pipefail

dir="$(dirname "$0")"

date="${DATE:-$(date +"%Y%m%d_%H%M%S")}"
benchs="${BENCHS:-$(cat default_names.txt)}"
strategies="${STRATEGIES- ansorcvec.yml ttile.yml constraintvec.yml}"
# seeds="${SEEDS:-$(seq 1 5)}"
cores="${CORES:-1}"
dry="${DRY:-}"
search="${SEARCH:-random}"
run=""
[ "$dry" = "" ] || run=echo

jobs=8
cpus="$jobs"
[ "$cpus" -ge "$cores" ] || cpus="$cores"
low_cpu=27
high_cpu=$((low_cpu + 2*(cpus-1)))
cpus="$(echo -n "$(seq $low_cpu 2 $high_cpu)" | tr '\n' ',')"
taskset="taskset -c $cpus"

flops="67.2e9"
trials="${TRIALS:-2048}"

host="$(hostname -s)"
results_dir="results/$host/$date"
mkdir -p "$results_dir"


# echo "Will run seeds: $seeds..." >&2
echo "Will run benchs: $benchs..." >&2
echo "Will apply strategies: $strategies..." >&2

# session_tag="run_all_ics/$host/$USER/$date"
for bench in $benchs; do
    for strategy in $strategies; do
        echo "Running: $bench/$strategy..." >&2
        (set -x; $run $taskset loop-explore --operator matmul --op-name "$bench" --descript "$strategy" --functions "ilp.py" --search "$search" --backends tvm --trials "$trials" --threads "$cores" --output "$results_dir/$bench-$strategy.csv" --batch 64 --jobs "$jobs" --peak-flops "$flops")
        echo "Done: date $date, bench $bench, strategy $strategy, cores $cores." >&2
    done
done
echo "Done: date $date, cores $cores." >&2
