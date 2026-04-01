# CLI Tools

## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    loop-explore --debug --search random --trials 100 --output results.random.csv

Use exploration script, for instance on input data generated on some tvm search (3D tiling + permutations), 2054 points here:

    time -p loop-explore --debug --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output data/results.mm06-tile4d.csv
    ...
    2054/2054 [55:54,  1.63s/it]
    real 3356 secs

Use exhaustive search on a tiling strategy limited to tile4d + only vectorized tilings (450 points):

    # TVM backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends tvm --output results.mm06-tile4dv-tvm.csv
    450/450 [24:04,  3.21s/it]
    real 1444.50

    # MLIR backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends mlir --output results.mm06-tile4dv-mlir.csv
    450/450 [22:34<00:00,  3.01s/it]
    real 1355.98

    # JIR backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends jir --output results.mm06-tile4dv-jir.csv
    450/450 [22:30<00:00,  3.00s/it]
    real 1352.37

Resume interrupted exploration while keeping reproducibility metadata:

    # Start a run
    loop-explore --search random --trials 200 --output results.random.csv

    # Resume the same output (skips already recorded samples)
    loop-explore --search random --trials 200 --output results.random.csv --resume

    # Append regardless of duplicates
    loop-explore --search random --trials 200 --output results.random.csv --append

Each run also writes `results.random.csv.meta.json` with the command arguments,
Python/runtime information, and git commit hash to simplify result provenance.

Test a single tiling:

    # Dumps and execute MLIR tiling
    loop-explore --dump --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 1.89 msecs, peak perf: 26.38%
    # Execute on all backends
    loop-explore --backends tvm mlir jir --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 0.61 msecs, peak perf: 82.08%

### Display

Result of exploration and display in `data/results.mm06-tile7d-all.svg` were generated with:

    loop-explore --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile7d  --search random --trials 1000 --output data/results.mm06-tile7d-all.csv
    loop-display --title 'Tile7D tiling strategy on 1000 samples for 256x256x512 matmul' data/results.mm06-tile7d-all.csv:tvm:X:peak:tvm data/results.mm06-tile7d-all.csv:mlir:X:peak:mlir data/results.mm06-tile7d-all.csv:jir:X:peak:jir --output data/results.mm06-tile7d-all.svg

Comparative performance distribution on tile4dv tilings in `data/mlir_results.mm06-tile4dv-all.svg` were generated with:

    loop-explore --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile4dv  --search exhaustive --output data/results.mm06-tile4dv-all.csv
    loop-display --title "Tile4DV tiling strategy exhaustive for 256x256x512 vectorized matmul" data/results.mm06-tile4dv-all.csv:tvm:X:peak:tvm data/results.mm06-tile4dv-all.csv:mlir:X:peak:mlir data/results.mm06-tile4dv-all.csv:jir:X:peak:jir --output data/results.mm06-tile4dv-all.svg

# mlir-loop: high-level scheduling specifications in MLIR

The ```mlir-loop``` tool provides a high-level syntax for
controlling the scheduling of MLIR linear algebra (```linalg```)
operators. For now, it only applies at ```memref``` level
(not ```tensor```) and supports the following transformations:
+ Tiling
+ Loop interchange
+ Vectorization
+ Unrolling

See the code below. For the simplicity of the example, it is a
single operator function, but the tool accepts multiple operator
functions.

```
func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I" = {"parallelize"},
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
```

Under the hood, this declarative "loop" attributes dialect is
translated into the corresponding MLIR ```transform``` dialect
command sequence through XTC API calls.

## Iterative Search

An iterative search can take advantage of a model to try to learn better tile sizes as loop explore is ran.
Setting the batch determines the number of guesses the model has per iteration.

    loop-explore --search iterative --batch 5 --trials 100 --strategy tile_goto --output results.random.csv

You can specify an optimizer preset to determine how aggressive the model converges.

    loop-explore --backends mlir --batch 5 --search iterative --optimizer random-forest-explore --operator conv2d --op-name AlexNet_02 --strategy tile_ppwrprp_vr --output output.csv

To specify your own model parameters you can pass a yaml file.

For example lets say you have config.yaml containing:

```
batch_candidates: 5000
beta: 2.5
alpha: 0.7
update_first: null
update_period: null
n_estimators: 300
max_depth: 10
min_samples_leaf: 4
max_features: 0.8
```

You can use those parameters to set the model behavior

    loop-explore --backends mlir --batch 5 --search iterative --optimizer-config config.yaml --strategy tile_goto --output output.csv

