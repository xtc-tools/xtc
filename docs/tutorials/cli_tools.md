# CLI Tools

## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    loop-explore --debug --search random --trials 100 --output results.random.csv

Explore a specific node of a serialized multi-node graph:

    loop-explore --trials 1 --graph-file tests/graphs/resnet18_12_pad_conv_relu.graph.yaml --node conv

The selected node also determines the operation dimensions used to build the
search space.

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

### Iterative Search

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
operators. It supports the following transformations:
+ Tiling
+ Loop interchange
+ Vectorization
+ Unrolling
+ Parallelize
+ Producer and Consumer Fusion (when using tensor dialect)

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

When using operator fusion, the tensor dialect needs to be used.
Below is an example of fusing a relu into a matmul.

```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
  func.func @matmul_relu(%arg0: tensor<4x512xf32>, %arg1: tensor<512x32xf32>, %arg2: memref<4x32xf32>) {
    %0 = tensor.empty() : tensor<4x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = linalg.matmul
        {
          loop.dims = ["I","J","K"],
          loop.schedule = {
            "I",
              "J",
                  "I#2",
                      "J#16" = {"fuse_consumer"},
                      "K"
          }
        }
    	ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) 
    	outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>

    %3 = tensor.empty() : tensor<4x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.generic {
		indexing_maps = [#map, #map1, #map],
		iterator_types = ["parallel", "parallel"]} 
		ins(%2, %cst_0 : tensor<4x32xf32>, f32) 
		outs(%3 : tensor<4x32xf32>) 
		attrs =  {__xtc_id_relu_} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.maximumf %in, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<4x32xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
    return
  }
```
