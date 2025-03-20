// RUN: mlir-loop %s --no-alias --always-vectorize --print-lowered-ir 2>&1 | grep omp | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = ["i","j"],
        loop.tiles_names = {"j" = ["j1"]},
        loop.tiles_sizes = {j1 = 8},
        loop.interchange = ["i","j","j1"],
        loop.vectorize = ["j1"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)

  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","j1"],
      loop.vectorize = ["j1"],
      loop.parallelize = ["i"]
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:           omp.parallel {
// CHECK-NEXT:        omp.wsloop {
// CHECK-NEXT:          omp.loop_nest (%arg3) : i64 = (%4) to (%5) step (%6) {
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:        omp.terminator
