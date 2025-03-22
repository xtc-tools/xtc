// RUN: mlir-loop %s --evaluate --no-alias

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32

  %zero_i32 = arith.constant 0 : i32
  %sixfour_i32 = arith.constant 64 : i32
  %one_index = arith.constant 1: index
  %size = memref.dim %C, %one_index : memref<256x256xf32>
  %cast = arith.index_cast %size: index to i32
  %mod = arith.remui %cast,%sixfour_i32: i32
  %divisibility = arith.cmpi "eq", %mod, %zero_i32 : i32

  scf.if %divisibility {
    linalg.fill
        {
          loop.dims = ["i","j"],
          loop.tiles_names = {"i" = ["i1"], "j" = ["j1"]},
          loop.tiles_sizes = {i1 = 1, j1 = 64},
          loop.interchange = ["i","j","i1","j1"],
          loop.vectorize = ["j1"]
      }
      ins(%cst : f32)
      outs(%C : memref<256x256xf32>)
    linalg.matmul
      {
        loop.dims = ["i","j","k"],
        loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
        loop.tiles_sizes = {i1 = 1, j1 = 64, k1 = 8},
        loop.interchange = ["i","j","k","i1","k1","j1"],
        loop.vectorize = ["j1"]
      }
      ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
      outs(%C : memref<256x256xf32>)
    scf.yield
  } else {
    linalg.fill
      ins(%cst : f32)
      outs(%C : memref<256x256xf32>)
    linalg.matmul
      ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
      outs(%C : memref<256x256xf32>)
    scf.yield
  }
  return
}
