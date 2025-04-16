// RUN: mlir-loop --old-syntax %s --no-alias --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %I: memref<1x30x30x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x28x128xf32>
) {
  linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
      ],
      iterator_types = [
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "reduction",
        "reduction",
        "reduction"
      ]
  }
  ins (%I, %K : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>)
  outs(%O : memref<1x28x28x128xf32>)
  attrs = {
    loop.dims = ["n","h","w","f","r","s","c"],
    loop.vectorize = ["h"]
  }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}

// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
