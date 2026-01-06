// RUN: mlir-loop --vectors-size 8 --no-alias --print-source-ir --print-transformed-ir %s 2>&1 | filecheck %s

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
    loop.schedule = {
      "n",
        "h",
          "w",
            "f",
              "r",
                "s",
                  "c"={"vectorize"}
     }
  }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}

// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>) outs(%arg2 : memref<1x28x28x128xf32>) attrs =  {__node0__} {
// CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK-NEXT:        %0 = arith.mulf %in, %in_0 : f32
// CHECK-NEXT:        %1 = arith.addf %out, %0 : f32
// CHECK-NEXT:        linalg.yield %1 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
// CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_super_vectorize(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op {
// CHECK-NEXT:      %0 = transform.apply_registered_pass "affine-super-vectorize" with options = {"virtual-vector-size" = 8 : i64} to %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield %0 : !transform.any_op
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__node0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__node0__/n" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/h" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__node0__/w" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__node0__/f" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__node0__/r" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "__node0__/s" : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %3 = transform.include @_super_vectorize failures(suppress) (%2) : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK:  RuntimeError: MLIR Error: NYI: non-trivial layout map

// CHECK:  // -----// IR Dump After transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      %c3 = arith.constant 3 : index
// CHECK-NEXT:      %c128 = arith.constant 128 : index
// CHECK-NEXT:      %c28 = arith.constant 28 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 30, 30, 64] [1, 1, 1, 1] : memref<1x30x30x64xf32> to memref<1x30x30x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0, 0, 0] [3, 3, 64, 128] [1, 1, 1, 1] : memref<3x3x64x128xf32> to memref<3x3x64x128xf32, strided<[24576, 8192, 128, 1]>>
// CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0, 0, 0] [1, 28, 28, 128] [1, 1, 1, 1] : memref<1x28x28x128xf32> to memref<1x28x28x128xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c28 step %c1 {
// CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %arg4, 0, 0] [1, 3, 30, 64] [1, 1, 1, 1] : memref<1x30x30x64xf32, strided<[57600, 1920, 64, 1], offset: ?>> to memref<1x3x30x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4, 0, 0] [1, 1, 28, 128] [1, 1, 1, 1] : memref<1x28x28x128xf32, strided<[100352, 3584, 128, 1], offset: ?>> to memref<1x1x28x128xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c28 step %c1 {
// CHECK-NEXT:            %subview_4 = memref.subview %subview_2[0, 0, %arg5, 0] [1, 3, 3, 64] [1, 1, 1, 1] : memref<1x3x30x64xf32, strided<[57600, 1920, 64, 1], offset: ?>> to memref<1x3x3x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:            %subview_5 = memref.subview %subview_3[0, 0, %arg5, 0] [1, 1, 1, 128] [1, 1, 1, 1] : memref<1x1x28x128xf32, strided<[100352, 3584, 128, 1], offset: ?>> to memref<1x1x1x128xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c128 step %c1 {
// CHECK-NEXT:              %subview_6 = memref.subview %subview_0[0, 0, 0, %arg6] [3, 3, 64, 1] [1, 1, 1, 1] : memref<3x3x64x128xf32, strided<[24576, 8192, 128, 1]>> to memref<3x3x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>>
// CHECK-NEXT:              %subview_7 = memref.subview %subview_5[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x128xf32, strided<[100352, 3584, 128, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:              scf.for %arg7 = %c0 to %c3 step %c1 {
// CHECK-NEXT:                %subview_8 = memref.subview %subview_4[0, %arg7, 0, 0] [1, 1, 3, 64] [1, 1, 1, 1] : memref<1x3x3x64xf32, strided<[57600, 1920, 64, 1], offset: ?>> to memref<1x1x3x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:                %subview_9 = memref.subview %subview_6[%arg7, 0, 0, 0] [1, 3, 64, 1] [1, 1, 1, 1] : memref<3x3x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>> to memref<1x3x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>>
// CHECK-NEXT:                scf.for %arg8 = %c0 to %c3 step %c1 {
// CHECK-NEXT:                  %subview_10 = memref.subview %subview_8[0, 0, %arg8, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x3x64xf32, strided<[57600, 1920, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:                  %subview_11 = memref.subview %subview_9[0, %arg8, 0, 0] [1, 1, 64, 1] [1, 1, 1, 1] : memref<1x3x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>> to memref<1x1x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>>
// CHECK-NEXT:                  affine.for %arg9 = 0 to 1 {
// CHECK-NEXT:                    affine.for %arg10 = 0 to 1 {
// CHECK-NEXT:                      affine.for %arg11 = 0 to 1 {
// CHECK-NEXT:                        affine.for %arg12 = 0 to 1 {
// CHECK-NEXT:                          affine.for %arg13 = 0 to 1 {
// CHECK-NEXT:                            affine.for %arg14 = 0 to 1 {
// CHECK-NEXT:                              affine.for %arg15 = 0 to 64 {
// CHECK-NEXT:                                %0 = affine.apply #map(%arg10, %arg13)
// CHECK-NEXT:                                %1 = affine.apply #map(%arg11, %arg14)
// CHECK-NEXT:                                %2 = affine.load %subview_10[%arg9, %0, %1, %arg15] : memref<1x1x1x64xf32, strided<[57600, 1920, 64, 1], offset: ?>>
// CHECK-NEXT:                                %3 = affine.load %subview_11[%arg13, %arg14, %arg15, %arg12] : memref<1x1x64x1xf32, strided<[24576, 8192, 128, 1], offset: ?>>
// CHECK-NEXT:                                %4 = affine.load %subview_7[%arg9, %arg10, %arg11, %arg12] : memref<1x1x1x1xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:                                %5 = arith.mulf %2, %3 : f32
// CHECK-NEXT:                                %6 = arith.addf %4, %5 : f32
// CHECK-NEXT:                                affine.store %6, %subview_7[%arg9, %arg10, %arg11, %arg12] : memref<1x1x1x1xf32, strided<[100352, 3584, 128, 1], offset: ?>>
// CHECK-NEXT:                              }
// CHECK-NEXT:                            }
// CHECK-NEXT:                          }
// CHECK-NEXT:                        }
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                } {"__node0__/s"}
// CHECK-NEXT:              } {"__node0__/r"}
// CHECK-NEXT:            } {"__node0__/f"}
// CHECK-NEXT:          } {"__node0__/w"}
// CHECK-NEXT:        } {"__node0__/h"}
// CHECK-NEXT:      } {"__node0__/n"}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
