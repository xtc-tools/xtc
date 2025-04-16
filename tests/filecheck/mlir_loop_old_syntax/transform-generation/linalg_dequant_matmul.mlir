// RUN: mlir-loop %s --no-alias --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %Bquant: memref<512x256xi8>,
  %C: memref<256x256xf32>
) {

  // Here, we initialize the result memref
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = ["i","j"],
        loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 64}},
        loop.interchange = ["i","j","i1","j1"],
        loop.vectorize = ["j1"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
    
  // The following describes a dequantization operation
  %B = memref.alloc(): memref<512x256xf32>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0,d1) -> (d0,d1)>,
      affine_map<(d0,d1) -> (d0,d1)>
    ],
    iterator_types = ["parallel","parallel"]
  } ins(%Bquant : memref<512x256xi8>) outs(%B : memref<512x256xf32>)
  attrs = {
      loop.dims = ["j","k"],
      loop.tiles = {"j" = {"j1" = 64}, "k" = {"k1" = 8}},
      loop.interchange = ["k","j","k1","j1"],
      loop.vectorize = ["j1"]
  }
  {
  ^0(%2 : i8, %3 : f32):
    %4 = arith.constant 1.000000e-01 : f32
    %5 = arith.constant 1 : i32
    %6 = arith.sitofp %2 : i8 to f32
    %7 = arith.sitofp %5 : i32 to f32
    %8 = arith.subf %6, %7 : f32
    %9 = arith.mulf %8, %4 : f32
    linalg.yield %9 : f32
  }

  // The matmul itself
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 64}, "k" = {"k1" = 8}},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"]
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
    
  memref.dealloc %B: memref<512x256xf32>
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xi8> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill {__id0__} ins(%cst : f32) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      %alloc = memref.alloc() : memref<512x256xf32>
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<512x256xi8>) outs(%alloc : memref<512x256xf32>) attrs =  {__id1__} {
// CHECK-NEXT:      ^bb0(%in: i8, %out: f32):
// CHECK-NEXT:        %cst_0 = arith.constant 1.000000e-01 : f32
// CHECK-NEXT:        %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:        %0 = arith.sitofp %in : i8 to f32
// CHECK-NEXT:        %1 = arith.sitofp %c1_i32 : i32 to f32
// CHECK-NEXT:        %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:        %3 = arith.mulf %2, %cst_0 : f32
// CHECK-NEXT:        linalg.yield %3 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      linalg.matmul {__id2__} ins(%arg0, %alloc : memref<256x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      memref.dealloc %alloc : memref<512x256xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__id0__i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__id0__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__id0__i1" : !transform.any_op
// CHECK-NEXT:      %1 = transform.structured.match attributes {__id1__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %1 tile_sizes [0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__id1__k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__id1__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "__id1__k1" : !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.match attributes {__id2__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_11 "__id2__i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_13 "__id2__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_15 "__id2__k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_17 "__id2__i1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_19 "__id2__k1" : !transform.any_op
// CHECK-NEXT:      %3 = transform.get_parent_op %loops_11 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %4 = transform.structured.vectorize_children_and_apply_patterns %3 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %4 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
