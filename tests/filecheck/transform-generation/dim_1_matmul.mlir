// RUN: mlir-loop %s --no-alias --always-vectorize --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = {"i"=256,"j"=256,"k"=512},
      loop.parallel_dims = ["i","j"],
      loop.reduction_dims = ["k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 1, j1 = 16, k1 = 32},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {"k1" = 32,"i1" = 1}
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      linalg.matmul {__id0__, loop.dims = {i = 256 : i64, j = 256 : i64, k = 512 : i64}, loop.interchange = ["i", "j", "k", "i1", "k1", "j1"], loop.parallel_dims = ["i", "j"], loop.reduction_dims = ["k"], loop.tiles_names = {i = ["i1"], j = ["j1"], k = ["k1"]}, loop.tiles_sizes = {i1 = 1 : i64, j1 = 16 : i64, k1 = 32 : i64}, loop.unroll = {i1 = 1 : i64, k1 = 32 : i64}, loop.vectorize = ["j1"]} ins(%arg0, %arg1 : memref<256x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op = transform.structured.tile_using_for %0 tile_sizes [0, 0, 0] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__id0__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_1, %loops_2 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_2 "__id0__k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_3, %loops_4 = transform.structured.tile_using_for %tiled_linalg_op_1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_4 "__id0__i1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_5, %loops_6 = transform.structured.tile_using_for %tiled_linalg_op_3 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_6 "__id0__k1" : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      %3 = transform.structured.match attributes {__id0__k1} in %2 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %3 {factor = 32 : i64} : !transform.any_op
// CHECK-NEXT:      %4 = transform.structured.match attributes {__id0__i1} in %2 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %4 {factor = 1 : i64} : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  