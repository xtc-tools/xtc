// RUN: mlir-loop %s --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    {
      loop.dims = {"i"=512,"j"=128},
      loop.parallel_dims = ["i","j"],
      loop.reduction_dims = [],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"]},
      loop.tiles_sizes = {i1 = 4, j1 = 64},
      loop.interchange = ["i","j","i1","j1"],
      loop.vectorize = ["j1"],
      loop.parallelize = ["i"],
      loop.unroll = {i1 = 4}
    }
    ins(%cst : f32)
    outs(%C : memref<512x128xf32>)
  linalg.matmul
    {
      loop.dims = {"i"=512,"j"=128,"k"=1024},
      loop.parallel_dims = ["i","j"],
      loop.reduction_dims = ["k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 4, j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","i1","j1"],
      loop.vectorize = ["j1"],
      loop.parallelize = ["i"],
      loop.unroll = {i1 = 4, k1 = 8},
      loop.add_attributes = ["JoeDassin"]
    }
    ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>)
    outs(%C : memref<512x128xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<512x1024xf32> {llvm.noalias}, %arg1: memref<1024x128xf32> {llvm.noalias}, %arg2: memref<512x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill {id0, loop.dims = {i = 512 : i64, j = 128 : i64}, loop.interchange = ["i", "j", "i1", "j1"], loop.parallel_dims = ["i", "j"], loop.parallelize = ["i"], loop.reduction_dims = [], loop.tiles_names = {i = ["i1"], j = ["j1"]}, loop.tiles_sizes = {i1 = 4 : i64, j1 = 64 : i64}, loop.unroll = {i1 = 4 : i64}, loop.vectorize = ["j1"]} ins(%cst : f32) outs(%arg2 : memref<512x128xf32>)
// CHECK-NEXT:      linalg.matmul {id1, loop.add_attributes = ["JoeDassin"], loop.dims = {i = 512 : i64, j = 128 : i64, k = 1024 : i64}, loop.interchange = ["i", "j", "k", "k1", "i1", "j1"], loop.parallel_dims = ["i", "j"], loop.parallelize = ["i"], loop.reduction_dims = ["k"], loop.tiles_names = {i = ["i1"], j = ["j1"], k = ["k1"]}, loop.tiles_sizes = {i1 = 4 : i64, j1 = 64 : i64, k1 = 8 : i64}, loop.unroll = {i1 = 4 : i64, k1 = 8 : i64}, loop.vectorize = ["j1"]} ins(%arg0, %arg1 : memref<512x1024xf32>, memref<1024x128xf32>) outs(%arg2 : memref<512x128xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {id0} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %forall_op "id0_i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op tile_sizes [0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "id0_j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "id0_i1" : !transform.any_op
// CHECK-NEXT:      %1 = transform.structured.match attributes {id1} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %1 tile_sizes [4, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %forall_op_3 "id1_i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op_2 tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "id1_j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "id1_k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "id1_k1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_11 "id1_i1" : !transform.any_op
// CHECK-NEXT:      transform.annotate %forall_op_3 "JoeDassin" : !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.match attributes {id0_i1} in %forall_op_3 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %2 {factor = 4 : i64} : !transform.any_op
// CHECK-NEXT:      %3 = transform.structured.match attributes {id1_i1} in %forall_op_3 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %3 {factor = 4 : i64} : !transform.any_op
// CHECK-NEXT:      %4 = transform.structured.match attributes {id1_k1} in %forall_op_3 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %4 {factor = 8 : i64} : !transform.any_op
// CHECK-NEXT:      %5 = transform.get_parent_op %forall_op_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %6 = transform.structured.vectorize_children_and_apply_patterns %5 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %6 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  