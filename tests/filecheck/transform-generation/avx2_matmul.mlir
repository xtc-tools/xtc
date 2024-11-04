// RUN: mlir-loop %s --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = {"i"=256,"j"=256},
        loop.parallel_dims = ["i","j"],
        loop.reduction_dims = [],
        loop.tiles_names = {"j" = ["j1"]},
        loop.tiles_sizes = {j1 = 8},
        loop.interchange = ["i","j","j1"],
        loop.vectorize = ["j1"],
        loop.parallelize = ["i"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = {"i"=256,"j"=256,"k"=512},
      loop.parallel_dims = ["i","j"],
      loop.reduction_dims = ["k"],
      loop.tiles_names = {"j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {k1 = 8}
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill {id0, loop.dims = {i = 256 : i64, j = 256 : i64}, loop.interchange = ["i", "j", "j1"], loop.parallel_dims = ["i", "j"], loop.parallelize = ["i"], loop.reduction_dims = [], loop.tiles_names = {j = ["j1"]}, loop.tiles_sizes = {j1 = 8 : i64}, loop.vectorize = ["j1"]} ins(%cst : f32) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      linalg.matmul {id1, loop.dims = {i = 256 : i64, j = 256 : i64, k = 512 : i64}, loop.interchange = ["i", "j", "k", "k1", "j1"], loop.parallel_dims = ["i", "j"], loop.reduction_dims = ["k"], loop.tiles_names = {j = ["j1"], k = ["k1"]}, loop.tiles_sizes = {j1 = 64 : i64, k1 = 8 : i64}, loop.unroll = {k1 = 8 : i64}, loop.vectorize = ["j1"]} ins(%arg0, %arg1 : memref<256x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {id0} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %forall_op "id0_i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op tile_sizes [0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "id0_j" : !transform.any_op
// CHECK-NEXT:      %1 = transform.structured.match attributes {id1} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "id1_i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "id1_j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "id1_k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "id1_k1" : !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.match attributes {id1_k1} in %loops_1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %2 {factor = 8 : i64} : !transform.any_op
// CHECK-NEXT:      %3 = transform.get_parent_op %loops_1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %4 = transform.structured.vectorize_children_and_apply_patterns %3 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %4 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  