// RUN: mlir-loop --old-syntax %s --no-alias --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<250x480xf32>,
  %B: memref<480x640xf32>,
  %C: memref<250x640xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<250x640xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {
        "i" = {"i1" = 5,  "i2" = 1},
        "j" = {"j1" = 64, "j2" = 32},
        "k" = {"k1" = 96, "k2" = 8}},
      loop.interchange = ["i","j","k","i1","k1","j1","i2","k2","j2"],
      loop.vectorize = ["j2"],
      loop.unroll = {"i2" = 1, "k2" = 8}
    }
    ins(%A, %B : memref<250x480xf32>, memref<480x640xf32>)
    outs(%C : memref<250x640xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<250x480xf32> {llvm.noalias}, %arg1: memref<480x640xf32> {llvm.noalias}, %arg2: memref<250x640xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg2 : memref<250x640xf32>)
// CHECK-NEXT:      linalg.matmul {__id0__} ins(%arg0, %arg1 : memref<250x480xf32>, memref<480x640xf32>) outs(%arg2 : memref<250x640xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [5, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__id0__i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__id0__j" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 96] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__id0__k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__id0__i1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__id0__k1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "__id0__j1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_11 "__id0__i2" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_13 "__id0__k2" : !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_13 {factor = 8 : i64} : !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_11 {factor = 1 : i64} : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
