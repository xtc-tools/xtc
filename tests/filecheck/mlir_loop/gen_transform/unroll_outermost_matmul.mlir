// RUN: mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<8x64xf32>,
  %B: memref<64x32xf32>,
  %C: memref<8x32xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.schedule = {
        "k",
          "i" = {"unroll" = 8},
            "j" = {"unroll" = 32},
              "j#16" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<8x64xf32>, memref<64x32xf32>)
    outs(%C : memref<8x32xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<8x64xf32> {llvm.noalias}, %arg1: memref<64x32xf32> {llvm.noalias}, %arg2: memref<8x32xf32> {llvm.noalias}) {
// CHECK-NEXT:      linalg.matmul {__node0__} ins(%arg0, %arg1 : memref<8x64xf32>, memref<64x32xf32>) outs(%arg2 : memref<8x32xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
// CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__node0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__node0__/k" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/i" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__node0__/j" : !transform.any_op
// CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_2) : (!transform.any_op) -> ()
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %1 {
// CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
// CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %1 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.match attributes {"__node0__/j"} in %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_3 {factor = 32 : i64} : !transform.any_op
// CHECK-NEXT:      %3 = transform.structured.match attributes {"__node0__/i"} in %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_1 {factor = 8 : i64} : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
