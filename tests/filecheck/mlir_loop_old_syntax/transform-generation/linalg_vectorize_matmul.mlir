// RUN: mlir-loop %s --no-alias --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<16x4xf32>,
  %B: memref<4x8xf32>,
  %C: memref<16x8xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<16x8xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.vectorize = ["i"]
    }
    ins(%A, %B : memref<16x4xf32>, memref<4x8xf32>)
    outs(%C : memref<16x8xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<16x4xf32> {llvm.noalias}, %arg1: memref<4x8xf32> {llvm.noalias}, %arg2: memref<16x8xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg2 : memref<16x8xf32>)
// CHECK-NEXT:      linalg.matmul {__id0__} ins(%arg0, %arg1 : memref<16x4xf32>, memref<4x8xf32>) outs(%arg2 : memref<16x8xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
