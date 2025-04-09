// RUN: mlir-opt --transform-interpreter --test-transform-dialect-erase-schedule %s | filecheck %s

module attributes {transform.with_named_sequence} {
  func.func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32> {llvm.noalias}, %filter: memref<1x3x8xf32> {llvm.noalias}, %output: memref<4x2x8xf32> {llvm.noalias}) {
    linalg.generic {indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 3 + d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
    ], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%input, %filter : memref<4x6x3xf32>, memref<1x3x8xf32>) outs(%output : memref<4x2x8xf32>) attrs =  {__id0__} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %2 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_contraction
    } : !transform.any_op
    transform.yield 
  }
}

// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
