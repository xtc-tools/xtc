// RUN: mlir-opt --transform-interpreter --test-transform-dialect-erase-schedule %s | filecheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module attributes {transform.with_named_sequence} {
  func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>) outs(%arg2 : memref<1x28x28x128xf32>) attrs =  {__id0__} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
    // transform.structured.vectorize %0: !transform.any_op
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
