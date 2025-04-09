// RUN: mlir-opt --transform-interpreter --test-transform-dialect-erase-schedule %s | filecheck %s

module attributes {transform.with_named_sequence} {
  func.func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32> {llvm.noalias}, %filter: memref<1x3x8xf32> {llvm.noalias}, %output: memref<4x2x8xf32> {llvm.noalias}) {
    linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>, __id0__}
      ins(%input, %filter : memref<4x6x3xf32>, memref<1x3x8xf32>)
      outs(%output : memref<4x2x8xf32>)
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
