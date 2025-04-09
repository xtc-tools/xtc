// RUN: mlir-opt --transform-interpreter --test-transform-dialect-erase-schedule %s | filecheck %s

module attributes {transform.with_named_sequence} {
func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      __id0__
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
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
