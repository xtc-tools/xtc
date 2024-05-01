// ~/bin/llvm-xdsl/bin/mlir-opt test.mlir --transform-interpreter --func-bufferize --buffer-deallocation --test-transform-dialect-erase-schedule --convert-scf-to-cf --canonicalize --convert-vector-to-llvm=enable-x86vector --test-lower-to-llvm


// -----// IR Dump Before InterpreterPass (transform-interpreter) //----- //
module attributes {transform.with_named_sequence} {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func @payload0(%arg0: tensor<512x1024xf32>, %arg1: tensor<1024x128xf32>) -> tensor<512x128xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512x128xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x1024xf32>, tensor<1024x128xf32>) outs(%cst : tensor<512x128xf32>) -> tensor<512x128xf32>
    return %0 : tensor<512x128xf32>
  }
  func.func @main() {
    %cst = arith.constant dense<1.000000e+00> : tensor<512x1024xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1024x128xf32>
    %0 = call @rtclock() : () -> f64
    %1 = call @payload0(%cst, %cst_0) : (tensor<512x1024xf32>, tensor<1024x128xf32>) -> tensor<512x128xf32>
    %2 = call @rtclock() : () -> f64
    %3 = arith.subf %2, %0 : f64
    call @printF64(%3) : (f64) -> ()
    return
  }
  transform.named_sequence @seq1(%arg0: !transform.any_op {transform.consumed}) {
    %tiled_op, %forall_op = transform.structured.tile_using_forall %arg0 tile_sizes [8, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op[0, 0, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op[0, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0[1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2[0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.annotate %loops_3 "k1" : !transform.any_op
    // transform.loop.unroll %loops_5 {factor = 4 : i64} : !transform.any_op

    %o0 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    
    transform.apply_patterns to %o0 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } {apply_cse} : !transform.any_op

    %0 = transform.structured.vectorize_children_and_apply_patterns %o0
    : (!transform.any_op) -> !transform.any_op


    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.fold_memref_alias_ops
      // transform.apply_patterns.vector.rank_reducing_subview_patterns
      transform.apply_patterns.canonicalization
   } {apply_cse} : !transform.any_op

   %1 = transform.structured.hoist_redundant_vector_transfers %0
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %1 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
    // transform.apply_patterns to %1 {
      // transform.apply_patterns.vector.lower_transfer max_transfer_rank = 99
      // transform.apply_patterns.canonicalization
    // } {apply_cse} : !transform.any_op

    transform.yield 
  }
  
  transform.named_sequence @seq0(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      %2 = transform.param.constant 512 : i64 -> !transform.param<i64>
      %3 = transform.param.constant 128 : i64 -> !transform.param<i64>
      %4 = transform.param.constant 1024 : i64 -> !transform.param<i64>
      %5 = transform.merge_handles %2, %3, %4 : !transform.param<i64>
      transform.match.param.cmpi eq %1, %5 : !transform.param<i64>
      transform.match.operation_name %arg1 ["linalg.matmul"] : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.bufferization.one_shot_bufferize %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 
    // %1 = transform.foreach_match in %arg0 
        @seq0 -> @seq1 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}


