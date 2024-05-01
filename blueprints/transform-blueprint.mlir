module attributes {transform.with_named_sequence} {

func.func @source(
  %A: tensor<512x128xf32>,
  %B: tensor<128x1024xf32>,
  %C: tensor<512x1024xf32>
) -> (tensor<512x1024xf32>) {
  %D = linalg.matmul ins (%A,%B: tensor<512x128xf32>,tensor<128x1024xf32>)
                     outs(%C: tensor<512x1024xf32>) -> tensor<512x1024xf32>
  func.return %D: tensor<512x1024xf32>
}

// Product of:
// mlir-opt matmul.mlir --linalg-generalize-named-ops
func.func @target(
  %A: tensor<512x128xf32>,
  %B: tensor<128x1024xf32>,
  %C: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction"
    ] } ins (%A, %B : tensor<512x128xf32>, tensor<128x1024xf32>)
        outs(%C : tensor<512x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<512x1024xf32>
  return %res : tensor<512x1024xf32>
}

transform.named_sequence @print_match(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "matched op" : !transform.any_op
    transform.yield
  }

transform.named_sequence @schedule(%mm0: !transform.any_op{transform.consumed}) {
  %mm1, %i0 = transform.structured.tile_using_forall %mm0 num_threads [8, 0]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %mm2, %j1 = transform.structured.tile_using_for %mm1 [0,0,4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %mm3, %k1 = transform.structured.tile_using_for %mm2 [0,4,0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %mm4, %i1 = transform.structured.tile_using_for %mm3 [1,0,0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // transform.structured.vectorize %mm4 : !transform.any_op
  // transform.loop.unroll %i1 { factor = 64 } : !transform.any_op
  // transform.loop.unroll %i1 { full } : !transform.any_op
  transform.yield
}

transform.named_sequence @match_generic_dims_and_body(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      %i = transform.param.constant 512 : i64 -> !transform.param<i64>
      %j = transform.param.constant 1024 : i64 -> !transform.param<i64>
      %k = transform.param.constant 128 : i64 -> !transform.param<i64>
      %2 = transform.merge_handles %i, %j, %k : !transform.param<i64>
      transform.match.param.cmpi eq %1, %2 : !transform.param<i64>
      transform.match.operation_name %arg1 ["linalg.generic"] : !transform.any_op
      transform.match.structured.body %arg1 { contraction = ["arith.mulf",  
"arith.addf"] } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    
    transform.yield %0 : !transform.any_op
  }

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    // %0 = transform.bufferization.one_shot_bufferize %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %arg0 @match_generic_dims_and_body -> @schedule : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
