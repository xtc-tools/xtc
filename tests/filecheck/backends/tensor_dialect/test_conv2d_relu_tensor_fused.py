# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="conv2d_nhwc_mini") as gb:
    c = O.conv2d(a, b, stride=(SH, SW), name="O")
    O.relu(c, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="O")
sch.fuse_consumer_at("f")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_relu_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_mini(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%0 : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x10x10x3xf32>, tensor<3x3x3x16xf32>) outs(%1 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.mulf %in, %in_1 fastmath<fast> : f32
# CHECK-NEXT:        %6 = arith.addf %out, %5 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %6 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %3 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_0 : tensor<1x8x8x16xf32>, f32) outs(%3 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.maximumf %in, %in_1 : f32
# CHECK-NEXT:        linalg.yield %5 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_O_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops:4 = transform.xtc.fuse_consumer %2 into %loops_7, %loops_9, %loops_11, %loops_13 : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops#0 "./b" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#1 "./h" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#2 "./w" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#3 "./f" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_mini(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %0) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        %5 = scf.for %arg5 = %c0_5 to %c8 step %c1_6 iter_args(%arg6 = %extracted_slice) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %c0_8 = arith.constant 0 : index
# CHECK-NEXT:          %c8_9 = arith.constant 8 : index
# CHECK-NEXT:          %c1_10 = arith.constant 1 : index
# CHECK-NEXT:          %6 = scf.for %arg7 = %c0_8 to %c8_9 step %c1_10 iter_args(%arg8 = %extracted_slice_7) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:            %extracted_slice_12 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_14 = arith.constant 1 : index
# CHECK-NEXT:            %7 = scf.for %arg9 = %c0_13 to %c16 step %c1_14 iter_args(%arg10 = %extracted_slice_12) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_16 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %8 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_16 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %inserted_slice_17 = tensor.insert_slice %8 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_17 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %inserted_slice_15 = tensor.insert_slice %7 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_15 : tensor<1x1x8x16xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %inserted_slice_11 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_11 : tensor<1x8x8x16xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %c0_1 = arith.constant 0 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3:2 = scf.for %arg3 = %c0_1 to %c1_2 step %c1_3 iter_args(%arg4 = %1, %arg5 = %2) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 10, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x10x10x3xf32>
# CHECK-NEXT:        %extracted_slice_5 = tensor.extract_slice %arg1[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_8 = arith.constant 1 : index
# CHECK-NEXT:        %extracted_slice_9 = tensor.extract_slice %arg5[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %5:2 = scf.for %arg6 = %c0_7 to %c8 step %c1_8 iter_args(%arg7 = %extracted_slice_6, %arg8 = %extracted_slice_9) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
# CHECK-NEXT:          %extracted_slice_11 = tensor.extract_slice %extracted_slice[0, %arg6, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x3x10x3xf32>
# CHECK-NEXT:          %extracted_slice_12 = tensor.extract_slice %extracted_slice_5[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:          %extracted_slice_13 = tensor.extract_slice %arg7[0, %arg6, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %c0_14 = arith.constant 0 : index
# CHECK-NEXT:          %c8_15 = arith.constant 8 : index
# CHECK-NEXT:          %c1_16 = arith.constant 1 : index
# CHECK-NEXT:          %extracted_slice_17 = tensor.extract_slice %arg8[0, %arg6, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %7:2 = scf.for %arg9 = %c0_14 to %c8_15 step %c1_16 iter_args(%arg10 = %extracted_slice_13, %arg11 = %extracted_slice_17) -> (tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>) {
# CHECK-NEXT:            %extracted_slice_20 = tensor.extract_slice %extracted_slice_11[0, 0, %arg9, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x10x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:            %extracted_slice_21 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:            %extracted_slice_22 = tensor.extract_slice %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_23 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_24 = arith.constant 1 : index
# CHECK-NEXT:            %extracted_slice_25 = tensor.extract_slice %arg11[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %9:2 = scf.for %arg12 = %c0_23 to %c16 step %c1_24 iter_args(%arg13 = %extracted_slice_22, %arg14 = %extracted_slice_25) -> (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_28 = tensor.extract_slice %extracted_slice_20[0, 0, 0, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:              %extracted_slice_29 = tensor.extract_slice %extracted_slice_21[0, 0, 0, %arg12] [3, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x1xf32>
# CHECK-NEXT:              %extracted_slice_30 = tensor.extract_slice %arg13[0, 0, 0, %arg12] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %c0_31 = arith.constant 0 : index
# CHECK-NEXT:              %c3 = arith.constant 3 : index
# CHECK-NEXT:              %c1_32 = arith.constant 1 : index
# CHECK-NEXT:              %11 = scf.for %arg15 = %c0_31 to %c3 step %c1_32 iter_args(%arg16 = %extracted_slice_30) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                %extracted_slice_36 = tensor.extract_slice %extracted_slice_28[0, %arg15, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x1x3x3xf32>
# CHECK-NEXT:                %extracted_slice_37 = tensor.extract_slice %extracted_slice_29[%arg15, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x1xf32> to tensor<1x3x3x1xf32>
# CHECK-NEXT:                %extracted_slice_38 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                %c0_39 = arith.constant 0 : index
# CHECK-NEXT:                %c3_40 = arith.constant 3 : index
# CHECK-NEXT:                %c1_41 = arith.constant 1 : index
# CHECK-NEXT:                %13 = scf.for %arg17 = %c0_39 to %c3_40 step %c1_41 iter_args(%arg18 = %extracted_slice_38) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                  %extracted_slice_43 = tensor.extract_slice %extracted_slice_36[0, 0, %arg17, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x3x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                  %extracted_slice_44 = tensor.extract_slice %extracted_slice_37[0, %arg17, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                  %extracted_slice_45 = tensor.extract_slice %arg18[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %c0_46 = arith.constant 0 : index
# CHECK-NEXT:                  %c3_47 = arith.constant 3 : index
# CHECK-NEXT:                  %c1_48 = arith.constant 1 : index
# CHECK-NEXT:                  %14 = scf.for %arg19 = %c0_46 to %c3_47 step %c1_48 iter_args(%arg20 = %extracted_slice_45) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                    %extracted_slice_50 = tensor.extract_slice %extracted_slice_43[0, 0, 0, %arg19] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %extracted_slice_51 = tensor.extract_slice %extracted_slice_44[0, 0, %arg19, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %extracted_slice_52 = tensor.extract_slice %arg20[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %15 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_50, %extracted_slice_51 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_52 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                    ^bb0(%in: f32, %in_54: f32, %out: f32):
# CHECK-NEXT:                      %16 = arith.mulf %in, %in_54 fastmath<fast> : f32
# CHECK-NEXT:                      %17 = arith.addf %out, %16 fastmath<fast> : f32
# CHECK-NEXT:                      linalg.yield %17 : f32
# CHECK-NEXT:                    } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %inserted_slice_53 = tensor.insert_slice %15 into %arg20[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                    scf.yield %inserted_slice_53 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                  } {"./c"}
# CHECK-NEXT:                  %inserted_slice_49 = tensor.insert_slice %14 into %arg18[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_49 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:                %inserted_slice_42 = tensor.insert_slice %13 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_42 : tensor<1x1x1x1xf32>
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:              %extracted_slice_33 = tensor.extract_slice %arg14[0, 0, 0, %arg12] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %cst_4 : tensor<1x1x1x1xf32>, f32) outs(%extracted_slice_33 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_36: f32, %out: f32):
# CHECK-NEXT:                %13 = arith.maximumf %in, %in_36 : f32
# CHECK-NEXT:                linalg.yield %13 : f32
# CHECK-NEXT:              } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %inserted_slice_34 = tensor.insert_slice %11 into %arg13[0, 0, 0, %arg12] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              %inserted_slice_35 = tensor.insert_slice %12 into %arg14[0, 0, 0, %arg12] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_34, %inserted_slice_35 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9#0, %cst_4 : tensor<1x1x1x16xf32>, f32) outs(%extracted_slice_25 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_28: f32, %out: f32):
# CHECK-NEXT:              %11 = arith.maximumf %in, %in_28 : f32
# CHECK-NEXT:              linalg.yield %11 : f32
# CHECK-NEXT:            } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:            %inserted_slice_26 = tensor.insert_slice %9#0 into %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:            %inserted_slice_27 = tensor.insert_slice %9#1 into %arg11[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_26, %inserted_slice_27 : tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %8 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7#0, %cst_4 : tensor<1x1x8x16xf32>, f32) outs(%extracted_slice_17 : tensor<1x1x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_20: f32, %out: f32):
# CHECK-NEXT:            %9 = arith.maximumf %in, %in_20 : f32
# CHECK-NEXT:            linalg.yield %9 : f32
# CHECK-NEXT:          } -> tensor<1x1x8x16xf32>
# CHECK-NEXT:          %inserted_slice_18 = tensor.insert_slice %7#0 into %arg7[0, %arg6, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:          %inserted_slice_19 = tensor.insert_slice %7#1 into %arg8[0, %arg6, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_18, %inserted_slice_19 : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %cst_4 : tensor<1x8x8x16xf32>, f32) outs(%extracted_slice_9 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_11: f32, %out: f32):
# CHECK-NEXT:          %7 = arith.maximumf %in, %in_11 : f32
# CHECK-NEXT:          linalg.yield %7 : f32
# CHECK-NEXT:        } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5#0 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:        %inserted_slice_10 = tensor.insert_slice %5#1 into %arg5[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice, %inserted_slice_10 : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3#0, %cst_4 : tensor<1x8x8x16xf32>, f32) outs(%2 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.maximumf %in, %in_5 : f32
# CHECK-NEXT:        linalg.yield %5 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %3#1 in restrict writable %arg2 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_mini(%arg0: memref<1x10x10x3xf32> {llvm.noalias}, %arg1: memref<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x8x8x16xf32>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %alloc) -> (memref<1x8x8x16xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %subview) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_1 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_1) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_3 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_3 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>)
# CHECK-NEXT:            %subview_4 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_3, %subview_4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg8 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_2 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_0 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x8x8x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      %1:2 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %0, %arg5 = %arg2) -> (memref<1x8x8x16xf32>, memref<1x8x8x16xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : memref<1x10x10x3xf32> to memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg5[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %2:2 = scf.for %arg6 = %c0 to %c8 step %c1 iter_args(%arg7 = %subview_0, %arg8 = %subview_1) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, 0, %arg6, 0] [1, 3, 3, 3] [1, 1, 1, 1] : memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %arg7[0, 0, %arg6, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %arg8[0, 0, %arg6, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %3:2 = scf.for %arg9 = %c0 to %c16 step %c1 iter_args(%arg10 = %subview_5, %arg11 = %subview_6) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_9 = memref.subview %arg1[0, 0, 0, %arg9] [3, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x16xf32> to memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg12 = %c0 to %c3 step %c1 iter_args(%arg13 = %subview_10) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_14 = memref.subview %subview_4[0, %arg12, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_15 = memref.subview %subview_9[%arg12, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg14 = %c0 to %c3 step %c1 iter_args(%arg15 = %arg13) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_16 = memref.subview %subview_14[0, 0, %arg14, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_17 = memref.subview %subview_15[0, %arg14, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %6 = scf.for %arg16 = %c0 to %c3 step %c1 iter_args(%arg17 = %arg15) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:                  %subview_18 = memref.subview %subview_16[0, 0, 0, %arg16] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_19 = memref.subview %subview_17[0, 0, %arg16, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_18, %subview_19 : memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>) outs(%arg17 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_20: f32, %out: f32):
# CHECK-NEXT:                    %7 = arith.mulf %in, %in_20 fastmath<fast> : f32
# CHECK-NEXT:                    %8 = arith.addf %out, %7 fastmath<fast> : f32
# CHECK-NEXT:                    linalg.yield %8 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  scf.yield %arg17 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:                } {"./c"}
# CHECK-NEXT:                scf.yield %6 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:              scf.yield %5 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:            %subview_11 = memref.subview %arg11[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>, f32) outs(%subview_11 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_14: f32, %out: f32):
# CHECK-NEXT:              %5 = arith.maximumf %in, %in_14 : f32
# CHECK-NEXT:              linalg.yield %5 : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            %subview_12 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %4, %subview_12 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %arg11[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_11, %subview_13 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg10, %arg11 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_7 = memref.subview %arg7[0, 0, %arg6, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3#0, %subview_7 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %arg8[0, 0, %arg6, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3#1, %subview_8 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg7, %arg8 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2#0, %subview_2 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg5[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2#1, %subview_3 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4, %arg5 : memref<1x8x8x16xf32>, memref<1x8x8x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %1#1, %arg2 : memref<1x8x8x16xf32> to memref<1x8x8x16xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'O'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [1x8x8x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
