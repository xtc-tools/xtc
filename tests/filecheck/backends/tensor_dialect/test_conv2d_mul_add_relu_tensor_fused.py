# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")
scale = O.tensor((F,), dtype, name="scale")
bias = O.tensor((F,), dtype, name="bias")

with O.graph(name="conv2d_mul_add_relu") as gb:
    c = O.conv2d(a, b, stride=(SH, SW), name="conv")
    d = O.mul(c, scale, name="mul")
    e = O.add(d, bias, name="add")
    f = O.relu(e, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="conv")
sch.fuse_consumer_at("f", chain=True)
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_mul_add_relu_mlir_tensor",
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
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_mul_add_relu(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: tensor<16xf32> {llvm.noalias}, %arg3: tensor<16xf32> {llvm.noalias}, %arg4: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%0 : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x10x10x3xf32>, tensor<3x3x3x16xf32>) outs(%1 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.mulf %in, %in_1 fastmath<fast> : f32
# CHECK-NEXT:        %10 = arith.addf %out, %9 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %10 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %3 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg2 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%3 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.mulf %in, %in_1 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %5 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %arg3 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%5 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.addf %in, %in_1 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %7 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %8 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %cst_0 : tensor<1x8x8x16xf32>, f32) outs(%7 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.maximumf %in, %in_1 : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %8 in restrict writable %arg4 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_conv_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
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
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_mul_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops:4 = transform.xtc.fuse_consumer %2 into %loops_7, %loops_9, %loops_11, %loops_13 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_add_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer_20, %new_loops_21:4 = transform.xtc.fuse_consumer %3 into %new_loops#0, %new_loops#1, %new_loops#2, %new_loops#3 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer_22, %new_loops_23:4 = transform.xtc.fuse_consumer %4 into %new_loops_21#0, %new_loops_21#1, %new_loops_21#2, %new_loops_21#3 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops_23#0 "./b" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops_23#1 "./h" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops_23#2 "./w" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops_23#3 "./f" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_mul_add_relu(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: tensor<16xf32> {llvm.noalias}, %arg3: tensor<16xf32> {llvm.noalias}, %arg4: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg5 = %c0 to %c1 step %c1_0 iter_args(%arg6 = %0) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        %9 = scf.for %arg7 = %c0_5 to %c8 step %c1_6 iter_args(%arg8 = %extracted_slice) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %arg8[0, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %c0_8 = arith.constant 0 : index
# CHECK-NEXT:          %c8_9 = arith.constant 8 : index
# CHECK-NEXT:          %c1_10 = arith.constant 1 : index
# CHECK-NEXT:          %10 = scf.for %arg9 = %c0_8 to %c8_9 step %c1_10 iter_args(%arg10 = %extracted_slice_7) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:            %extracted_slice_12 = tensor.extract_slice %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_14 = arith.constant 1 : index
# CHECK-NEXT:            %11 = scf.for %arg11 = %c0_13 to %c16 step %c1_14 iter_args(%arg12 = %extracted_slice_12) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_16 = tensor.extract_slice %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %12 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%extracted_slice_16 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %inserted_slice_17 = tensor.insert_slice %12 into %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_17 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %inserted_slice_15 = tensor.insert_slice %11 into %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_15 : tensor<1x1x8x16xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %inserted_slice_11 = tensor.insert_slice %10 into %arg8[0, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_11 : tensor<1x8x8x16xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %9 into %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %c0_1 = arith.constant 0 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %3 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %4 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %5:4 = scf.for %arg5 = %c0_1 to %c1_2 step %c1_3 iter_args(%arg6 = %1, %arg7 = %2, %arg8 = %3, %arg9 = %4) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg5, 0, 0, 0] [1, 10, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x10x10x3xf32>
# CHECK-NEXT:        %extracted_slice_5 = tensor.extract_slice %arg1[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_8 = arith.constant 1 : index
# CHECK-NEXT:        %extracted_slice_9 = tensor.extract_slice %arg2[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:        %extracted_slice_10 = tensor.extract_slice %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %extracted_slice_11 = tensor.extract_slice %arg3[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:        %extracted_slice_12 = tensor.extract_slice %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %extracted_slice_13 = tensor.extract_slice %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:        %9:4 = scf.for %arg10 = %c0_7 to %c8 step %c1_8 iter_args(%arg11 = %extracted_slice_6, %arg12 = %extracted_slice_10, %arg13 = %extracted_slice_12, %arg14 = %extracted_slice_13) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
# CHECK-NEXT:          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg10, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x3x10x3xf32>
# CHECK-NEXT:          %extracted_slice_15 = tensor.extract_slice %extracted_slice_5[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:          %extracted_slice_16 = tensor.extract_slice %arg11[0, %arg10, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %c0_17 = arith.constant 0 : index
# CHECK-NEXT:          %c8_18 = arith.constant 8 : index
# CHECK-NEXT:          %c1_19 = arith.constant 1 : index
# CHECK-NEXT:          %extracted_slice_20 = tensor.extract_slice %extracted_slice_9[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:          %extracted_slice_21 = tensor.extract_slice %arg11[0, %arg10, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %extracted_slice_22 = tensor.extract_slice %extracted_slice_11[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:          %extracted_slice_23 = tensor.extract_slice %arg11[0, %arg10, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %extracted_slice_24 = tensor.extract_slice %arg11[0, %arg10, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:          %13:4 = scf.for %arg15 = %c0_17 to %c8_18 step %c1_19 iter_args(%arg16 = %extracted_slice_16, %arg17 = %extracted_slice_21, %arg18 = %extracted_slice_23, %arg19 = %extracted_slice_24) -> (tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>) {
# CHECK-NEXT:            %extracted_slice_26 = tensor.extract_slice %extracted_slice_14[0, 0, %arg15, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x10x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:            %extracted_slice_27 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:            %extracted_slice_28 = tensor.extract_slice %arg16[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_29 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_30 = arith.constant 1 : index
# CHECK-NEXT:            %extracted_slice_31 = tensor.extract_slice %extracted_slice_20[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:            %extracted_slice_32 = tensor.extract_slice %arg16[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %extracted_slice_33 = tensor.extract_slice %extracted_slice_22[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:            %extracted_slice_34 = tensor.extract_slice %arg16[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %extracted_slice_35 = tensor.extract_slice %arg16[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %17:4 = scf.for %arg20 = %c0_29 to %c16 step %c1_30 iter_args(%arg21 = %extracted_slice_28, %arg22 = %extracted_slice_32, %arg23 = %extracted_slice_34, %arg24 = %extracted_slice_35) -> (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_37 = tensor.extract_slice %extracted_slice_26[0, 0, 0, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:              %extracted_slice_38 = tensor.extract_slice %extracted_slice_27[0, 0, 0, %arg20] [3, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x1xf32>
# CHECK-NEXT:              %extracted_slice_39 = tensor.extract_slice %arg21[0, 0, 0, %arg20] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %c0_40 = arith.constant 0 : index
# CHECK-NEXT:              %c3 = arith.constant 3 : index
# CHECK-NEXT:              %c1_41 = arith.constant 1 : index
# CHECK-NEXT:              %21 = scf.for %arg25 = %c0_40 to %c3 step %c1_41 iter_args(%arg26 = %extracted_slice_39) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                %extracted_slice_46 = tensor.extract_slice %extracted_slice_37[0, %arg25, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x1x3x3xf32>
# CHECK-NEXT:                %extracted_slice_47 = tensor.extract_slice %extracted_slice_38[%arg25, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x1xf32> to tensor<1x3x3x1xf32>
# CHECK-NEXT:                %extracted_slice_48 = tensor.extract_slice %arg26[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                %c0_49 = arith.constant 0 : index
# CHECK-NEXT:                %c3_50 = arith.constant 3 : index
# CHECK-NEXT:                %c1_51 = arith.constant 1 : index
# CHECK-NEXT:                %25 = scf.for %arg27 = %c0_49 to %c3_50 step %c1_51 iter_args(%arg28 = %extracted_slice_48) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                  %extracted_slice_53 = tensor.extract_slice %extracted_slice_46[0, 0, %arg27, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x3x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                  %extracted_slice_54 = tensor.extract_slice %extracted_slice_47[0, %arg27, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                  %extracted_slice_55 = tensor.extract_slice %arg28[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %c0_56 = arith.constant 0 : index
# CHECK-NEXT:                  %c3_57 = arith.constant 3 : index
# CHECK-NEXT:                  %c1_58 = arith.constant 1 : index
# CHECK-NEXT:                  %26 = scf.for %arg29 = %c0_56 to %c3_57 step %c1_58 iter_args(%arg30 = %extracted_slice_55) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                    %extracted_slice_60 = tensor.extract_slice %extracted_slice_53[0, 0, 0, %arg29] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %extracted_slice_61 = tensor.extract_slice %extracted_slice_54[0, 0, %arg29, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %extracted_slice_62 = tensor.extract_slice %arg30[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %27 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_60, %extracted_slice_61 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_62 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                    ^bb0(%in: f32, %in_64: f32, %out: f32):
# CHECK-NEXT:                      %28 = arith.mulf %in, %in_64 fastmath<fast> : f32
# CHECK-NEXT:                      %29 = arith.addf %out, %28 fastmath<fast> : f32
# CHECK-NEXT:                      linalg.yield %29 : f32
# CHECK-NEXT:                    } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                    %inserted_slice_63 = tensor.insert_slice %27 into %arg30[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                    scf.yield %inserted_slice_63 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                  } {"./c"}
# CHECK-NEXT:                  %inserted_slice_59 = tensor.insert_slice %26 into %arg28[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_59 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:                %inserted_slice_52 = tensor.insert_slice %25 into %arg26[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_52 : tensor<1x1x1x1xf32>
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:              %extracted_slice_42 = tensor.extract_slice %extracted_slice_31[%arg20] [1] [1] : tensor<16xf32> to tensor<1xf32>
# CHECK-NEXT:              %22 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %extracted_slice_42 : tensor<1x1x1x1xf32>, tensor<1xf32>) outs(%21 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                %25 = arith.mulf %in, %in_46 fastmath<fast> : f32
# CHECK-NEXT:                linalg.yield %25 : f32
# CHECK-NEXT:              } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %extracted_slice_43 = tensor.extract_slice %extracted_slice_33[%arg20] [1] [1] : tensor<16xf32> to tensor<1xf32>
# CHECK-NEXT:              %23 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1xf32>) outs(%22 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                %25 = arith.addf %in, %in_46 fastmath<fast> : f32
# CHECK-NEXT:                linalg.yield %25 : f32
# CHECK-NEXT:              } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %extracted_slice_44 = tensor.extract_slice %arg21[0, 0, 0, %arg20] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:              %24 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %cst_4 : tensor<1x1x1x1xf32>, f32) outs(%23 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                %25 = arith.maximumf %in, %in_46 : f32
# CHECK-NEXT:                linalg.yield %25 : f32
# CHECK-NEXT:              } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:              %inserted_slice_45 = tensor.insert_slice %24 into %arg21[0, 0, 0, %arg20] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_45, %arg22, %arg23, %arg24 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:            %18 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17#0, %extracted_slice_31 : tensor<1x1x1x16xf32>, tensor<16xf32>) outs(%extracted_slice_32 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_37: f32, %out: f32):
# CHECK-NEXT:              %21 = arith.mulf %in, %in_37 fastmath<fast> : f32
# CHECK-NEXT:              linalg.yield %21 : f32
# CHECK-NEXT:            } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:            %19 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17#0, %extracted_slice_33 : tensor<1x1x1x16xf32>, tensor<16xf32>) outs(%extracted_slice_34 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_37: f32, %out: f32):
# CHECK-NEXT:              %21 = arith.addf %in, %in_37 fastmath<fast> : f32
# CHECK-NEXT:              linalg.yield %21 : f32
# CHECK-NEXT:            } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:            %20 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17#0, %cst_4 : tensor<1x1x1x16xf32>, f32) outs(%extracted_slice_35 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_37: f32, %out: f32):
# CHECK-NEXT:              %21 = arith.maximumf %in, %in_37 : f32
# CHECK-NEXT:              linalg.yield %21 : f32
# CHECK-NEXT:            } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:            %inserted_slice_36 = tensor.insert_slice %17#0 into %arg16[0, 0, %arg15, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_36, %arg17, %arg18, %arg19 : tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x8x16xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#0, %extracted_slice_20 : tensor<1x1x8x16xf32>, tensor<16xf32>) outs(%extracted_slice_21 : tensor<1x1x8x16xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_26: f32, %out: f32):
# CHECK-NEXT:            %17 = arith.mulf %in, %in_26 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %17 : f32
# CHECK-NEXT:          } -> tensor<1x1x8x16xf32>
# CHECK-NEXT:          %15 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#0, %extracted_slice_22 : tensor<1x1x8x16xf32>, tensor<16xf32>) outs(%extracted_slice_23 : tensor<1x1x8x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_26: f32, %out: f32):
# CHECK-NEXT:            %17 = arith.addf %in, %in_26 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %17 : f32
# CHECK-NEXT:          } -> tensor<1x1x8x16xf32>
# CHECK-NEXT:          %16 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#0, %cst_4 : tensor<1x1x8x16xf32>, f32) outs(%extracted_slice_24 : tensor<1x1x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_26: f32, %out: f32):
# CHECK-NEXT:            %17 = arith.maximumf %in, %in_26 : f32
# CHECK-NEXT:            linalg.yield %17 : f32
# CHECK-NEXT:          } -> tensor<1x1x8x16xf32>
# CHECK-NEXT:          %inserted_slice_25 = tensor.insert_slice %13#0 into %arg11[0, %arg10, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_25, %arg12, %arg13, %arg14 : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9#0, %extracted_slice_9 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%extracted_slice_10 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_14: f32, %out: f32):
# CHECK-NEXT:          %13 = arith.mulf %in, %in_14 fastmath<fast> : f32
# CHECK-NEXT:          linalg.yield %13 : f32
# CHECK-NEXT:        } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:        %11 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9#0, %extracted_slice_11 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%extracted_slice_12 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_14: f32, %out: f32):
# CHECK-NEXT:          %13 = arith.addf %in, %in_14 fastmath<fast> : f32
# CHECK-NEXT:          linalg.yield %13 : f32
# CHECK-NEXT:        } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:        %12 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9#0, %cst_4 : tensor<1x8x8x16xf32>, f32) outs(%extracted_slice_13 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_14: f32, %out: f32):
# CHECK-NEXT:          %13 = arith.maximumf %in, %in_14 : f32
# CHECK-NEXT:          linalg.yield %13 : f32
# CHECK-NEXT:        } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %9#0 into %arg6[%arg5, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice, %arg7, %arg8, %arg9 : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %arg2 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%2 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.mulf %in, %in_5 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %arg3 : tensor<1x8x8x16xf32>, tensor<16xf32>) outs(%3 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.addf %in, %in_5 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      %8 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %cst_4 : tensor<1x8x8x16xf32>, f32) outs(%4 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:        %9 = arith.maximumf %in, %in_5 : f32
# CHECK-NEXT:        linalg.yield %9 : f32
# CHECK-NEXT:      } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %5#0 in restrict writable %arg4 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
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
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_mul_add_relu(%arg0: memref<1x10x10x3xf32> {llvm.noalias}, %arg1: memref<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<16xf32> {llvm.noalias}, %arg3: memref<16xf32> {llvm.noalias}, %arg4: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %0 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (memref<1x8x8x16xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %subview) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_1 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg9 = %c0 to %c16 step %c1 iter_args(%arg10 = %subview_1) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_3 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%subview_3 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>)
# CHECK-NEXT:            %subview_4 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_3, %subview_4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg10 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_2 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_2 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg8 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_0 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_0 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg6 : memref<1x8x8x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %0) -> (memref<1x8x8x16xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg5, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : memref<1x10x10x3xf32> to memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %subview_0) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, 0, %arg7, 0] [1, 3, 3, 3] [1, 1, 1, 1] : memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg9 = %c0 to %c16 step %c1 iter_args(%arg10 = %subview_3) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_5 = memref.subview %arg1[0, 0, 0, %arg9] [3, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x16xf32> to memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %subview_6) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_10 = memref.subview %subview_2[0, %arg11, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_11 = memref.subview %subview_5[%arg11, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg13 = %c0 to %c3 step %c1 iter_args(%arg14 = %arg12) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_12 = memref.subview %subview_10[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_13 = memref.subview %subview_11[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %6 = scf.for %arg15 = %c0 to %c3 step %c1 iter_args(%arg16 = %arg14) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:                  %subview_14 = memref.subview %subview_12[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_15 = memref.subview %subview_13[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_14, %subview_15 : memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>) outs(%arg16 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_16: f32, %out: f32):
# CHECK-NEXT:                    %7 = arith.mulf %in, %in_16 fastmath<fast> : f32
# CHECK-NEXT:                    %8 = arith.addf %out, %7 fastmath<fast> : f32
# CHECK-NEXT:                    linalg.yield %8 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  scf.yield %arg16 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:                } {"./c"}
# CHECK-NEXT:                scf.yield %6 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:              scf.yield %5 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:            %subview_7 = memref.subview %arg2[%arg9] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %subview_7 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>) outs(%4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:              %5 = arith.mulf %in, %in_10 fastmath<fast> : f32
# CHECK-NEXT:              linalg.yield %5 : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            %subview_8 = memref.subview %arg3[%arg9] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %subview_8 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>) outs(%4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:              %5 = arith.addf %in, %in_10 fastmath<fast> : f32
# CHECK-NEXT:              linalg.yield %5 : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>, f32) outs(%4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:              %5 = arith.maximumf %in, %in_10 : f32
# CHECK-NEXT:              linalg.yield %5 : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            %subview_9 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %4, %subview_9 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg10 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./f"}
# CHECK-NEXT:          %subview_4 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_4 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg8 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_1 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_1 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg6 : memref<1x8x8x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %1, %arg4 : memref<1x8x8x16xf32> to memref<1x8x8x16xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv2d_mul_add_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    - %2 : 16xfloat32
# CHECK-NEXT:    - %3 : 16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %7 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %4: conv2d(%0, %1, stride=(1, 1)) {name = 'conv'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %5: mul(%4, %2) {name = 'mul'} : [1x8x8x16xfloat32, 16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %6: add(%5, %3) {name = 'add'} : [1x8x8x16xfloat32, 16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %7: relu(%6) {name = 'relu'} : [1x8x8x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
