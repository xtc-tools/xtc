# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K = 4, 32, 256
a = O.tensor((I, K), "float16", name="A")
b = O.tensor((K, J), "float16", name="B")

with O.graph(name="matmul_f16_acc_f32") as gb:
    m = O.matmul(a, b, name="C", acc_dtype="float32")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="C")
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.tile("k", {"k1": 64})
sch.fuse_consumer_at("j")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_f16_acc_f32_tensor_fused",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_f16_acc_f32(%arg0: tensor<4x256xf16> {llvm.noalias}, %arg1: tensor<256x32xf16> {llvm.noalias}, %arg2: memref<4x32xf16> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %2 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:      %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x256xf16>, tensor<256x32xf16>) outs(%2 : tensor<4x32xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: f16, %in_1: f16, %out: f32):
# CHECK-NEXT:        %7 = arith.extf %in : f16 to f32
# CHECK-NEXT:        %8 = arith.extf %in_1 : f16 to f32
# CHECK-NEXT:        %9 = arith.mulf %7, %8 : f32
# CHECK-NEXT:        %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:        linalg.yield %10 : f32
# CHECK-NEXT:      } -> tensor<4x32xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x32xf32>) outs(%0 : tensor<4x32xf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:        %7 = arith.truncf %in : f32 to f16
# CHECK-NEXT:        linalg.yield %7 : f16
# CHECK-NEXT:      } -> tensor<4x32xf16>
# CHECK-NEXT:      %5 = tensor.empty() : tensor<4x32xf16>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f16
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_0 : tensor<4x32xf16>, f16) outs(%5 : tensor<4x32xf16>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f16, %in_1: f16, %out: f16):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_1 : f16
# CHECK-NEXT:        linalg.yield %7 : f16
# CHECK-NEXT:      } -> tensor<4x32xf16>
# CHECK-NEXT:      bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<4x32xf16>, memref<4x32xf16>) -> ()
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./j1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./k1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_cast_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops:3 = transform.xtc.fuse_consumer %2 into %loops_3, %loops_5, %loops_7 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer_14, %new_loops_15:3 = transform.xtc.fuse_consumer %3 into %new_loops#0, %new_loops#1, %new_loops#2 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops_15#0 "./i" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops_15#1 "./i1" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops_15#2 "./j" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_f16_acc_f32(%arg0: tensor<4x256xf16> {llvm.noalias}, %arg1: tensor<256x32xf16> {llvm.noalias}, %arg2: memref<4x32xf16> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<4x32xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_4 = arith.constant 1 : index
# CHECK-NEXT:        %7 = scf.for %arg5 = %c0_3 to %c32 step %c1_4 iter_args(%arg6 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:          %extracted_slice_5 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %8 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_5 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_6 = tensor.insert_slice %8 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_6 : tensor<1x32xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %7 into %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %3 = tensor.empty() : tensor<4x32xf16>
# CHECK-NEXT:      %cst_2 = arith.constant 0.000000e+00 : f16
# CHECK-NEXT:      %4:3 = scf.for %arg3 = %c0_0 to %c4_1 step %c2 iter_args(%arg4 = %2, %arg5 = %0, %arg6 = %3) -> (tensor<4x32xf32>, tensor<4x32xf16>, tensor<4x32xf16>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [2, 256] [1, 1] : tensor<4x256xf16> to tensor<2x256xf16>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg1[0, 0] [256, 32] [1, 1] : tensor<256x32xf16> to tensor<256x32xf16>
# CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c2_6 = arith.constant 2 : index
# CHECK-NEXT:        %c1_7 = arith.constant 1 : index
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %arg5[%arg3, 0] [2, 32] [1, 1] : tensor<4x32xf16> to tensor<2x32xf16>
# CHECK-NEXT:        %extracted_slice_9 = tensor.extract_slice %arg6[%arg3, 0] [2, 32] [1, 1] : tensor<4x32xf16> to tensor<2x32xf16>
# CHECK-NEXT:        %7:3 = scf.for %arg7 = %c0_5 to %c2_6 step %c1_7 iter_args(%arg8 = %extracted_slice_4, %arg9 = %extracted_slice_8, %arg10 = %extracted_slice_9) -> (tensor<2x32xf32>, tensor<2x32xf16>, tensor<2x32xf16>) {
# CHECK-NEXT:          %extracted_slice_12 = tensor.extract_slice %extracted_slice[%arg7, 0] [1, 256] [1, 1] : tensor<2x256xf16> to tensor<1x256xf16>
# CHECK-NEXT:          %extracted_slice_13 = tensor.extract_slice %extracted_slice_3[0, 0] [256, 32] [1, 1] : tensor<256x32xf16> to tensor<256x32xf16>
# CHECK-NEXT:          %extracted_slice_14 = tensor.extract_slice %arg8[%arg7, 0] [1, 32] [1, 1] : tensor<2x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:          %c0_15 = arith.constant 0 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          %extracted_slice_16 = tensor.extract_slice %arg9[%arg7, 0] [1, 32] [1, 1] : tensor<2x32xf16> to tensor<1x32xf16>
# CHECK-NEXT:          %extracted_slice_17 = tensor.extract_slice %arg10[%arg7, 0] [1, 32] [1, 1] : tensor<2x32xf16> to tensor<1x32xf16>
# CHECK-NEXT:          %10:3 = scf.for %arg11 = %c0_15 to %c32 step %c16 iter_args(%arg12 = %extracted_slice_14, %arg13 = %extracted_slice_16, %arg14 = %extracted_slice_17) -> (tensor<1x32xf32>, tensor<1x32xf16>, tensor<1x32xf16>) {
# CHECK-NEXT:            %extracted_slice_21 = tensor.extract_slice %extracted_slice_12[0, 0] [1, 256] [1, 1] : tensor<1x256xf16> to tensor<1x256xf16>
# CHECK-NEXT:            %extracted_slice_22 = tensor.extract_slice %extracted_slice_13[0, %arg11] [256, 16] [1, 1] : tensor<256x32xf16> to tensor<256x16xf16>
# CHECK-NEXT:            %extracted_slice_23 = tensor.extract_slice %arg12[0, %arg11] [1, 16] [1, 1] : tensor<1x32xf32> to tensor<1x16xf32>
# CHECK-NEXT:            %c0_24 = arith.constant 0 : index
# CHECK-NEXT:            %c16_25 = arith.constant 16 : index
# CHECK-NEXT:            %c1_26 = arith.constant 1 : index
# CHECK-NEXT:            %13 = scf.for %arg15 = %c0_24 to %c16_25 step %c1_26 iter_args(%arg16 = %extracted_slice_23) -> (tensor<1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_32 = tensor.extract_slice %extracted_slice_21[0, 0] [1, 256] [1, 1] : tensor<1x256xf16> to tensor<1x256xf16>
# CHECK-NEXT:              %extracted_slice_33 = tensor.extract_slice %extracted_slice_22[0, %arg15] [256, 1] [1, 1] : tensor<256x16xf16> to tensor<256x1xf16>
# CHECK-NEXT:              %extracted_slice_34 = tensor.extract_slice %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:              %c0_35 = arith.constant 0 : index
# CHECK-NEXT:              %c256 = arith.constant 256 : index
# CHECK-NEXT:              %c64 = arith.constant 64 : index
# CHECK-NEXT:              %16 = scf.for %arg17 = %c0_35 to %c256 step %c64 iter_args(%arg18 = %extracted_slice_34) -> (tensor<1x1xf32>) {
# CHECK-NEXT:                %extracted_slice_37 = tensor.extract_slice %extracted_slice_32[0, %arg17] [1, 64] [1, 1] : tensor<1x256xf16> to tensor<1x64xf16>
# CHECK-NEXT:                %extracted_slice_38 = tensor.extract_slice %extracted_slice_33[%arg17, 0] [64, 1] [1, 1] : tensor<256x1xf16> to tensor<64x1xf16>
# CHECK-NEXT:                %extracted_slice_39 = tensor.extract_slice %arg18[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %c0_40 = arith.constant 0 : index
# CHECK-NEXT:                %c64_41 = arith.constant 64 : index
# CHECK-NEXT:                %c1_42 = arith.constant 1 : index
# CHECK-NEXT:                %17 = scf.for %arg19 = %c0_40 to %c64_41 step %c1_42 iter_args(%arg20 = %extracted_slice_39) -> (tensor<1x1xf32>) {
# CHECK-NEXT:                  %extracted_slice_44 = tensor.extract_slice %extracted_slice_37[0, %arg19] [1, 1] [1, 1] : tensor<1x64xf16> to tensor<1x1xf16>
# CHECK-NEXT:                  %extracted_slice_45 = tensor.extract_slice %extracted_slice_38[%arg19, 0] [1, 1] [1, 1] : tensor<64x1xf16> to tensor<1x1xf16>
# CHECK-NEXT:                  %extracted_slice_46 = tensor.extract_slice %arg20[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %18 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_44, %extracted_slice_45 : tensor<1x1xf16>, tensor<1x1xf16>) outs(%extracted_slice_46 : tensor<1x1xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:                  ^bb0(%in: f16, %in_48: f16, %out: f32):
# CHECK-NEXT:                    %19 = arith.extf %in : f16 to f32
# CHECK-NEXT:                    %20 = arith.extf %in_48 : f16 to f32
# CHECK-NEXT:                    %21 = arith.mulf %19, %20 : f32
# CHECK-NEXT:                    %22 = arith.addf %out, %21 : f32
# CHECK-NEXT:                    linalg.yield %22 : f32
# CHECK-NEXT:                  } -> tensor<1x1xf32>
# CHECK-NEXT:                  %inserted_slice_47 = tensor.insert_slice %18 into %arg20[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_47 : tensor<1x1xf32>
# CHECK-NEXT:                } {"./k1"}
# CHECK-NEXT:                %inserted_slice_43 = tensor.insert_slice %17 into %arg18[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_43 : tensor<1x1xf32>
# CHECK-NEXT:              } {"./k"}
# CHECK-NEXT:              %inserted_slice_36 = tensor.insert_slice %16 into %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_36 : tensor<1x16xf32>
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %extracted_slice_27 = tensor.extract_slice %arg13[0, %arg11] [1, 16] [1, 1] : tensor<1x32xf16> to tensor<1x16xf16>
# CHECK-NEXT:            %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<1x16xf32>) outs(%extracted_slice_27 : tensor<1x16xf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:            ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:              %16 = arith.truncf %in : f32 to f16
# CHECK-NEXT:              linalg.yield %16 : f16
# CHECK-NEXT:            } -> tensor<1x16xf16>
# CHECK-NEXT:            %inserted_slice_28 = tensor.insert_slice %13 into %arg12[0, %arg11] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<1x32xf32>
# CHECK-NEXT:            %extracted_slice_29 = tensor.extract_slice %arg14[0, %arg11] [1, 16] [1, 1] : tensor<1x32xf16> to tensor<1x16xf16>
# CHECK-NEXT:            %15 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%14, %cst_2 : tensor<1x16xf16>, f16) outs(%extracted_slice_29 : tensor<1x16xf16>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f16, %in_32: f16, %out: f16):
# CHECK-NEXT:              %16 = arith.maximumf %in, %in_32 : f16
# CHECK-NEXT:              linalg.yield %16 : f16
# CHECK-NEXT:            } -> tensor<1x16xf16>
# CHECK-NEXT:            %inserted_slice_30 = tensor.insert_slice %14 into %arg13[0, %arg11] [1, 16] [1, 1] : tensor<1x16xf16> into tensor<1x32xf16>
# CHECK-NEXT:            %inserted_slice_31 = tensor.insert_slice %15 into %arg14[0, %arg11] [1, 16] [1, 1] : tensor<1x16xf16> into tensor<1x32xf16>
# CHECK-NEXT:            scf.yield %inserted_slice_28, %inserted_slice_30, %inserted_slice_31 : tensor<1x32xf32>, tensor<1x32xf16>, tensor<1x32xf16>
# CHECK-NEXT:          } {"./j"}
# CHECK-NEXT:          %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%10#0 : tensor<1x32xf32>) outs(%extracted_slice_16 : tensor<1x32xf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:          ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:            %13 = arith.truncf %in : f32 to f16
# CHECK-NEXT:            linalg.yield %13 : f16
# CHECK-NEXT:          } -> tensor<1x32xf16>
# CHECK-NEXT:          %inserted_slice_18 = tensor.insert_slice %10#0 into %arg8[%arg7, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<2x32xf32>
# CHECK-NEXT:          %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%10#1, %cst_2 : tensor<1x32xf16>, f16) outs(%extracted_slice_17 : tensor<1x32xf16>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:          ^bb0(%in: f16, %in_21: f16, %out: f16):
# CHECK-NEXT:            %13 = arith.maximumf %in, %in_21 : f16
# CHECK-NEXT:            linalg.yield %13 : f16
# CHECK-NEXT:          } -> tensor<1x32xf16>
# CHECK-NEXT:          %inserted_slice_19 = tensor.insert_slice %10#1 into %arg9[%arg7, 0] [1, 32] [1, 1] : tensor<1x32xf16> into tensor<2x32xf16>
# CHECK-NEXT:          %inserted_slice_20 = tensor.insert_slice %10#2 into %arg10[%arg7, 0] [1, 32] [1, 1] : tensor<1x32xf16> into tensor<2x32xf16>
# CHECK-NEXT:          scf.yield %inserted_slice_18, %inserted_slice_19, %inserted_slice_20 : tensor<2x32xf32>, tensor<2x32xf16>, tensor<2x32xf16>
# CHECK-NEXT:        } {"./i1"}
# CHECK-NEXT:        %8 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%7#0 : tensor<2x32xf32>) outs(%extracted_slice_8 : tensor<2x32xf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:        ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:          %10 = arith.truncf %in : f32 to f16
# CHECK-NEXT:          linalg.yield %10 : f16
# CHECK-NEXT:        } -> tensor<2x32xf16>
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %7#0 into %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        %9 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%7#1, %cst_2 : tensor<2x32xf16>, f16) outs(%extracted_slice_9 : tensor<2x32xf16>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f16, %in_12: f16, %out: f16):
# CHECK-NEXT:          %10 = arith.maximumf %in, %in_12 : f16
# CHECK-NEXT:          linalg.yield %10 : f16
# CHECK-NEXT:        } -> tensor<2x32xf16>
# CHECK-NEXT:        %inserted_slice_10 = tensor.insert_slice %7#1 into %arg5[%arg3, 0] [2, 32] [1, 1] : tensor<2x32xf16> into tensor<4x32xf16>
# CHECK-NEXT:        %inserted_slice_11 = tensor.insert_slice %7#2 into %arg6[%arg3, 0] [2, 32] [1, 1] : tensor<2x32xf16> into tensor<4x32xf16>
# CHECK-NEXT:        scf.yield %inserted_slice, %inserted_slice_10, %inserted_slice_11 : tensor<4x32xf32>, tensor<4x32xf16>, tensor<4x32xf16>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %5 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%4#0 : tensor<4x32xf32>) outs(%0 : tensor<4x32xf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:        %7 = arith.truncf %in : f32 to f16
# CHECK-NEXT:        linalg.yield %7 : f16
# CHECK-NEXT:      } -> tensor<4x32xf16>
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%4#1, %cst_2 : tensor<4x32xf16>, f16) outs(%3 : tensor<4x32xf16>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f16, %in_3: f16, %out: f16):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_3 : f16
# CHECK-NEXT:        linalg.yield %7 : f16
# CHECK-NEXT:      } -> tensor<4x32xf16>
# CHECK-NEXT:      bufferization.materialize_in_destination %4#2 in restrict writable %arg2 : (tensor<4x32xf16>, memref<4x32xf16>) -> ()
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
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_f16_acc_f32(%arg0: memref<4x256xf16> {llvm.noalias}, %arg1: memref<256x32xf16> {llvm.noalias}, %arg2: memref<4x32xf16> {llvm.noalias}) {
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f16
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf16>
# CHECK-NEXT:      %alloca_1 = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %alloca_1) -> (memref<4x32xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_3 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst_0 : f32) outs(%subview_3 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_3, %subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_2 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %1:3 = scf.for %arg3 = %c0 to %c4 step %c2 iter_args(%arg4 = %0, %arg5 = %alloca, %arg6 = %arg2) -> (memref<4x32xf32>, memref<4x32xf16>, memref<4x32xf16>) {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [2, 256] [1, 1] : memref<4x256xf16> to memref<2x256xf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg5[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf16> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_4 = memref.subview %arg6[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf16> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2:3 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %subview_2, %arg9 = %subview_3, %arg10 = %subview_4) -> (memref<2x32xf32, strided<[32, 1], offset: ?>>, memref<2x32xf16, strided<[32, 1], offset: ?>>, memref<2x32xf16, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_8 = memref.subview %subview[%arg7, 0] [1, 256] [1, 1] : memref<2x256xf16, strided<[256, 1], offset: ?>> to memref<1x256xf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %subview_9 = memref.subview %arg8[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_10 = memref.subview %arg9[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_11 = memref.subview %arg10[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %3:3 = scf.for %arg11 = %c0 to %c32 step %c16 iter_args(%arg12 = %subview_9, %arg13 = %subview_10, %arg14 = %subview_11) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>, memref<1x32xf16, strided<[32, 1], offset: ?>>, memref<1x32xf16, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_15 = memref.subview %arg1[0, %arg11] [256, 16] [1, 1] : memref<256x32xf16> to memref<256x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_16 = memref.subview %arg12[0, %arg11] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg15 = %c0 to %c16 step %c1 iter_args(%arg16 = %subview_16) -> (memref<1x16xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_22 = memref.subview %subview_15[0, %arg15] [256, 1] [1, 1] : memref<256x16xf16, strided<[32, 1], offset: ?>> to memref<256x1xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %subview_23 = memref.subview %arg16[0, %arg15] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg17 = %c0 to %c256 step %c64 iter_args(%arg18 = %subview_23) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_25 = memref.subview %subview_8[0, %arg17] [1, 64] [1, 1] : memref<1x256xf16, strided<[256, 1], offset: ?>> to memref<1x64xf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                %subview_26 = memref.subview %subview_22[%arg17, 0] [64, 1] [1, 1] : memref<256x1xf16, strided<[32, 1], offset: ?>> to memref<64x1xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                %6 = scf.for %arg19 = %c0 to %c64 step %c1 iter_args(%arg20 = %arg18) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:                  %subview_27 = memref.subview %subview_25[0, %arg19] [1, 1] [1, 1] : memref<1x64xf16, strided<[256, 1], offset: ?>> to memref<1x1xf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                  %subview_28 = memref.subview %subview_26[%arg19, 0] [1, 1] [1, 1] : memref<64x1xf16, strided<[32, 1], offset: ?>> to memref<1x1xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_27, %subview_28 : memref<1x1xf16, strided<[256, 1], offset: ?>>, memref<1x1xf16, strided<[32, 1], offset: ?>>) outs(%arg20 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:                  ^bb0(%in: f16, %in_29: f16, %out: f32):
# CHECK-NEXT:                    %7 = arith.extf %in : f16 to f32
# CHECK-NEXT:                    %8 = arith.extf %in_29 : f16 to f32
# CHECK-NEXT:                    %9 = arith.mulf %7, %8 : f32
# CHECK-NEXT:                    %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                    linalg.yield %10 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  scf.yield %arg20 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                } {"./k1"}
# CHECK-NEXT:                scf.yield %6 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              } {"./k"}
# CHECK-NEXT:              %subview_24 = memref.subview %arg16[0, %arg15] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %5, %subview_24 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg16 : memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %subview_17 = memref.subview %arg13[0, %arg11] [1, 16] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<1x16xf32, strided<[32, 1], offset: ?>>) outs(%subview_17 : memref<1x16xf16, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:            ^bb0(%in: f32, %out: f16):
# CHECK-NEXT:              %5 = arith.truncf %in : f32 to f16
# CHECK-NEXT:              linalg.yield %5 : f16
# CHECK-NEXT:            }
# CHECK-NEXT:            %subview_18 = memref.subview %arg12[0, %arg11] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %4, %subview_18 : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %arg14[0, %arg11] [1, 16] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview_17, %cst : memref<1x16xf16, strided<[32, 1], offset: ?>>, f16) outs(%subview_19 : memref<1x16xf16, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f16, %in_22: f16, %out: f16):
# CHECK-NEXT:              %5 = arith.maximumf %in, %in_22 : f16
# CHECK-NEXT:              linalg.yield %5 : f16
# CHECK-NEXT:            }
# CHECK-NEXT:            %subview_20 = memref.subview %arg13[0, %arg11] [1, 16] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_17, %subview_20 : memref<1x16xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_21 = memref.subview %arg14[0, %arg11] [1, 16] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %subview_19, %subview_21 : memref<1x16xf16, strided<[32, 1], offset: ?>> to memref<1x16xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg12, %arg13, %arg14 : memref<1x32xf32, strided<[32, 1], offset: ?>>, memref<1x32xf16, strided<[32, 1], offset: ?>>, memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {"./j"}
# CHECK-NEXT:          %subview_12 = memref.subview %arg8[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3#0, %subview_12 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_13 = memref.subview %arg9[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3#1, %subview_13 : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_14 = memref.subview %arg10[%arg7, 0] [1, 32] [1, 1] : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3#2, %subview_14 : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg8, %arg9, %arg10 : memref<2x32xf32, strided<[32, 1], offset: ?>>, memref<2x32xf16, strided<[32, 1], offset: ?>>, memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./i1"}
# CHECK-NEXT:        %subview_5 = memref.subview %arg4[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2#0, %subview_5 : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_6 = memref.subview %arg5[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf16> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2#1, %subview_6 : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg6[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf16> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2#2, %subview_7 : memref<2x32xf16, strided<[32, 1], offset: ?>> to memref<2x32xf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4, %arg5, %arg6 : memref<4x32xf32>, memref<4x32xf16>, memref<4x32xf16>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      memref.copy %1#2, %arg2 : memref<4x32xf16> to memref<4x32xf16>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul_f16_acc_f32
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x256xfloat16
# CHECK-NEXT:    - %1 : 256x32xfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 4x32xfloat16
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1, acc_dtype=float32) {name = 'C'} : [4x256xfloat16, 256x32xfloat16] -> [4x32xfloat16]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [4x32xfloat16] -> [4x32xfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
