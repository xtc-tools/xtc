# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="conv2d_nhwc_mini") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_mini_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_mini(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%0 : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
# CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x10x10x3xf32>, tensor<3x3x3x16xf32>) outs(%1 : tensor<1x8x8x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:       %3 = arith.mulf %in, %in_0 : f32
# CHECK-NEXT:       %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:       linalg.yield %4 : f32
# CHECK-NEXT:     } -> tensor<1x8x8x16xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_O_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_mini(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c1_0 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %0) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:       %c0_4 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %c1_5 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_4 to %c8 step %c1_5 iter_args(%arg6 = %extracted_slice) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:         %c0_7 = arith.constant 0 : index
# CHECK-NEXT:         %c8_8 = arith.constant 8 : index
# CHECK-NEXT:         %c1_9 = arith.constant 1 : index
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0_7 to %c8_8 step %c1_9 iter_args(%arg8 = %extracted_slice_6) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:           %extracted_slice_11 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_12 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_13 = arith.constant 1 : index
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0_12 to %c16 step %c1_13 iter_args(%arg10 = %extracted_slice_11) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_15 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %6 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_15 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_16 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_16 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_14 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_14 : tensor<1x1x8x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_10 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_10 : tensor<1x8x8x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %c0_1 = arith.constant 0 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0_1 to %c1_2 step %c1_3 iter_args(%arg4 = %1) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 10, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x10x10x3xf32>
# CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %arg1[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:       %c0_6 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %c1_7 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_6 to %c8 step %c1_7 iter_args(%arg6 = %extracted_slice_5) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %extracted_slice[0, %arg5, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x3x10x3xf32>
# CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c8_12 = arith.constant 8 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0_11 to %c8_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %extracted_slice_8[0, 0, %arg7, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x10x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:           %extracted_slice_16 = tensor.extract_slice %extracted_slice_9[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_18 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_19 = arith.constant 1 : index
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0_18 to %c16 step %c1_19 iter_args(%arg10 = %extracted_slice_17) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:             %extracted_slice_22 = tensor.extract_slice %extracted_slice_16[0, 0, 0, %arg9] [3, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x1xf32>
# CHECK-NEXT:             %extracted_slice_23 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %c0_24 = arith.constant 0 : index
# CHECK-NEXT:             %c3 = arith.constant 3 : index
# CHECK-NEXT:             %c1_25 = arith.constant 1 : index
# CHECK-NEXT:             %6 = scf.for %arg11 = %c0_24 to %c3 step %c1_25 iter_args(%arg12 = %extracted_slice_23) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:               %extracted_slice_27 = tensor.extract_slice %extracted_slice_21[0, %arg11, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x1x3x3xf32>
# CHECK-NEXT:               %extracted_slice_28 = tensor.extract_slice %extracted_slice_22[%arg11, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x1xf32> to tensor<1x3x3x1xf32>
# CHECK-NEXT:               %extracted_slice_29 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:               %c0_30 = arith.constant 0 : index
# CHECK-NEXT:               %c3_31 = arith.constant 3 : index
# CHECK-NEXT:               %c1_32 = arith.constant 1 : index
# CHECK-NEXT:               %7 = scf.for %arg13 = %c0_30 to %c3_31 step %c1_32 iter_args(%arg14 = %extracted_slice_29) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %extracted_slice_27[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x3x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %extracted_slice_35 = tensor.extract_slice %extracted_slice_28[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %c0_37 = arith.constant 0 : index
# CHECK-NEXT:                 %c3_38 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_39 = arith.constant 1 : index
# CHECK-NEXT:                 %8 = scf.for %arg15 = %c0_37 to %c3_38 step %c1_39 iter_args(%arg16 = %extracted_slice_36) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                   %extracted_slice_41 = tensor.extract_slice %extracted_slice_34[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_42 = tensor.extract_slice %extracted_slice_35[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_43 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_41, %extracted_slice_42 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_43 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                   ^bb0(%in: f32, %in_45: f32, %out: f32):
# CHECK-NEXT:                     %10 = arith.mulf %in, %in_45 : f32
# CHECK-NEXT:                     %11 = arith.addf %out, %10 : f32
# CHECK-NEXT:                     linalg.yield %11 : f32
# CHECK-NEXT:                   } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %inserted_slice_44 = tensor.insert_slice %9 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_44 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %inserted_slice_40 = tensor.insert_slice %8 into %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_40 : tensor<1x1x1x1xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               %inserted_slice_33 = tensor.insert_slice %7 into %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_33 : tensor<1x1x1x1xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_26 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_26 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_20 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_20 : tensor<1x1x8x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x8x8x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_mini(%arg0: tensor<1x10x10x3xf32> {llvm.noalias}, %arg1: tensor<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x8x8x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c1_0 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %0) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:       %c0_4 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %c1_5 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_4 to %c8 step %c1_5 iter_args(%arg6 = %extracted_slice) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:         %c0_7 = arith.constant 0 : index
# CHECK-NEXT:         %c8_8 = arith.constant 8 : index
# CHECK-NEXT:         %c1_9 = arith.constant 1 : index
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0_7 to %c8_8 step %c1_9 iter_args(%arg8 = %extracted_slice_6) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:           %extracted_slice_11 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_12 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_13 = arith.constant 1 : index
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0_12 to %c16 step %c1_13 iter_args(%arg10 = %extracted_slice_11) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_15 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %6 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_15 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_16 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_16 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_14 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_14 : tensor<1x1x8x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_10 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_10 : tensor<1x8x8x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %c0_1 = arith.constant 0 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0_1 to %c1_2 step %c1_3 iter_args(%arg4 = %1) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 10, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x10x10x3xf32>
# CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %arg1[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x8x8x16xf32>
# CHECK-NEXT:       %c0_6 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %c1_7 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_6 to %c8 step %c1_7 iter_args(%arg6 = %extracted_slice_5) -> (tensor<1x8x8x16xf32>) {
# CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %extracted_slice[0, %arg5, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : tensor<1x10x10x3xf32> to tensor<1x3x10x3xf32>
# CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> to tensor<1x1x8x16xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c8_12 = arith.constant 8 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0_11 to %c8_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x8x16xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %extracted_slice_8[0, 0, %arg7, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x10x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:           %extracted_slice_16 = tensor.extract_slice %extracted_slice_9[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x16xf32>
# CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_18 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_19 = arith.constant 1 : index
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0_18 to %c16 step %c1_19 iter_args(%arg10 = %extracted_slice_17) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 3, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x3x3x3xf32>
# CHECK-NEXT:             %extracted_slice_22 = tensor.extract_slice %extracted_slice_16[0, 0, 0, %arg9] [3, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<3x3x3x1xf32>
# CHECK-NEXT:             %extracted_slice_23 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %c0_24 = arith.constant 0 : index
# CHECK-NEXT:             %c3 = arith.constant 3 : index
# CHECK-NEXT:             %c1_25 = arith.constant 1 : index
# CHECK-NEXT:             %6 = scf.for %arg11 = %c0_24 to %c3 step %c1_25 iter_args(%arg12 = %extracted_slice_23) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:               %extracted_slice_27 = tensor.extract_slice %extracted_slice_21[0, %arg11, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x3x3x3xf32> to tensor<1x1x3x3xf32>
# CHECK-NEXT:               %extracted_slice_28 = tensor.extract_slice %extracted_slice_22[%arg11, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<3x3x3x1xf32> to tensor<1x3x3x1xf32>
# CHECK-NEXT:               %extracted_slice_29 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:               %c0_30 = arith.constant 0 : index
# CHECK-NEXT:               %c3_31 = arith.constant 3 : index
# CHECK-NEXT:               %c1_32 = arith.constant 1 : index
# CHECK-NEXT:               %7 = scf.for %arg13 = %c0_30 to %c3_31 step %c1_32 iter_args(%arg14 = %extracted_slice_29) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %extracted_slice_27[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x3x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %extracted_slice_35 = tensor.extract_slice %extracted_slice_28[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %c0_37 = arith.constant 0 : index
# CHECK-NEXT:                 %c3_38 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_39 = arith.constant 1 : index
# CHECK-NEXT:                 %8 = scf.for %arg15 = %c0_37 to %c3_38 step %c1_39 iter_args(%arg16 = %extracted_slice_36) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                   %extracted_slice_41 = tensor.extract_slice %extracted_slice_34[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_42 = tensor.extract_slice %extracted_slice_35[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_43 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_41, %extracted_slice_42 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_43 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                   ^bb0(%in: f32, %in_45: f32, %out: f32):
# CHECK-NEXT:                     %10 = arith.mulf %in, %in_45 : f32
# CHECK-NEXT:                     %11 = arith.addf %out, %10 : f32
# CHECK-NEXT:                     linalg.yield %11 : f32
# CHECK-NEXT:                   } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %inserted_slice_44 = tensor.insert_slice %9 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_44 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %inserted_slice_40 = tensor.insert_slice %8 into %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_40 : tensor<1x1x1x1xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               %inserted_slice_33 = tensor.insert_slice %7 into %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_33 : tensor<1x1x1x1xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_26 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_26 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_20 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x8x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_20 : tensor<1x1x8x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : tensor<1x1x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x8x8x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : tensor<1x8x8x16xf32> into tensor<1x8x8x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x8x8x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x8x8x16xf32>, memref<1x8x8x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_mini(%arg0: memref<1x10x10x3xf32> {llvm.noalias}, %arg1: memref<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x8x8x16xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %subview) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         %3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_1) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_3 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_3 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>)
# CHECK-NEXT:           %subview_4 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_3, %subview_4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %3, %subview_2 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_0 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x8x8x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %0) -> (memref<1x8x8x16xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : memref<1x10x10x3xf32> to memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %subview[0, 0, %arg5, 0] [1, 3, 3, 3] [1, 1, 1, 1] : memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         %3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_3) -> (memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_5 = memref.subview %arg1[0, 0, 0, %arg7] [3, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x16xf32> to memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           %4 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %subview_6) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:             %subview_8 = memref.subview %subview_2[0, %arg9, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:             %subview_9 = memref.subview %subview_5[%arg9, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:             %5 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:               %subview_10 = memref.subview %subview_8[0, 0, %arg11, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:               %subview_11 = memref.subview %subview_9[0, %arg11, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:               %6 = scf.for %arg13 = %c0 to %c3 step %c1 iter_args(%arg14 = %arg12) -> (memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) {
# CHECK-NEXT:                 %subview_12 = memref.subview %subview_10[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_13 = memref.subview %subview_11[0, 0, %arg13, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_12, %subview_13 : memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>) outs(%arg14 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_14: f32, %out: f32):
# CHECK-NEXT:                   %7 = arith.mulf %in, %in_14 : f32
# CHECK-NEXT:                   %8 = arith.addf %out, %7 : f32
# CHECK-NEXT:                   linalg.yield %8 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 scf.yield %arg14 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:               } {"./c"}
# CHECK-NEXT:               scf.yield %6 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:             } {"./s"}
# CHECK-NEXT:             scf.yield %5 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           } {"./r"}
# CHECK-NEXT:           %subview_7 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %4, %subview_7 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %3, %subview_4 : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_1 : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x8x8x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     memref.copy %1, %arg2 : memref<1x8x8x16xf32> to memref<1x8x8x16xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: conv2d_nhwc_mini
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:   - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %2 : 1x8x8x16xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'O'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
