# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_nhwc_mini_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x12x12x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:     ^bb0(%out: f32):
# CHECK-NEXT:       %5 = linalg.index 0 : index
# CHECK-NEXT:       %6 = linalg.index 1 : index
# CHECK-NEXT:       %7 = linalg.index 2 : index
# CHECK-NEXT:       %8 = linalg.index 3 : index
# CHECK-NEXT:       %c0 = arith.constant 0 : index
# CHECK-NEXT:       %c0_1 = arith.constant 0 : index
# CHECK-NEXT:       %9 = arith.subi %5, %c0_1 : index
# CHECK-NEXT:       %c1 = arith.constant 1 : index
# CHECK-NEXT:       %10 = arith.cmpi sge, %9, %c0 : index
# CHECK-NEXT:       %11 = arith.cmpi slt, %9, %c1 : index
# CHECK-NEXT:       %c2 = arith.constant 2 : index
# CHECK-NEXT:       %12 = arith.subi %6, %c2 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %13 = arith.cmpi sge, %12, %c0 : index
# CHECK-NEXT:       %14 = arith.cmpi slt, %12, %c8 : index
# CHECK-NEXT:       %c2_2 = arith.constant 2 : index
# CHECK-NEXT:       %15 = arith.subi %7, %c2_2 : index
# CHECK-NEXT:       %c8_3 = arith.constant 8 : index
# CHECK-NEXT:       %16 = arith.cmpi sge, %15, %c0 : index
# CHECK-NEXT:       %17 = arith.cmpi slt, %15, %c8_3 : index
# CHECK-NEXT:       %c0_4 = arith.constant 0 : index
# CHECK-NEXT:       %18 = arith.subi %8, %c0_4 : index
# CHECK-NEXT:       %c3 = arith.constant 3 : index
# CHECK-NEXT:       %19 = arith.cmpi sge, %18, %c0 : index
# CHECK-NEXT:       %20 = arith.cmpi slt, %18, %c3 : index
# CHECK-NEXT:       %21 = arith.andi %10, %11 : i1
# CHECK-NEXT:       %22 = arith.andi %21, %13 : i1
# CHECK-NEXT:       %23 = arith.andi %22, %14 : i1
# CHECK-NEXT:       %24 = arith.andi %23, %16 : i1
# CHECK-NEXT:       %25 = arith.andi %24, %17 : i1
# CHECK-NEXT:       %26 = arith.andi %25, %19 : i1
# CHECK-NEXT:       %27 = arith.andi %26, %20 : i1
# CHECK-NEXT:       %28 = scf.if %27 -> (f32) {
# CHECK-NEXT:         %extracted = tensor.extract %arg0[%9, %12, %15, %18] : tensor<1x8x8x3xf32>
# CHECK-NEXT:         scf.yield %extracted : f32
# CHECK-NEXT:       } else {
# CHECK-NEXT:         scf.yield %cst : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       linalg.yield %28 : f32
# CHECK-NEXT:     } -> tensor<1x12x12x3xf32>
# CHECK-NEXT:     %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%1, %arg1 : tensor<1x12x12x3xf32>, tensor<5x5x3x16xf32>) outs(%3 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:       %5 = arith.mulf %in, %in_1 : f32
# CHECK-NEXT:       %6 = arith.addf %out, %5 : f32
# CHECK-NEXT:       linalg.yield %6 : f32
# CHECK-NEXT:     } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./c" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_conv_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_19 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_21 "./f" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_22, %loops_23 = transform.structured.tile_using_for %tiled_linalg_op_20 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_23 "./r" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_24, %loops_25 = transform.structured.tile_using_for %tiled_linalg_op_22 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_25 "./s" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_26, %loops_27 = transform.structured.tile_using_for %tiled_linalg_op_24 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_27 "./c" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map1 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c1_0 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %0) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 12, 12, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x12x12x3xf32>
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c12 = arith.constant 12 : index
# CHECK-NEXT:       %c1_9 = arith.constant 1 : index
# CHECK-NEXT:       %5 = scf.for %arg5 = %c0_8 to %c12 step %c1_9 iter_args(%arg6 = %extracted_slice) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x1x12x3xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c12_12 = arith.constant 12 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0_11 to %c12_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x12x3xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x12x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:           %c0_16 = arith.constant 0 : index
# CHECK-NEXT:           %c3 = arith.constant 3 : index
# CHECK-NEXT:           %c1_17 = arith.constant 1 : index
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0_16 to %c3 step %c1_17 iter_args(%arg10 = %extracted_slice_15) -> (tensor<1x1x1x3xf32>) {
# CHECK-NEXT:             %extracted_slice_19 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_19 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:             ^bb0(%out: f32):
# CHECK-NEXT:               %9 = linalg.index 0 : index
# CHECK-NEXT:               %10 = affine.apply #map1(%arg3)[%9]
# CHECK-NEXT:               %11 = linalg.index 1 : index
# CHECK-NEXT:               %12 = affine.apply #map1(%arg5)[%11]
# CHECK-NEXT:               %13 = linalg.index 2 : index
# CHECK-NEXT:               %14 = affine.apply #map1(%arg7)[%13]
# CHECK-NEXT:               %15 = linalg.index 3 : index
# CHECK-NEXT:               %16 = affine.apply #map1(%arg9)[%15]
# CHECK-NEXT:               %c0_21 = arith.constant 0 : index
# CHECK-NEXT:               %c0_22 = arith.constant 0 : index
# CHECK-NEXT:               %17 = arith.subi %10, %c0_22 : index
# CHECK-NEXT:               %c1_23 = arith.constant 1 : index
# CHECK-NEXT:               %18 = arith.cmpi sge, %17, %c0_21 : index
# CHECK-NEXT:               %19 = arith.cmpi slt, %17, %c1_23 : index
# CHECK-NEXT:               %c2 = arith.constant 2 : index
# CHECK-NEXT:               %20 = arith.subi %12, %c2 : index
# CHECK-NEXT:               %c8 = arith.constant 8 : index
# CHECK-NEXT:               %21 = arith.cmpi sge, %20, %c0_21 : index
# CHECK-NEXT:               %22 = arith.cmpi slt, %20, %c8 : index
# CHECK-NEXT:               %c2_24 = arith.constant 2 : index
# CHECK-NEXT:               %23 = arith.subi %14, %c2_24 : index
# CHECK-NEXT:               %c8_25 = arith.constant 8 : index
# CHECK-NEXT:               %24 = arith.cmpi sge, %23, %c0_21 : index
# CHECK-NEXT:               %25 = arith.cmpi slt, %23, %c8_25 : index
# CHECK-NEXT:               %c0_26 = arith.constant 0 : index
# CHECK-NEXT:               %26 = arith.subi %16, %c0_26 : index
# CHECK-NEXT:               %c3_27 = arith.constant 3 : index
# CHECK-NEXT:               %27 = arith.cmpi sge, %26, %c0_21 : index
# CHECK-NEXT:               %28 = arith.cmpi slt, %26, %c3_27 : index
# CHECK-NEXT:               %29 = arith.andi %18, %19 : i1
# CHECK-NEXT:               %30 = arith.andi %29, %21 : i1
# CHECK-NEXT:               %31 = arith.andi %30, %22 : i1
# CHECK-NEXT:               %32 = arith.andi %31, %24 : i1
# CHECK-NEXT:               %33 = arith.andi %32, %25 : i1
# CHECK-NEXT:               %34 = arith.andi %33, %27 : i1
# CHECK-NEXT:               %35 = arith.andi %34, %28 : i1
# CHECK-NEXT:               %36 = scf.if %35 -> (f32) {
# CHECK-NEXT:                 %extracted = tensor.extract %arg0[%17, %20, %23, %26] : tensor<1x8x8x3xf32>
# CHECK-NEXT:                 scf.yield %extracted : f32
# CHECK-NEXT:               } else {
# CHECK-NEXT:                 scf.yield %cst : f32
# CHECK-NEXT:               }
# CHECK-NEXT:               linalg.yield %36 : f32
# CHECK-NEXT:             } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_20 = tensor.insert_slice %8 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x3xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_20 : tensor<1x1x1x3xf32>
# CHECK-NEXT:           } {"./c"}
# CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %7 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x1x3xf32> into tensor<1x1x12x3xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_18 : tensor<1x1x12x3xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : tensor<1x1x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x12x12x3xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0, 0, 0] [1, 12, 12, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x12x12x3xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_2 = arith.constant 0 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %c1_4 = arith.constant 1 : index
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0_2 to %c1_3 step %c1_4 iter_args(%arg4 = %2) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_9 = arith.constant 1 : index
# CHECK-NEXT:       %5 = scf.for %arg5 = %c0_8 to %c4 step %c1_9 iter_args(%arg6 = %extracted_slice) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c4_12 = arith.constant 4 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0_11 to %c4_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_16 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_17 = arith.constant 1 : index
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0_16 to %c16 step %c1_17 iter_args(%arg10 = %extracted_slice_15) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_19 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %8 = linalg.fill {__xtc_id_conv_0_} ins(%cst_1 : f32) outs(%extracted_slice_19 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_20 = tensor.insert_slice %8 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_20 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %7 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_18 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %6 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %c0_5 = arith.constant 0 : index
# CHECK-NEXT:     %c1_6 = arith.constant 1 : index
# CHECK-NEXT:     %c1_7 = arith.constant 1 : index
# CHECK-NEXT:     %4 = scf.for %arg3 = %c0_5 to %c1_6 step %c1_7 iter_args(%arg4 = %3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %1[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:       %extracted_slice_8 = tensor.extract_slice %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_10 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %5 = scf.for %arg5 = %c0_10 to %c4 step %c1_11 iter_args(%arg6 = %extracted_slice_9) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %6 = affine.apply #map2(%arg5)
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, %6, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:         %extracted_slice_13 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_15 = arith.constant 0 : index
# CHECK-NEXT:         %c4_16 = arith.constant 4 : index
# CHECK-NEXT:         %c1_17 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg7 = %c0_15 to %c4_16 step %c1_17 iter_args(%arg8 = %extracted_slice_14) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %8 = affine.apply #map2(%arg7)
# CHECK-NEXT:           %extracted_slice_19 = tensor.extract_slice %extracted_slice_12[0, 0, %8, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_22 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_23 = arith.constant 1 : index
# CHECK-NEXT:           %9 = scf.for %arg9 = %c0_22 to %c16 step %c1_23 iter_args(%arg10 = %extracted_slice_21) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_25 = tensor.extract_slice %extracted_slice_19[0, 0, 0, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:             %extracted_slice_26 = tensor.extract_slice %extracted_slice_20[0, 0, 0, %arg9] [5, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x1xf32>
# CHECK-NEXT:             %extracted_slice_27 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %c0_28 = arith.constant 0 : index
# CHECK-NEXT:             %c5 = arith.constant 5 : index
# CHECK-NEXT:             %c1_29 = arith.constant 1 : index
# CHECK-NEXT:             %10 = scf.for %arg11 = %c0_28 to %c5 step %c1_29 iter_args(%arg12 = %extracted_slice_27) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:               %extracted_slice_31 = tensor.extract_slice %extracted_slice_25[0, %arg11, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x1x5x3xf32>
# CHECK-NEXT:               %extracted_slice_32 = tensor.extract_slice %extracted_slice_26[%arg11, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x1xf32> to tensor<1x5x3x1xf32>
# CHECK-NEXT:               %extracted_slice_33 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:               %c0_34 = arith.constant 0 : index
# CHECK-NEXT:               %c5_35 = arith.constant 5 : index
# CHECK-NEXT:               %c1_36 = arith.constant 1 : index
# CHECK-NEXT:               %11 = scf.for %arg13 = %c0_34 to %c5_35 step %c1_36 iter_args(%arg14 = %extracted_slice_33) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                 %extracted_slice_38 = tensor.extract_slice %extracted_slice_31[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x5x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_32[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x5x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %c0_41 = arith.constant 0 : index
# CHECK-NEXT:                 %c3 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_42 = arith.constant 1 : index
# CHECK-NEXT:                 %12 = scf.for %arg15 = %c0_41 to %c3 step %c1_42 iter_args(%arg16 = %extracted_slice_40) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                   %extracted_slice_44 = tensor.extract_slice %extracted_slice_38[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_45 = tensor.extract_slice %extracted_slice_39[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_46 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %13 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_45 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_46 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                   ^bb0(%in: f32, %in_48: f32, %out: f32):
# CHECK-NEXT:                     %14 = arith.mulf %in, %in_48 : f32
# CHECK-NEXT:                     %15 = arith.addf %out, %14 : f32
# CHECK-NEXT:                     linalg.yield %15 : f32
# CHECK-NEXT:                   } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %inserted_slice_47 = tensor.insert_slice %13 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_47 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %inserted_slice_43 = tensor.insert_slice %12 into %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_43 : tensor<1x1x1x1xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               %inserted_slice_37 = tensor.insert_slice %11 into %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_37 : tensor<1x1x1x1xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_30 = tensor.insert_slice %10 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_30 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_24 = tensor.insert_slice %9 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_24 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %7 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c5 = arith.constant 5 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c12 = arith.constant 12 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%arg4 = %alloc) -> (memref<1x12x12x3xf32>) {
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c12 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %subview_2) -> (memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_4 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:           linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_4 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:           ^bb0(%out: f32):
# CHECK-NEXT:             %5 = arith.subi %arg3, %c2 : index
# CHECK-NEXT:             %6 = arith.cmpi sge, %5, %c0 : index
# CHECK-NEXT:             %7 = arith.cmpi slt, %5, %c8 : index
# CHECK-NEXT:             %8 = arith.subi %arg5, %c2 : index
# CHECK-NEXT:             %9 = arith.cmpi sge, %8, %c0 : index
# CHECK-NEXT:             %10 = arith.cmpi slt, %8, %c8 : index
# CHECK-NEXT:             %11 = arith.cmpi sge, %arg7, %c0 : index
# CHECK-NEXT:             %12 = arith.cmpi slt, %arg7, %c3 : index
# CHECK-NEXT:             %13 = arith.andi %6, %7 : i1
# CHECK-NEXT:             %14 = arith.andi %13, %9 : i1
# CHECK-NEXT:             %15 = arith.andi %14, %10 : i1
# CHECK-NEXT:             %16 = arith.andi %15, %11 : i1
# CHECK-NEXT:             %17 = arith.andi %16, %12 : i1
# CHECK-NEXT:             %18 = scf.if %17 -> (f32) {
# CHECK-NEXT:               %19 = memref.load %arg0[%c0, %5, %8, %arg7] : memref<1x8x8x3xf32>
# CHECK-NEXT:               scf.yield %19 : f32
# CHECK-NEXT:             } else {
# CHECK-NEXT:               scf.yield %cst : f32
# CHECK-NEXT:             }
# CHECK-NEXT:             linalg.yield %18 : f32
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_5 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_4, %subview_5 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         } {"./c"}
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %4, %subview_3 : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %3, %subview_1 : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x12x12x3xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_2) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_4 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%subview_4 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>)
# CHECK-NEXT:           %subview_5 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_4, %subview_5 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %4, %subview_3 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %3, %subview_1 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     %subview = memref.subview %0[0, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>>
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:       %3 = affine.apply #map1(%arg3)
# CHECK-NEXT:       %subview_0 = memref.subview %subview[0, %3, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>> to memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_1) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %5 = affine.apply #map1(%arg5)
# CHECK-NEXT:         %subview_3 = memref.subview %subview_0[0, 0, %5, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_4) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_6 = memref.subview %arg1[0, 0, 0, %arg7] [5, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_7 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0 to %c5 step %c1 iter_args(%arg10 = %subview_7) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:             %subview_9 = memref.subview %subview_3[0, %arg9, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:             %subview_10 = memref.subview %subview_6[%arg9, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:             %8 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:               %subview_11 = memref.subview %subview_9[0, 0, %arg11, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:               %subview_12 = memref.subview %subview_10[0, %arg11, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:               %9 = scf.for %arg13 = %c0 to %c3 step %c1 iter_args(%arg14 = %arg12) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                 %subview_13 = memref.subview %subview_11[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_14 = memref.subview %subview_12[0, 0, %arg13, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_13, %subview_14 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%arg14 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_15: f32, %out: f32):
# CHECK-NEXT:                   %10 = arith.mulf %in, %in_15 : f32
# CHECK-NEXT:                   %11 = arith.addf %out, %10 : f32
# CHECK-NEXT:                   linalg.yield %11 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 scf.yield %arg14 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:               } {"./c"}
# CHECK-NEXT:               scf.yield %9 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:             } {"./s"}
# CHECK-NEXT:             scf.yield %8 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           } {"./r"}
# CHECK-NEXT:           %subview_8 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %7, %subview_8 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_5 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %6, %subview_5 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_2 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     memref.copy %2, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: pad_conv2d_nhwc_mini
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:   - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %3 : 1x4x4x16xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:   - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
