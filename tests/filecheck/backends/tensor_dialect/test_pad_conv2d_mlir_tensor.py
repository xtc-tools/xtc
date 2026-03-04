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
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %padded = tensor.pad %arg0 nofold low[0, 2, 2, 0] high[0, 2, 2, 0] {
# CHECK-NEXT:     ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
# CHECK-NEXT:       tensor.yield %cst : f32
# CHECK-NEXT:     } {__xtc_id_pad_} : tensor<1x8x8x3xf32> to tensor<1x12x12x3xf32>
# CHECK-NEXT:     %1 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %2 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%1 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<1x12x12x3xf32>, tensor<5x5x3x16xf32>) outs(%2 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:       %4 = arith.mulf %in, %in_1 : f32
# CHECK-NEXT:       %5 = arith.addf %out, %4 : f32
# CHECK-NEXT:       linalg.yield %5 : f32
# CHECK-NEXT:     } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
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
# CHECK-NEXT: #map = affine_map<(d0) -> (-d0 + 2)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> (0, -d0 + 2)>
# CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 - 2)>
# CHECK-NEXT: #map3 = affine_map<(d0) -> (d0 - 2, 0)>
# CHECK-NEXT: #map4 = affine_map<(d0) -> (d0, 8)>
# CHECK-NEXT: #map5 = affine_map<(d0) -> (-d0 + 1)>
# CHECK-NEXT: #map6 = affine_map<(d0) -> (-d0 + 8)>
# CHECK-NEXT: #map7 = affine_map<(d0, d1) -> (-d0 + 8, -d1 + 1)>
# CHECK-NEXT: #map8 = affine_map<(d0) -> (d0, 0)>
# CHECK-NEXT: #map9 = affine_map<(d0, d1) -> (-d0 - d1 + 1)>
# CHECK-NEXT: #map10 = affine_map<(d0) -> (0, d0)>
# CHECK-NEXT: #map11 = affine_map<(d0) -> (-d0)>
# CHECK-NEXT: #map12 = affine_map<(d0) -> (-d0, 0)>
# CHECK-NEXT: #map13 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT: #map14 = affine_map<(d0, d1) -> (d0 - d1)>
# CHECK-NEXT: #map15 = affine_map<(d0, d1, d2) -> (d0 - d1, -d2 + 1)>
# CHECK-NEXT: #map16 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map17 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map18 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map19 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c1_0 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %1) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %6 = arith.cmpi eq, %c8, %c0_8 : index
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c8_10 = arith.constant 8 : index
# CHECK-NEXT:       %7 = arith.cmpi eq, %c8_10, %c0_9 : index
# CHECK-NEXT:       %8 = arith.ori %7, %6 : i1
# CHECK-NEXT:       %9 = scf.if %8 -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:         %generated = tensor.generate  {
# CHECK-NEXT:         ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
# CHECK-NEXT:           tensor.yield %cst : f32
# CHECK-NEXT:         } : tensor<1x12x12x3xf32>
# CHECK-NEXT:         scf.yield %generated : tensor<1x12x12x3xf32>
# CHECK-NEXT:       } else {
# CHECK-NEXT:         %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 8, 8, 3] [1, 1, 1, 1] : tensor<1x8x8x3xf32> to tensor<1x8x8x3xf32>
# CHECK-NEXT:         %10 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c12 = arith.constant 12 : index
# CHECK-NEXT:         %c1_12 = arith.constant 1 : index
# CHECK-NEXT:         %11 = scf.for %arg5 = %c0_11 to %c12 step %c1_12 iter_args(%arg6 = %10) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:           %12 = affine.apply #map(%arg5)
# CHECK-NEXT:           %13 = affine.max #map1(%arg5)
# CHECK-NEXT:           %14 = affine.apply #map2(%arg5)
# CHECK-NEXT:           %15 = affine.max #map3(%arg5)
# CHECK-NEXT:           %16 = affine.min #map4(%15)
# CHECK-NEXT:           %17 = affine.apply #map5(%13)
# CHECK-NEXT:           %18 = affine.apply #map6(%16)
# CHECK-NEXT:           %19 = affine.min #map7(%16, %13)
# CHECK-NEXT:           %20 = affine.max #map8(%19)
# CHECK-NEXT:           %c0_13 = arith.constant 0 : index
# CHECK-NEXT:           %21 = arith.cmpi eq, %20, %c0_13 : index
# CHECK-NEXT:           %22 = affine.apply #map5(%20)
# CHECK-NEXT:           %23 = affine.apply #map9(%13, %20)
# CHECK-NEXT:           %c0_14 = arith.constant 0 : index
# CHECK-NEXT:           %c8_15 = arith.constant 8 : index
# CHECK-NEXT:           %24 = arith.cmpi eq, %c8_15, %c0_14 : index
# CHECK-NEXT:           %25 = arith.ori %24, %21 : i1
# CHECK-NEXT:           %26 = scf.if %25 -> (tensor<1x1x12x3xf32>) {
# CHECK-NEXT:             %generated = tensor.generate  {
# CHECK-NEXT:             ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
# CHECK-NEXT:               tensor.yield %cst : f32
# CHECK-NEXT:             } : tensor<1x1x12x3xf32>
# CHECK-NEXT:             scf.yield %generated : tensor<1x1x12x3xf32>
# CHECK-NEXT:           } else {
# CHECK-NEXT:             %extracted_slice_17 = tensor.extract_slice %extracted_slice[0, %16, 0, 0] [1, %20, 8, 3] [1, 1, 1, 1] : tensor<1x8x8x3xf32> to tensor<1x?x8x3xf32>
# CHECK-NEXT:             %c1_18 = arith.constant 1 : index
# CHECK-NEXT:             %27 = tensor.empty() : tensor<1x1x12x3xf32>
# CHECK-NEXT:             %c1_19 = arith.constant 1 : index
# CHECK-NEXT:             %c0_20 = arith.constant 0 : index
# CHECK-NEXT:             %c12_21 = arith.constant 12 : index
# CHECK-NEXT:             %c1_22 = arith.constant 1 : index
# CHECK-NEXT:             %28 = scf.for %arg7 = %c0_20 to %c12_21 step %c1_22 iter_args(%arg8 = %27) -> (tensor<1x1x12x3xf32>) {
# CHECK-NEXT:               %c1_23 = arith.constant 1 : index
# CHECK-NEXT:               %29 = affine.max #map10(%13)
# CHECK-NEXT:               %30 = affine.apply #map11(%13)
# CHECK-NEXT:               %31 = affine.max #map12(%13)
# CHECK-NEXT:               %32 = affine.min #map13(%31, %20)
# CHECK-NEXT:               %33 = affine.apply #map5(%29)
# CHECK-NEXT:               %34 = affine.apply #map14(%20, %32)
# CHECK-NEXT:               %35 = affine.min #map15(%20, %32, %29)
# CHECK-NEXT:               %36 = affine.max #map8(%35)
# CHECK-NEXT:               %c0_24 = arith.constant 0 : index
# CHECK-NEXT:               %37 = arith.cmpi eq, %36, %c0_24 : index
# CHECK-NEXT:               %38 = affine.apply #map5(%36)
# CHECK-NEXT:               %39 = affine.apply #map9(%29, %36)
# CHECK-NEXT:               %40 = affine.apply #map(%arg7)
# CHECK-NEXT:               %41 = affine.max #map1(%arg7)
# CHECK-NEXT:               %42 = affine.apply #map2(%arg7)
# CHECK-NEXT:               %43 = affine.max #map3(%arg7)
# CHECK-NEXT:               %44 = affine.min #map4(%43)
# CHECK-NEXT:               %45 = affine.apply #map5(%41)
# CHECK-NEXT:               %46 = affine.apply #map6(%44)
# CHECK-NEXT:               %47 = affine.min #map7(%44, %41)
# CHECK-NEXT:               %48 = affine.max #map8(%47)
# CHECK-NEXT:               %c0_25 = arith.constant 0 : index
# CHECK-NEXT:               %49 = arith.cmpi eq, %48, %c0_25 : index
# CHECK-NEXT:               %50 = arith.ori %49, %37 : i1
# CHECK-NEXT:               %51 = affine.apply #map5(%48)
# CHECK-NEXT:               %52 = affine.apply #map9(%41, %48)
# CHECK-NEXT:               %53 = scf.if %50 -> (tensor<1x1x1x3xf32>) {
# CHECK-NEXT:                 %generated = tensor.generate  {
# CHECK-NEXT:                 ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
# CHECK-NEXT:                   tensor.yield %cst : f32
# CHECK-NEXT:                 } : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 scf.yield %generated : tensor<1x1x1x3xf32>
# CHECK-NEXT:               } else {
# CHECK-NEXT:                 %extracted_slice_27 = tensor.extract_slice %extracted_slice_17[0, %32, %44, 0] [1, %36, %48, 3] [1, 1, 1, 1] : tensor<1x?x8x3xf32> to tensor<1x?x?x3xf32>
# CHECK-NEXT:                 %c1_28 = arith.constant 1 : index
# CHECK-NEXT:                 %c2 = arith.constant 2 : index
# CHECK-NEXT:                 %54 = tensor.empty() : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %c1_29 = arith.constant 1 : index
# CHECK-NEXT:                 %c2_30 = arith.constant 2 : index
# CHECK-NEXT:                 %c0_31 = arith.constant 0 : index
# CHECK-NEXT:                 %c3 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_32 = arith.constant 1 : index
# CHECK-NEXT:                 %55 = scf.for %arg9 = %c0_31 to %c3 step %c1_32 iter_args(%arg10 = %54) -> (tensor<1x1x1x3xf32>) {
# CHECK-NEXT:                   %c1_34 = arith.constant 1 : index
# CHECK-NEXT:                   %56 = affine.max #map10(%29)
# CHECK-NEXT:                   %57 = affine.apply #map11(%29)
# CHECK-NEXT:                   %58 = affine.max #map12(%29)
# CHECK-NEXT:                   %59 = affine.min #map13(%58, %36)
# CHECK-NEXT:                   %60 = affine.apply #map5(%56)
# CHECK-NEXT:                   %61 = affine.apply #map14(%36, %59)
# CHECK-NEXT:                   %62 = affine.min #map15(%36, %59, %56)
# CHECK-NEXT:                   %63 = affine.max #map8(%62)
# CHECK-NEXT:                   %c0_35 = arith.constant 0 : index
# CHECK-NEXT:                   %64 = arith.cmpi eq, %63, %c0_35 : index
# CHECK-NEXT:                   %65 = affine.apply #map5(%63)
# CHECK-NEXT:                   %66 = affine.apply #map9(%56, %63)
# CHECK-NEXT:                   %c2_36 = arith.constant 2 : index
# CHECK-NEXT:                   %67 = affine.max #map10(%41)
# CHECK-NEXT:                   %68 = affine.apply #map11(%41)
# CHECK-NEXT:                   %69 = affine.max #map12(%41)
# CHECK-NEXT:                   %70 = affine.min #map13(%69, %48)
# CHECK-NEXT:                   %71 = affine.apply #map5(%67)
# CHECK-NEXT:                   %72 = affine.apply #map14(%48, %70)
# CHECK-NEXT:                   %73 = affine.min #map15(%48, %70, %67)
# CHECK-NEXT:                   %74 = affine.max #map8(%73)
# CHECK-NEXT:                   %c0_37 = arith.constant 0 : index
# CHECK-NEXT:                   %75 = arith.cmpi eq, %74, %c0_37 : index
# CHECK-NEXT:                   %76 = arith.ori %75, %64 : i1
# CHECK-NEXT:                   %77 = affine.apply #map5(%74)
# CHECK-NEXT:                   %78 = affine.apply #map9(%67, %74)
# CHECK-NEXT:                   %79 = scf.if %76 -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                     %generated = tensor.generate  {
# CHECK-NEXT:                     ^bb0(%arg11: index, %arg12: index, %arg13: index, %arg14: index):
# CHECK-NEXT:                       tensor.yield %cst : f32
# CHECK-NEXT:                     } : tensor<1x1x1x1xf32>
# CHECK-NEXT:                     scf.yield %generated : tensor<1x1x1x1xf32>
# CHECK-NEXT:                   } else {
# CHECK-NEXT:                     %extracted_slice_39 = tensor.extract_slice %extracted_slice_27[0, %59, %70, %arg9] [1, %63, %74, 1] [1, 1, 1, 1] : tensor<1x?x?x3xf32> to tensor<1x?x?x1xf32>
# CHECK-NEXT:                     %padded = tensor.pad %extracted_slice_39 nofold low[0, %56, %67, 0] high[0, %66, %78, 0] {
# CHECK-NEXT:                     ^bb0(%arg11: index, %arg12: index, %arg13: index, %arg14: index):
# CHECK-NEXT:                       tensor.yield %cst : f32
# CHECK-NEXT:                     } {__xtc_id_pad_} : tensor<1x?x?x1xf32> to tensor<1x?x?x1xf32>
# CHECK-NEXT:                     %cast_40 = tensor.cast %padded : tensor<1x?x?x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                     scf.yield %cast_40 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                   }
# CHECK-NEXT:                   %inserted_slice_38 = tensor.insert_slice %79 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x3xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_38 : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %cast_33 = tensor.cast %55 : tensor<1x1x1x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 scf.yield %cast_33 : tensor<1x1x1x3xf32>
# CHECK-NEXT:               }
# CHECK-NEXT:               %inserted_slice_26 = tensor.insert_slice %53 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x1x3xf32> into tensor<1x1x12x3xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_26 : tensor<1x1x12x3xf32>
# CHECK-NEXT:             } {"./w"}
# CHECK-NEXT:             %cast = tensor.cast %28 : tensor<1x1x12x3xf32> to tensor<1x1x12x3xf32>
# CHECK-NEXT:             scf.yield %cast : tensor<1x1x12x3xf32>
# CHECK-NEXT:           }
# CHECK-NEXT:           %inserted_slice_16 = tensor.insert_slice %26 into %arg6[0, %arg5, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : tensor<1x1x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_16 : tensor<1x12x12x3xf32>
# CHECK-NEXT:         } {"./h"}
# CHECK-NEXT:         scf.yield %11 : tensor<1x12x12x3xf32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %9 into %arg4[%arg3, 0, 0, 0] [1, 12, 12, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x12x12x3xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %3 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_2 = arith.constant 0 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %c1_4 = arith.constant 1 : index
# CHECK-NEXT:     %4 = scf.for %arg3 = %c0_2 to %c1_3 step %c1_4 iter_args(%arg4 = %3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_9 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0_8 to %c4 step %c1_9 iter_args(%arg6 = %extracted_slice) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c4_12 = arith.constant 4 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg7 = %c0_11 to %c4_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_16 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_17 = arith.constant 1 : index
# CHECK-NEXT:           %8 = scf.for %arg9 = %c0_16 to %c16 step %c1_17 iter_args(%arg10 = %extracted_slice_15) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_19 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %9 = linalg.fill {__xtc_id_conv_0_} ins(%cst_1 : f32) outs(%extracted_slice_19 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_20 = tensor.insert_slice %9 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_20 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %8 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_18 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %7 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %c0_5 = arith.constant 0 : index
# CHECK-NEXT:     %c1_6 = arith.constant 1 : index
# CHECK-NEXT:     %c1_7 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg3 = %c0_5 to %c1_6 step %c1_7 iter_args(%arg4 = %4) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %2[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:       %extracted_slice_8 = tensor.extract_slice %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_10 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0_10 to %c4 step %c1_11 iter_args(%arg6 = %extracted_slice_9) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %7 = affine.apply #map16(%arg5)
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, %7, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:         %extracted_slice_13 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_15 = arith.constant 0 : index
# CHECK-NEXT:         %c4_16 = arith.constant 4 : index
# CHECK-NEXT:         %c1_17 = arith.constant 1 : index
# CHECK-NEXT:         %8 = scf.for %arg7 = %c0_15 to %c4_16 step %c1_17 iter_args(%arg8 = %extracted_slice_14) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %9 = affine.apply #map16(%arg7)
# CHECK-NEXT:           %extracted_slice_19 = tensor.extract_slice %extracted_slice_12[0, 0, %9, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_22 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_23 = arith.constant 1 : index
# CHECK-NEXT:           %10 = scf.for %arg9 = %c0_22 to %c16 step %c1_23 iter_args(%arg10 = %extracted_slice_21) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_25 = tensor.extract_slice %extracted_slice_19[0, 0, 0, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:             %extracted_slice_26 = tensor.extract_slice %extracted_slice_20[0, 0, 0, %arg9] [5, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x1xf32>
# CHECK-NEXT:             %extracted_slice_27 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %c0_28 = arith.constant 0 : index
# CHECK-NEXT:             %c5 = arith.constant 5 : index
# CHECK-NEXT:             %c1_29 = arith.constant 1 : index
# CHECK-NEXT:             %11 = scf.for %arg11 = %c0_28 to %c5 step %c1_29 iter_args(%arg12 = %extracted_slice_27) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:               %extracted_slice_31 = tensor.extract_slice %extracted_slice_25[0, %arg11, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x1x5x3xf32>
# CHECK-NEXT:               %extracted_slice_32 = tensor.extract_slice %extracted_slice_26[%arg11, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x1xf32> to tensor<1x5x3x1xf32>
# CHECK-NEXT:               %extracted_slice_33 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:               %c0_34 = arith.constant 0 : index
# CHECK-NEXT:               %c5_35 = arith.constant 5 : index
# CHECK-NEXT:               %c1_36 = arith.constant 1 : index
# CHECK-NEXT:               %12 = scf.for %arg13 = %c0_34 to %c5_35 step %c1_36 iter_args(%arg14 = %extracted_slice_33) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                 %extracted_slice_38 = tensor.extract_slice %extracted_slice_31[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x5x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_32[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x5x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %c0_41 = arith.constant 0 : index
# CHECK-NEXT:                 %c3 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_42 = arith.constant 1 : index
# CHECK-NEXT:                 %13 = scf.for %arg15 = %c0_41 to %c3 step %c1_42 iter_args(%arg16 = %extracted_slice_40) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                   %extracted_slice_44 = tensor.extract_slice %extracted_slice_38[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_45 = tensor.extract_slice %extracted_slice_39[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_46 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %14 = linalg.generic {indexing_maps = [#map17, #map18, #map19], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_45 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_46 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                   ^bb0(%in: f32, %in_48: f32, %out: f32):
# CHECK-NEXT:                     %15 = arith.mulf %in, %in_48 : f32
# CHECK-NEXT:                     %16 = arith.addf %out, %15 : f32
# CHECK-NEXT:                     linalg.yield %16 : f32
# CHECK-NEXT:                   } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %inserted_slice_47 = tensor.insert_slice %14 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_47 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %inserted_slice_43 = tensor.insert_slice %13 into %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_43 : tensor<1x1x1x1xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               %inserted_slice_37 = tensor.insert_slice %12 into %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_37 : tensor<1x1x1x1xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_30 = tensor.insert_slice %11 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_30 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_24 = tensor.insert_slice %10 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_24 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %8 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (-d0 + 2)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> (0, -d0 + 2)>
# CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 - 2)>
# CHECK-NEXT: #map3 = affine_map<(d0) -> (d0 - 2, 0)>
# CHECK-NEXT: #map4 = affine_map<(d0) -> (d0, 8)>
# CHECK-NEXT: #map5 = affine_map<(d0) -> (-d0 + 1)>
# CHECK-NEXT: #map6 = affine_map<(d0) -> (-d0 + 8)>
# CHECK-NEXT: #map7 = affine_map<(d0, d1) -> (-d0 + 8, -d1 + 1)>
# CHECK-NEXT: #map8 = affine_map<(d0) -> (d0, 0)>
# CHECK-NEXT: #map9 = affine_map<(d0, d1) -> (-d0 - d1 + 1)>
# CHECK-NEXT: #map10 = affine_map<(d0) -> (0, d0)>
# CHECK-NEXT: #map11 = affine_map<(d0) -> (-d0)>
# CHECK-NEXT: #map12 = affine_map<(d0) -> (-d0, 0)>
# CHECK-NEXT: #map13 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT: #map14 = affine_map<(d0, d1) -> (d0 - d1)>
# CHECK-NEXT: #map15 = affine_map<(d0, d1, d2) -> (d0 - d1, -d2 + 1)>
# CHECK-NEXT: #map16 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map17 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map18 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map19 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c1_0 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c1 step %c1_0 iter_args(%arg4 = %1) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c8 = arith.constant 8 : index
# CHECK-NEXT:       %6 = arith.cmpi eq, %c8, %c0_8 : index
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c8_10 = arith.constant 8 : index
# CHECK-NEXT:       %7 = arith.cmpi eq, %c8_10, %c0_9 : index
# CHECK-NEXT:       %8 = arith.ori %7, %6 : i1
# CHECK-NEXT:       %9 = scf.if %8 -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:         %generated = tensor.generate  {
# CHECK-NEXT:         ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
# CHECK-NEXT:           tensor.yield %cst : f32
# CHECK-NEXT:         } : tensor<1x12x12x3xf32>
# CHECK-NEXT:         scf.yield %generated : tensor<1x12x12x3xf32>
# CHECK-NEXT:       } else {
# CHECK-NEXT:         %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 8, 8, 3] [1, 1, 1, 1] : tensor<1x8x8x3xf32> to tensor<1x8x8x3xf32>
# CHECK-NEXT:         %10 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c12 = arith.constant 12 : index
# CHECK-NEXT:         %c1_12 = arith.constant 1 : index
# CHECK-NEXT:         %11 = scf.for %arg5 = %c0_11 to %c12 step %c1_12 iter_args(%arg6 = %10) -> (tensor<1x12x12x3xf32>) {
# CHECK-NEXT:           %12 = affine.apply #map(%arg5)
# CHECK-NEXT:           %13 = affine.max #map1(%arg5)
# CHECK-NEXT:           %14 = affine.apply #map2(%arg5)
# CHECK-NEXT:           %15 = affine.max #map3(%arg5)
# CHECK-NEXT:           %16 = affine.min #map4(%15)
# CHECK-NEXT:           %17 = affine.apply #map5(%13)
# CHECK-NEXT:           %18 = affine.apply #map6(%16)
# CHECK-NEXT:           %19 = affine.min #map7(%16, %13)
# CHECK-NEXT:           %20 = affine.max #map8(%19)
# CHECK-NEXT:           %c0_13 = arith.constant 0 : index
# CHECK-NEXT:           %21 = arith.cmpi eq, %20, %c0_13 : index
# CHECK-NEXT:           %22 = affine.apply #map5(%20)
# CHECK-NEXT:           %23 = affine.apply #map9(%13, %20)
# CHECK-NEXT:           %c0_14 = arith.constant 0 : index
# CHECK-NEXT:           %c8_15 = arith.constant 8 : index
# CHECK-NEXT:           %24 = arith.cmpi eq, %c8_15, %c0_14 : index
# CHECK-NEXT:           %25 = arith.ori %24, %21 : i1
# CHECK-NEXT:           %26 = scf.if %25 -> (tensor<1x1x12x3xf32>) {
# CHECK-NEXT:             %generated = tensor.generate  {
# CHECK-NEXT:             ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
# CHECK-NEXT:               tensor.yield %cst : f32
# CHECK-NEXT:             } : tensor<1x1x12x3xf32>
# CHECK-NEXT:             scf.yield %generated : tensor<1x1x12x3xf32>
# CHECK-NEXT:           } else {
# CHECK-NEXT:             %extracted_slice_17 = tensor.extract_slice %extracted_slice[0, %16, 0, 0] [1, %20, 8, 3] [1, 1, 1, 1] : tensor<1x8x8x3xf32> to tensor<1x?x8x3xf32>
# CHECK-NEXT:             %c1_18 = arith.constant 1 : index
# CHECK-NEXT:             %27 = tensor.empty() : tensor<1x1x12x3xf32>
# CHECK-NEXT:             %c1_19 = arith.constant 1 : index
# CHECK-NEXT:             %c0_20 = arith.constant 0 : index
# CHECK-NEXT:             %c12_21 = arith.constant 12 : index
# CHECK-NEXT:             %c1_22 = arith.constant 1 : index
# CHECK-NEXT:             %28 = scf.for %arg7 = %c0_20 to %c12_21 step %c1_22 iter_args(%arg8 = %27) -> (tensor<1x1x12x3xf32>) {
# CHECK-NEXT:               %c1_23 = arith.constant 1 : index
# CHECK-NEXT:               %29 = affine.max #map10(%13)
# CHECK-NEXT:               %30 = affine.apply #map11(%13)
# CHECK-NEXT:               %31 = affine.max #map12(%13)
# CHECK-NEXT:               %32 = affine.min #map13(%31, %20)
# CHECK-NEXT:               %33 = affine.apply #map5(%29)
# CHECK-NEXT:               %34 = affine.apply #map14(%20, %32)
# CHECK-NEXT:               %35 = affine.min #map15(%20, %32, %29)
# CHECK-NEXT:               %36 = affine.max #map8(%35)
# CHECK-NEXT:               %c0_24 = arith.constant 0 : index
# CHECK-NEXT:               %37 = arith.cmpi eq, %36, %c0_24 : index
# CHECK-NEXT:               %38 = affine.apply #map5(%36)
# CHECK-NEXT:               %39 = affine.apply #map9(%29, %36)
# CHECK-NEXT:               %40 = affine.apply #map(%arg7)
# CHECK-NEXT:               %41 = affine.max #map1(%arg7)
# CHECK-NEXT:               %42 = affine.apply #map2(%arg7)
# CHECK-NEXT:               %43 = affine.max #map3(%arg7)
# CHECK-NEXT:               %44 = affine.min #map4(%43)
# CHECK-NEXT:               %45 = affine.apply #map5(%41)
# CHECK-NEXT:               %46 = affine.apply #map6(%44)
# CHECK-NEXT:               %47 = affine.min #map7(%44, %41)
# CHECK-NEXT:               %48 = affine.max #map8(%47)
# CHECK-NEXT:               %c0_25 = arith.constant 0 : index
# CHECK-NEXT:               %49 = arith.cmpi eq, %48, %c0_25 : index
# CHECK-NEXT:               %50 = arith.ori %49, %37 : i1
# CHECK-NEXT:               %51 = affine.apply #map5(%48)
# CHECK-NEXT:               %52 = affine.apply #map9(%41, %48)
# CHECK-NEXT:               %53 = scf.if %50 -> (tensor<1x1x1x3xf32>) {
# CHECK-NEXT:                 %generated = tensor.generate  {
# CHECK-NEXT:                 ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
# CHECK-NEXT:                   tensor.yield %cst : f32
# CHECK-NEXT:                 } : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 scf.yield %generated : tensor<1x1x1x3xf32>
# CHECK-NEXT:               } else {
# CHECK-NEXT:                 %extracted_slice_27 = tensor.extract_slice %extracted_slice_17[0, %32, %44, 0] [1, %36, %48, 3] [1, 1, 1, 1] : tensor<1x?x8x3xf32> to tensor<1x?x?x3xf32>
# CHECK-NEXT:                 %c1_28 = arith.constant 1 : index
# CHECK-NEXT:                 %c2 = arith.constant 2 : index
# CHECK-NEXT:                 %54 = tensor.empty() : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %c1_29 = arith.constant 1 : index
# CHECK-NEXT:                 %c2_30 = arith.constant 2 : index
# CHECK-NEXT:                 %c0_31 = arith.constant 0 : index
# CHECK-NEXT:                 %c3 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_32 = arith.constant 1 : index
# CHECK-NEXT:                 %55 = scf.for %arg9 = %c0_31 to %c3 step %c1_32 iter_args(%arg10 = %54) -> (tensor<1x1x1x3xf32>) {
# CHECK-NEXT:                   %c1_34 = arith.constant 1 : index
# CHECK-NEXT:                   %56 = affine.max #map10(%29)
# CHECK-NEXT:                   %57 = affine.apply #map11(%29)
# CHECK-NEXT:                   %58 = affine.max #map12(%29)
# CHECK-NEXT:                   %59 = affine.min #map13(%58, %36)
# CHECK-NEXT:                   %60 = affine.apply #map5(%56)
# CHECK-NEXT:                   %61 = affine.apply #map14(%36, %59)
# CHECK-NEXT:                   %62 = affine.min #map15(%36, %59, %56)
# CHECK-NEXT:                   %63 = affine.max #map8(%62)
# CHECK-NEXT:                   %c0_35 = arith.constant 0 : index
# CHECK-NEXT:                   %64 = arith.cmpi eq, %63, %c0_35 : index
# CHECK-NEXT:                   %65 = affine.apply #map5(%63)
# CHECK-NEXT:                   %66 = affine.apply #map9(%56, %63)
# CHECK-NEXT:                   %c2_36 = arith.constant 2 : index
# CHECK-NEXT:                   %67 = affine.max #map10(%41)
# CHECK-NEXT:                   %68 = affine.apply #map11(%41)
# CHECK-NEXT:                   %69 = affine.max #map12(%41)
# CHECK-NEXT:                   %70 = affine.min #map13(%69, %48)
# CHECK-NEXT:                   %71 = affine.apply #map5(%67)
# CHECK-NEXT:                   %72 = affine.apply #map14(%48, %70)
# CHECK-NEXT:                   %73 = affine.min #map15(%48, %70, %67)
# CHECK-NEXT:                   %74 = affine.max #map8(%73)
# CHECK-NEXT:                   %c0_37 = arith.constant 0 : index
# CHECK-NEXT:                   %75 = arith.cmpi eq, %74, %c0_37 : index
# CHECK-NEXT:                   %76 = arith.ori %75, %64 : i1
# CHECK-NEXT:                   %77 = affine.apply #map5(%74)
# CHECK-NEXT:                   %78 = affine.apply #map9(%67, %74)
# CHECK-NEXT:                   %79 = scf.if %76 -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                     %generated = tensor.generate  {
# CHECK-NEXT:                     ^bb0(%arg11: index, %arg12: index, %arg13: index, %arg14: index):
# CHECK-NEXT:                       tensor.yield %cst : f32
# CHECK-NEXT:                     } : tensor<1x1x1x1xf32>
# CHECK-NEXT:                     scf.yield %generated : tensor<1x1x1x1xf32>
# CHECK-NEXT:                   } else {
# CHECK-NEXT:                     %extracted_slice_39 = tensor.extract_slice %extracted_slice_27[0, %59, %70, %arg9] [1, %63, %74, 1] [1, 1, 1, 1] : tensor<1x?x?x3xf32> to tensor<1x?x?x1xf32>
# CHECK-NEXT:                     %padded = tensor.pad %extracted_slice_39 nofold low[0, %56, %67, 0] high[0, %66, %78, 0] {
# CHECK-NEXT:                     ^bb0(%arg11: index, %arg12: index, %arg13: index, %arg14: index):
# CHECK-NEXT:                       tensor.yield %cst : f32
# CHECK-NEXT:                     } {__xtc_id_pad_} : tensor<1x?x?x1xf32> to tensor<1x?x?x1xf32>
# CHECK-NEXT:                     %cast_40 = tensor.cast %padded : tensor<1x?x?x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                     scf.yield %cast_40 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                   }
# CHECK-NEXT:                   %inserted_slice_38 = tensor.insert_slice %79 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x3xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_38 : tensor<1x1x1x3xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %cast_33 = tensor.cast %55 : tensor<1x1x1x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 scf.yield %cast_33 : tensor<1x1x1x3xf32>
# CHECK-NEXT:               }
# CHECK-NEXT:               %inserted_slice_26 = tensor.insert_slice %53 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x1x3xf32> into tensor<1x1x12x3xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_26 : tensor<1x1x12x3xf32>
# CHECK-NEXT:             } {"./w"}
# CHECK-NEXT:             %cast = tensor.cast %28 : tensor<1x1x12x3xf32> to tensor<1x1x12x3xf32>
# CHECK-NEXT:             scf.yield %cast : tensor<1x1x12x3xf32>
# CHECK-NEXT:           }
# CHECK-NEXT:           %inserted_slice_16 = tensor.insert_slice %26 into %arg6[0, %arg5, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : tensor<1x1x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_16 : tensor<1x12x12x3xf32>
# CHECK-NEXT:         } {"./h"}
# CHECK-NEXT:         scf.yield %11 : tensor<1x12x12x3xf32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %9 into %arg4[%arg3, 0, 0, 0] [1, 12, 12, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> into tensor<1x12x12x3xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x12x12x3xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %3 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:     %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_2 = arith.constant 0 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %c1_4 = arith.constant 1 : index
# CHECK-NEXT:     %4 = scf.for %arg3 = %c0_2 to %c1_3 step %c1_4 iter_args(%arg4 = %3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_8 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_9 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0_8 to %c4 step %c1_9 iter_args(%arg6 = %extracted_slice) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_11 = arith.constant 0 : index
# CHECK-NEXT:         %c4_12 = arith.constant 4 : index
# CHECK-NEXT:         %c1_13 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg7 = %c0_11 to %c4_12 step %c1_13 iter_args(%arg8 = %extracted_slice_10) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_16 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_17 = arith.constant 1 : index
# CHECK-NEXT:           %8 = scf.for %arg9 = %c0_16 to %c16 step %c1_17 iter_args(%arg10 = %extracted_slice_15) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_19 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %9 = linalg.fill {__xtc_id_conv_0_} ins(%cst_1 : f32) outs(%extracted_slice_19 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_20 = tensor.insert_slice %9 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_20 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %8 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_18 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_14 = tensor.insert_slice %7 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_14 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %c0_5 = arith.constant 0 : index
# CHECK-NEXT:     %c1_6 = arith.constant 1 : index
# CHECK-NEXT:     %c1_7 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg3 = %c0_5 to %c1_6 step %c1_7 iter_args(%arg4 = %4) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %2[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:       %extracted_slice_8 = tensor.extract_slice %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:       %c0_10 = arith.constant 0 : index
# CHECK-NEXT:       %c4 = arith.constant 4 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0_10 to %c4 step %c1_11 iter_args(%arg6 = %extracted_slice_9) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:         %7 = affine.apply #map16(%arg5)
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, %7, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:         %extracted_slice_13 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:         %c0_15 = arith.constant 0 : index
# CHECK-NEXT:         %c4_16 = arith.constant 4 : index
# CHECK-NEXT:         %c1_17 = arith.constant 1 : index
# CHECK-NEXT:         %8 = scf.for %arg7 = %c0_15 to %c4_16 step %c1_17 iter_args(%arg8 = %extracted_slice_14) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:           %9 = affine.apply #map16(%arg7)
# CHECK-NEXT:           %extracted_slice_19 = tensor.extract_slice %extracted_slice_12[0, 0, %9, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:           %c0_22 = arith.constant 0 : index
# CHECK-NEXT:           %c16 = arith.constant 16 : index
# CHECK-NEXT:           %c1_23 = arith.constant 1 : index
# CHECK-NEXT:           %10 = scf.for %arg9 = %c0_22 to %c16 step %c1_23 iter_args(%arg10 = %extracted_slice_21) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:             %extracted_slice_25 = tensor.extract_slice %extracted_slice_19[0, 0, 0, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:             %extracted_slice_26 = tensor.extract_slice %extracted_slice_20[0, 0, 0, %arg9] [5, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x1xf32>
# CHECK-NEXT:             %extracted_slice_27 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %c0_28 = arith.constant 0 : index
# CHECK-NEXT:             %c5 = arith.constant 5 : index
# CHECK-NEXT:             %c1_29 = arith.constant 1 : index
# CHECK-NEXT:             %11 = scf.for %arg11 = %c0_28 to %c5 step %c1_29 iter_args(%arg12 = %extracted_slice_27) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:               %extracted_slice_31 = tensor.extract_slice %extracted_slice_25[0, %arg11, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x1x5x3xf32>
# CHECK-NEXT:               %extracted_slice_32 = tensor.extract_slice %extracted_slice_26[%arg11, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : tensor<5x5x3x1xf32> to tensor<1x5x3x1xf32>
# CHECK-NEXT:               %extracted_slice_33 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:               %c0_34 = arith.constant 0 : index
# CHECK-NEXT:               %c5_35 = arith.constant 5 : index
# CHECK-NEXT:               %c1_36 = arith.constant 1 : index
# CHECK-NEXT:               %12 = scf.for %arg13 = %c0_34 to %c5_35 step %c1_36 iter_args(%arg14 = %extracted_slice_33) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                 %extracted_slice_38 = tensor.extract_slice %extracted_slice_31[0, 0, %arg13, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x5x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_32[0, %arg13, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : tensor<1x5x3x1xf32> to tensor<1x1x3x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %c0_41 = arith.constant 0 : index
# CHECK-NEXT:                 %c3 = arith.constant 3 : index
# CHECK-NEXT:                 %c1_42 = arith.constant 1 : index
# CHECK-NEXT:                 %13 = scf.for %arg15 = %c0_41 to %c3 step %c1_42 iter_args(%arg16 = %extracted_slice_40) -> (tensor<1x1x1x1xf32>) {
# CHECK-NEXT:                   %extracted_slice_44 = tensor.extract_slice %extracted_slice_38[0, 0, 0, %arg15] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_45 = tensor.extract_slice %extracted_slice_39[0, 0, %arg15, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x3x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %extracted_slice_46 = tensor.extract_slice %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %14 = linalg.generic {indexing_maps = [#map17, #map18, #map19], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_45 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%extracted_slice_46 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                   ^bb0(%in: f32, %in_48: f32, %out: f32):
# CHECK-NEXT:                     %15 = arith.mulf %in, %in_48 : f32
# CHECK-NEXT:                     %16 = arith.addf %out, %15 : f32
# CHECK-NEXT:                     linalg.yield %16 : f32
# CHECK-NEXT:                   } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                   %inserted_slice_47 = tensor.insert_slice %14 into %arg16[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                   scf.yield %inserted_slice_47 : tensor<1x1x1x1xf32>
# CHECK-NEXT:                 } {"./c"}
# CHECK-NEXT:                 %inserted_slice_43 = tensor.insert_slice %13 into %arg14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_43 : tensor<1x1x1x1xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               %inserted_slice_37 = tensor.insert_slice %12 into %arg12[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x1xf32>
# CHECK-NEXT:               scf.yield %inserted_slice_37 : tensor<1x1x1x1xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_30 = tensor.insert_slice %11 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_30 : tensor<1x1x1x16xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_24 = tensor.insert_slice %10 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_24 : tensor<1x1x4x16xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %8 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x4x4x16xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (-d0 + 2, 0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> (0, d0 - 2)>
# CHECK-NEXT: #map2 = affine_map<(d0) -> (8, d0)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1) -> (-d0 + 8, -d1 + 1)>
# CHECK-NEXT: #map4 = affine_map<(d0) -> (0, d0)>
# CHECK-NEXT: #map5 = affine_map<(d0) -> (-d0, 0)>
# CHECK-NEXT: #map6 = affine_map<(d0, d1) -> (d1, d0)>
# CHECK-NEXT: #map7 = affine_map<(d0, d1, d2) -> (-d2 + 1, d0 - d1)>
# CHECK-NEXT: #map8 = affine_map<(d0, d1) -> (-d0 - d1 + 1)>
# CHECK-NEXT: #map9 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map11 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map12 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c5 = arith.constant 5 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c12 = arith.constant 12 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x1x12x3xf32>
# CHECK-NEXT:     %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<1x1x12x3xf32>
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%arg4 = %alloc) -> (memref<1x12x12x3xf32>) {
# CHECK-NEXT:       %3 = affine.max #map(%arg3)
# CHECK-NEXT:       %4 = affine.max #map1(%arg3)
# CHECK-NEXT:       %5 = affine.min #map2(%4)
# CHECK-NEXT:       %6 = affine.min #map3(%5, %3)
# CHECK-NEXT:       %7 = affine.max #map4(%6)
# CHECK-NEXT:       %8 = arith.cmpi eq, %7, %c0 : index
# CHECK-NEXT:       %9 = scf.if %8 -> (memref<1x1x12x3xf32>) {
# CHECK-NEXT:         linalg.map outs(%alloca : memref<1x1x12x3xf32>)
# CHECK-NEXT:           () {
# CHECK-NEXT:             %10 = linalg.index 0 : index
# CHECK-NEXT:             %11 = linalg.index 1 : index
# CHECK-NEXT:             %12 = linalg.index 2 : index
# CHECK-NEXT:             %13 = linalg.index 3 : index
# CHECK-NEXT:             linalg.yield %cst : f32
# CHECK-NEXT:           }
# CHECK-NEXT:         scf.yield %alloca : memref<1x1x12x3xf32>
# CHECK-NEXT:       } else {
# CHECK-NEXT:         %subview_2 = memref.subview %arg0[0, %5, 0, 0] [1, %7, 8, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32> to memref<1x?x8x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_3, %alloca_0 : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x12x3xf32>
# CHECK-NEXT:         %alloca_4 = memref.alloca() {alignment = 256 : i64} : memref<1x1x1x3xf32>
# CHECK-NEXT:         %alloca_5 = memref.alloca() {alignment = 256 : i64} : memref<1x1x1x3xf32>
# CHECK-NEXT:         %10 = scf.for %arg5 = %c0 to %c12 step %c1 iter_args(%arg6 = %alloca_0) -> (memref<1x1x12x3xf32>) {
# CHECK-NEXT:           %11 = affine.max #map5(%3)
# CHECK-NEXT:           %12 = affine.min #map6(%11, %7)
# CHECK-NEXT:           %13 = affine.min #map7(%7, %12, %3)
# CHECK-NEXT:           %14 = affine.max #map4(%13)
# CHECK-NEXT:           %15 = arith.cmpi eq, %14, %c0 : index
# CHECK-NEXT:           %16 = affine.max #map(%arg5)
# CHECK-NEXT:           %17 = affine.max #map1(%arg5)
# CHECK-NEXT:           %18 = affine.min #map2(%17)
# CHECK-NEXT:           %19 = affine.min #map3(%18, %16)
# CHECK-NEXT:           %20 = affine.max #map4(%19)
# CHECK-NEXT:           %21 = arith.cmpi eq, %20, %c0 : index
# CHECK-NEXT:           %22 = arith.ori %21, %15 : i1
# CHECK-NEXT:           %23 = scf.if %22 -> (memref<1x1x1x3xf32>) {
# CHECK-NEXT:             linalg.map outs(%alloca_4 : memref<1x1x1x3xf32>)
# CHECK-NEXT:               () {
# CHECK-NEXT:                 %24 = linalg.index 0 : index
# CHECK-NEXT:                 %25 = linalg.index 1 : index
# CHECK-NEXT:                 %26 = linalg.index 2 : index
# CHECK-NEXT:                 %27 = linalg.index 3 : index
# CHECK-NEXT:                 linalg.yield %cst : f32
# CHECK-NEXT:               }
# CHECK-NEXT:             scf.yield %alloca_4 : memref<1x1x1x3xf32>
# CHECK-NEXT:           } else {
# CHECK-NEXT:             %subview_7 = memref.subview %subview_2[0, %12, %18, 0] [1, %14, %20, 3] [1, 1, 1, 1] : memref<1x?x8x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x?x?x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:             %subview_8 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x12x3xf32> to memref<1x1x1x3xf32, strided<[36, 36, 3, 1], offset: ?>>
# CHECK-NEXT:             memref.copy %subview_8, %alloca_5 : memref<1x1x1x3xf32, strided<[36, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32>
# CHECK-NEXT:             %alloca_9 = memref.alloca() {alignment = 256 : i64} : memref<1x1x1x1xf32>
# CHECK-NEXT:             %alloca_10 = memref.alloca() {alignment = 256 : i64} : memref<1x1x1x1xf32>
# CHECK-NEXT:             %24 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %alloca_5) -> (memref<1x1x1x3xf32>) {
# CHECK-NEXT:               %25 = affine.min #map6(%11, %14)
# CHECK-NEXT:               %26 = affine.min #map7(%14, %25, %3)
# CHECK-NEXT:               %27 = affine.max #map4(%26)
# CHECK-NEXT:               %28 = arith.cmpi eq, %27, %c0 : index
# CHECK-NEXT:               %29 = affine.apply #map8(%3, %27)
# CHECK-NEXT:               %30 = affine.max #map5(%16)
# CHECK-NEXT:               %31 = affine.min #map6(%30, %20)
# CHECK-NEXT:               %32 = affine.min #map7(%20, %31, %16)
# CHECK-NEXT:               %33 = affine.max #map4(%32)
# CHECK-NEXT:               %34 = arith.cmpi eq, %33, %c0 : index
# CHECK-NEXT:               %35 = arith.ori %34, %28 : i1
# CHECK-NEXT:               %36 = affine.apply #map8(%16, %33)
# CHECK-NEXT:               %37 = scf.if %35 -> (memref<1x1x1x1xf32>) {
# CHECK-NEXT:                 linalg.map outs(%alloca_9 : memref<1x1x1x1xf32>)
# CHECK-NEXT:                   () {
# CHECK-NEXT:                     %38 = linalg.index 0 : index
# CHECK-NEXT:                     %39 = linalg.index 1 : index
# CHECK-NEXT:                     %40 = linalg.index 2 : index
# CHECK-NEXT:                     %41 = linalg.index 3 : index
# CHECK-NEXT:                     linalg.yield %cst : f32
# CHECK-NEXT:                   }
# CHECK-NEXT:                 scf.yield %alloca_9 : memref<1x1x1x1xf32>
# CHECK-NEXT:               } else {
# CHECK-NEXT:                 %subview_12 = memref.subview %subview_7[0, %25, %31, %arg7] [1, %27, %33, 1] [1, 1, 1, 1] : memref<1x?x?x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x?x?x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                 linalg.map outs(%alloca_10 : memref<1x1x1x1xf32>)
# CHECK-NEXT:                   () {
# CHECK-NEXT:                     %38 = linalg.index 0 : index
# CHECK-NEXT:                     %39 = linalg.index 1 : index
# CHECK-NEXT:                     %40 = linalg.index 2 : index
# CHECK-NEXT:                     %41 = linalg.index 3 : index
# CHECK-NEXT:                     linalg.yield %cst : f32
# CHECK-NEXT:                   }
# CHECK-NEXT:                 %c1_13 = arith.constant 1 : index
# CHECK-NEXT:                 %dim = memref.dim %subview_12, %c1_13 : memref<1x?x?x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                 %c2 = arith.constant 2 : index
# CHECK-NEXT:                 %dim_14 = memref.dim %subview_12, %c2 : memref<1x?x?x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_15 = memref.subview %alloca_10[0, %3, %16, 0] [1, %dim, %dim_14, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32> to memref<1x?x?x1xf32, strided<[1, 1, 1, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_12, %subview_15 : memref<1x?x?x1xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x?x?x1xf32, strided<[1, 1, 1, 1], offset: ?>>
# CHECK-NEXT:                 scf.yield %alloca_10 : memref<1x1x1x1xf32>
# CHECK-NEXT:               }
# CHECK-NEXT:               %subview_11 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32> to memref<1x1x1x1xf32, strided<[3, 3, 3, 1], offset: ?>>
# CHECK-NEXT:               memref.copy %37, %subview_11 : memref<1x1x1x1xf32> to memref<1x1x1x1xf32, strided<[3, 3, 3, 1], offset: ?>>
# CHECK-NEXT:               scf.yield %arg8 : memref<1x1x1x3xf32>
# CHECK-NEXT:             } {"./c"}
# CHECK-NEXT:             scf.yield %24 : memref<1x1x1x3xf32>
# CHECK-NEXT:           }
# CHECK-NEXT:           %subview_6 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x12x3xf32> to memref<1x1x1x3xf32, strided<[36, 36, 3, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %23, %subview_6 : memref<1x1x1x3xf32> to memref<1x1x1x3xf32, strided<[36, 36, 3, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg6 : memref<1x1x12x3xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         scf.yield %10 : memref<1x1x12x3xf32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %9, %subview_1 : memref<1x1x12x3xf32> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x12x12x3xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_1) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_3) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_5 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%subview_5 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>)
# CHECK-NEXT:           %subview_6 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_5, %subview_6 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %4, %subview_4 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %3, %subview_2 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:     } {"./h"}
# CHECK-NEXT:     %subview = memref.subview %0[0, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>>
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:       %3 = affine.apply #map9(%arg3)
# CHECK-NEXT:       %subview_1 = memref.subview %subview[0, %3, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>> to memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:       %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_2) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:         %5 = affine.apply #map9(%arg5)
# CHECK-NEXT:         %subview_4 = memref.subview %subview_1[0, 0, %5, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_5 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         %6 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_5) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_7 = memref.subview %arg1[0, 0, 0, %arg7] [5, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:           %subview_8 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0 to %c5 step %c1 iter_args(%arg10 = %subview_8) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:             %subview_10 = memref.subview %subview_4[0, %arg9, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:             %subview_11 = memref.subview %subview_7[%arg9, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:             %8 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:               %subview_12 = memref.subview %subview_10[0, 0, %arg11, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:               %subview_13 = memref.subview %subview_11[0, %arg11, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:               %9 = scf.for %arg13 = %c0 to %c3 step %c1 iter_args(%arg14 = %arg12) -> (memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                 %subview_14 = memref.subview %subview_12[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_15 = memref.subview %subview_13[0, 0, %arg13, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_14, %subview_15 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%arg14 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_16: f32, %out: f32):
# CHECK-NEXT:                   %10 = arith.mulf %in, %in_16 : f32
# CHECK-NEXT:                   %11 = arith.addf %out, %10 : f32
# CHECK-NEXT:                   linalg.yield %11 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 scf.yield %arg14 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:               } {"./c"}
# CHECK-NEXT:               scf.yield %9 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:             } {"./s"}
# CHECK-NEXT:             scf.yield %8 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           } {"./r"}
# CHECK-NEXT:           %subview_9 = memref.subview %arg8[0, 0, 0, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %7, %subview_9 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         } {"./f"}
# CHECK-NEXT:         %subview_6 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %6, %subview_6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       } {"./w"}
# CHECK-NEXT:       %subview_3 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_3 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
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
