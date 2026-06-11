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
# fuse above split
sch.split("r", {"r0": 0, "r1": 2},root="c")
sch.interchange(["b", "h", "w", "r0","r1"],root="c")
sch.interchange(["s", "c", "f"],"c/r0")
sch.interchange(["s", "c", "f"],"c/r1")
sch.fuse_producer_at("w", 0,root="c")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="fuse_then_split_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x12x12x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %5 = linalg.index 0 : index
# CHECK-NEXT:        %6 = linalg.index 1 : index
# CHECK-NEXT:        %7 = linalg.index 2 : index
# CHECK-NEXT:        %8 = linalg.index 3 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_1 = arith.constant 0 : index
# CHECK-NEXT:        %9 = arith.subi %5, %c0_1 : index
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %10 = arith.cmpi sge, %9, %c0 : index
# CHECK-NEXT:        %11 = arith.cmpi slt, %9, %c1 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %12 = arith.subi %6, %c2 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %13 = arith.cmpi sge, %12, %c0 : index
# CHECK-NEXT:        %14 = arith.cmpi slt, %12, %c8 : index
# CHECK-NEXT:        %c2_2 = arith.constant 2 : index
# CHECK-NEXT:        %15 = arith.subi %7, %c2_2 : index
# CHECK-NEXT:        %c8_3 = arith.constant 8 : index
# CHECK-NEXT:        %16 = arith.cmpi sge, %15, %c0 : index
# CHECK-NEXT:        %17 = arith.cmpi slt, %15, %c8_3 : index
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %18 = arith.subi %8, %c0_4 : index
# CHECK-NEXT:        %c3 = arith.constant 3 : index
# CHECK-NEXT:        %19 = arith.cmpi sge, %18, %c0 : index
# CHECK-NEXT:        %20 = arith.cmpi slt, %18, %c3 : index
# CHECK-NEXT:        %21 = arith.andi %10, %11 : i1
# CHECK-NEXT:        %22 = arith.andi %21, %13 : i1
# CHECK-NEXT:        %23 = arith.andi %22, %14 : i1
# CHECK-NEXT:        %24 = arith.andi %23, %16 : i1
# CHECK-NEXT:        %25 = arith.andi %24, %17 : i1
# CHECK-NEXT:        %26 = arith.andi %25, %19 : i1
# CHECK-NEXT:        %27 = arith.andi %26, %20 : i1
# CHECK-NEXT:        %28 = scf.if %27 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%9, %12, %15, %18] : tensor<1x8x8x3xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %28 : f32
# CHECK-NEXT:      } -> tensor<1x12x12x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%1, %arg1 : tensor<1x12x12x3xf32>, tensor<5x5x3x16xf32>) outs(%3 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.mulf %in, %in_1 fastmath<fast> : f32
# CHECK-NEXT:        %6 = arith.addf %out, %5 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %6 : f32
# CHECK-NEXT:      } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "c/b" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %2 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "c/h" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %3 into %loops_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "c/w" : !transform.any_op
# CHECK-NEXT:      %5 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %5 into %loops_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %6 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %7 = transform.structured.split %6 after 2  {dimension = 4 : i64} : !transform.any_op
# CHECK-NEXT:      %8:2 = transform.split_handle %7 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %8#0 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "c/r0/s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "c/r0/c" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "c/r0/f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %8#1 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "c/r1/s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "c/r1/c" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "c/r1/f" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map1 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT:  #map2 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x12x12x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %5 = linalg.index 0 : index
# CHECK-NEXT:        %6 = linalg.index 1 : index
# CHECK-NEXT:        %7 = linalg.index 2 : index
# CHECK-NEXT:        %8 = linalg.index 3 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %9 = arith.subi %5, %c0_3 : index
# CHECK-NEXT:        %c1_4 = arith.constant 1 : index
# CHECK-NEXT:        %10 = arith.cmpi sge, %9, %c0_2 : index
# CHECK-NEXT:        %11 = arith.cmpi slt, %9, %c1_4 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %12 = arith.subi %6, %c2 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %13 = arith.cmpi sge, %12, %c0_2 : index
# CHECK-NEXT:        %14 = arith.cmpi slt, %12, %c8 : index
# CHECK-NEXT:        %c2_5 = arith.constant 2 : index
# CHECK-NEXT:        %15 = arith.subi %7, %c2_5 : index
# CHECK-NEXT:        %c8_6 = arith.constant 8 : index
# CHECK-NEXT:        %16 = arith.cmpi sge, %15, %c0_2 : index
# CHECK-NEXT:        %17 = arith.cmpi slt, %15, %c8_6 : index
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %18 = arith.subi %8, %c0_7 : index
# CHECK-NEXT:        %c3 = arith.constant 3 : index
# CHECK-NEXT:        %19 = arith.cmpi sge, %18, %c0_2 : index
# CHECK-NEXT:        %20 = arith.cmpi slt, %18, %c3 : index
# CHECK-NEXT:        %21 = arith.andi %10, %11 : i1
# CHECK-NEXT:        %22 = arith.andi %21, %13 : i1
# CHECK-NEXT:        %23 = arith.andi %22, %14 : i1
# CHECK-NEXT:        %24 = arith.andi %23, %16 : i1
# CHECK-NEXT:        %25 = arith.andi %24, %17 : i1
# CHECK-NEXT:        %26 = arith.andi %25, %19 : i1
# CHECK-NEXT:        %27 = arith.andi %26, %20 : i1
# CHECK-NEXT:        %28 = scf.if %27 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%9, %12, %15, %18] : tensor<1x8x8x3xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %28 : f32
# CHECK-NEXT:      } -> tensor<1x12x12x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %4 = scf.for %arg3 = %c0 to %c1 step %c1_1 iter_args(%arg4 = %3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %0[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:        %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<1x11x11x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:        ^bb0(%out: f32):
# CHECK-NEXT:          %7 = linalg.index 0 : index
# CHECK-NEXT:          %8 = affine.apply #map1(%arg3)[%7]
# CHECK-NEXT:          %9 = linalg.index 1 : index
# CHECK-NEXT:          %10 = linalg.index 2 : index
# CHECK-NEXT:          %11 = linalg.index 3 : index
# CHECK-NEXT:          %c0_6 = arith.constant 0 : index
# CHECK-NEXT:          %c0_7 = arith.constant 0 : index
# CHECK-NEXT:          %12 = arith.subi %8, %c0_7 : index
# CHECK-NEXT:          %c1_8 = arith.constant 1 : index
# CHECK-NEXT:          %13 = arith.cmpi sge, %12, %c0_6 : index
# CHECK-NEXT:          %14 = arith.cmpi slt, %12, %c1_8 : index
# CHECK-NEXT:          %c2 = arith.constant 2 : index
# CHECK-NEXT:          %15 = arith.subi %9, %c2 : index
# CHECK-NEXT:          %c8 = arith.constant 8 : index
# CHECK-NEXT:          %16 = arith.cmpi sge, %15, %c0_6 : index
# CHECK-NEXT:          %17 = arith.cmpi slt, %15, %c8 : index
# CHECK-NEXT:          %c2_9 = arith.constant 2 : index
# CHECK-NEXT:          %18 = arith.subi %10, %c2_9 : index
# CHECK-NEXT:          %c8_10 = arith.constant 8 : index
# CHECK-NEXT:          %19 = arith.cmpi sge, %18, %c0_6 : index
# CHECK-NEXT:          %20 = arith.cmpi slt, %18, %c8_10 : index
# CHECK-NEXT:          %c0_11 = arith.constant 0 : index
# CHECK-NEXT:          %21 = arith.subi %11, %c0_11 : index
# CHECK-NEXT:          %c3 = arith.constant 3 : index
# CHECK-NEXT:          %22 = arith.cmpi sge, %21, %c0_6 : index
# CHECK-NEXT:          %23 = arith.cmpi slt, %21, %c3 : index
# CHECK-NEXT:          %24 = arith.andi %13, %14 : i1
# CHECK-NEXT:          %25 = arith.andi %24, %16 : i1
# CHECK-NEXT:          %26 = arith.andi %25, %17 : i1
# CHECK-NEXT:          %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:          %28 = arith.andi %27, %20 : i1
# CHECK-NEXT:          %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:          %30 = arith.andi %29, %23 : i1
# CHECK-NEXT:          %31 = scf.if %30 -> (f32) {
# CHECK-NEXT:            %extracted = tensor.extract %arg0[%12, %15, %18, %21] : tensor<1x8x8x3xf32>
# CHECK-NEXT:            scf.yield %extracted : f32
# CHECK-NEXT:          } else {
# CHECK-NEXT:            scf.yield %cst : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          linalg.yield %31 : f32
# CHECK-NEXT:        } -> tensor<1x11x11x3xf32>
# CHECK-NEXT:        %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c4 = arith.constant 4 : index
# CHECK-NEXT:        %c1_5 = arith.constant 1 : index
# CHECK-NEXT:        %6 = scf.for %arg5 = %c0_4 to %c4 step %c1_5 iter_args(%arg6 = %extracted_slice_3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:          %7 = affine.apply #map2(%arg5)
# CHECK-NEXT:          %8 = affine.apply #map2(%arg5)
# CHECK-NEXT:          %extracted_slice_6 = tensor.extract_slice %extracted_slice[0, %8, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:          %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_6 : tensor<1x5x11x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %11 = linalg.index 0 : index
# CHECK-NEXT:            %12 = affine.apply #map1(%arg3)[%11]
# CHECK-NEXT:            %13 = linalg.index 1 : index
# CHECK-NEXT:            %14 = affine.apply #map3(%arg5)[%13]
# CHECK-NEXT:            %15 = linalg.index 2 : index
# CHECK-NEXT:            %16 = linalg.index 3 : index
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %17 = arith.subi %12, %c0_14 : index
# CHECK-NEXT:            %c1_15 = arith.constant 1 : index
# CHECK-NEXT:            %18 = arith.cmpi sge, %17, %c0_13 : index
# CHECK-NEXT:            %19 = arith.cmpi slt, %17, %c1_15 : index
# CHECK-NEXT:            %c2 = arith.constant 2 : index
# CHECK-NEXT:            %20 = arith.subi %14, %c2 : index
# CHECK-NEXT:            %c8 = arith.constant 8 : index
# CHECK-NEXT:            %21 = arith.cmpi sge, %20, %c0_13 : index
# CHECK-NEXT:            %22 = arith.cmpi slt, %20, %c8 : index
# CHECK-NEXT:            %c2_16 = arith.constant 2 : index
# CHECK-NEXT:            %23 = arith.subi %15, %c2_16 : index
# CHECK-NEXT:            %c8_17 = arith.constant 8 : index
# CHECK-NEXT:            %24 = arith.cmpi sge, %23, %c0_13 : index
# CHECK-NEXT:            %25 = arith.cmpi slt, %23, %c8_17 : index
# CHECK-NEXT:            %c0_18 = arith.constant 0 : index
# CHECK-NEXT:            %26 = arith.subi %16, %c0_18 : index
# CHECK-NEXT:            %c3 = arith.constant 3 : index
# CHECK-NEXT:            %27 = arith.cmpi sge, %26, %c0_13 : index
# CHECK-NEXT:            %28 = arith.cmpi slt, %26, %c3 : index
# CHECK-NEXT:            %29 = arith.andi %18, %19 : i1
# CHECK-NEXT:            %30 = arith.andi %29, %21 : i1
# CHECK-NEXT:            %31 = arith.andi %30, %22 : i1
# CHECK-NEXT:            %32 = arith.andi %31, %24 : i1
# CHECK-NEXT:            %33 = arith.andi %32, %25 : i1
# CHECK-NEXT:            %34 = arith.andi %33, %27 : i1
# CHECK-NEXT:            %35 = arith.andi %34, %28 : i1
# CHECK-NEXT:            %36 = scf.if %35 -> (f32) {
# CHECK-NEXT:              %extracted = tensor.extract %arg0[%17, %20, %23, %26] : tensor<1x8x8x3xf32>
# CHECK-NEXT:              scf.yield %extracted : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %36 : f32
# CHECK-NEXT:          } -> tensor<1x5x11x3xf32>
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:          %c0_9 = arith.constant 0 : index
# CHECK-NEXT:          %c4_10 = arith.constant 4 : index
# CHECK-NEXT:          %c1_11 = arith.constant 1 : index
# CHECK-NEXT:          %10 = scf.for %arg7 = %c0_9 to %c4_10 step %c1_11 iter_args(%arg8 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:            %11 = affine.apply #map2(%arg7)
# CHECK-NEXT:            %12 = affine.apply #map2(%arg7)
# CHECK-NEXT:            %extracted_slice_13 = tensor.extract_slice %extracted_slice_6[0, 0, %12, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:            %13 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_13 : tensor<1x5x5x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:            ^bb0(%out: f32):
# CHECK-NEXT:              %16 = linalg.index 0 : index
# CHECK-NEXT:              %17 = affine.apply #map1(%arg3)[%16]
# CHECK-NEXT:              %18 = linalg.index 1 : index
# CHECK-NEXT:              %19 = affine.apply #map3(%arg5)[%18]
# CHECK-NEXT:              %20 = linalg.index 2 : index
# CHECK-NEXT:              %21 = affine.apply #map3(%arg7)[%20]
# CHECK-NEXT:              %22 = linalg.index 3 : index
# CHECK-NEXT:              %c0_30 = arith.constant 0 : index
# CHECK-NEXT:              %c0_31 = arith.constant 0 : index
# CHECK-NEXT:              %23 = arith.subi %17, %c0_31 : index
# CHECK-NEXT:              %c1_32 = arith.constant 1 : index
# CHECK-NEXT:              %24 = arith.cmpi sge, %23, %c0_30 : index
# CHECK-NEXT:              %25 = arith.cmpi slt, %23, %c1_32 : index
# CHECK-NEXT:              %c2 = arith.constant 2 : index
# CHECK-NEXT:              %26 = arith.subi %19, %c2 : index
# CHECK-NEXT:              %c8 = arith.constant 8 : index
# CHECK-NEXT:              %27 = arith.cmpi sge, %26, %c0_30 : index
# CHECK-NEXT:              %28 = arith.cmpi slt, %26, %c8 : index
# CHECK-NEXT:              %c2_33 = arith.constant 2 : index
# CHECK-NEXT:              %29 = arith.subi %21, %c2_33 : index
# CHECK-NEXT:              %c8_34 = arith.constant 8 : index
# CHECK-NEXT:              %30 = arith.cmpi sge, %29, %c0_30 : index
# CHECK-NEXT:              %31 = arith.cmpi slt, %29, %c8_34 : index
# CHECK-NEXT:              %c0_35 = arith.constant 0 : index
# CHECK-NEXT:              %32 = arith.subi %22, %c0_35 : index
# CHECK-NEXT:              %c3 = arith.constant 3 : index
# CHECK-NEXT:              %33 = arith.cmpi sge, %32, %c0_30 : index
# CHECK-NEXT:              %34 = arith.cmpi slt, %32, %c3 : index
# CHECK-NEXT:              %35 = arith.andi %24, %25 : i1
# CHECK-NEXT:              %36 = arith.andi %35, %27 : i1
# CHECK-NEXT:              %37 = arith.andi %36, %28 : i1
# CHECK-NEXT:              %38 = arith.andi %37, %30 : i1
# CHECK-NEXT:              %39 = arith.andi %38, %31 : i1
# CHECK-NEXT:              %40 = arith.andi %39, %33 : i1
# CHECK-NEXT:              %41 = arith.andi %40, %34 : i1
# CHECK-NEXT:              %42 = scf.if %41 -> (f32) {
# CHECK-NEXT:                %extracted = tensor.extract %arg0[%23, %26, %29, %32] : tensor<1x8x8x3xf32>
# CHECK-NEXT:                scf.yield %extracted : f32
# CHECK-NEXT:              } else {
# CHECK-NEXT:                scf.yield %cst : f32
# CHECK-NEXT:              }
# CHECK-NEXT:              linalg.yield %42 : f32
# CHECK-NEXT:            } -> tensor<1x5x5x3xf32>
# CHECK-NEXT:            %extracted_slice_14 = tensor.extract_slice %extracted_slice_7[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<5x5x3x16xf32>
# CHECK-NEXT:            %extracted_slice_15 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %extracted_slice_16 = tensor.extract_slice %13[0, 0, 0, 0] [1, 2, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x2x5x3xf32>
# CHECK-NEXT:            %extracted_slice_17 = tensor.extract_slice %extracted_slice_14[0, 0, 0, 0] [2, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<2x5x3x16xf32>
# CHECK-NEXT:            %extracted_slice_18 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_19 = arith.constant 0 : index
# CHECK-NEXT:            %c5 = arith.constant 5 : index
# CHECK-NEXT:            %c1_20 = arith.constant 1 : index
# CHECK-NEXT:            %14 = scf.for %arg9 = %c0_19 to %c5 step %c1_20 iter_args(%arg10 = %extracted_slice_18) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_30 = tensor.extract_slice %extracted_slice_16[0, 0, %arg9, 0] [1, 2, 1, 3] [1, 1, 1, 1] : tensor<1x2x5x3xf32> to tensor<1x2x1x3xf32>
# CHECK-NEXT:              %extracted_slice_31 = tensor.extract_slice %extracted_slice_17[0, %arg9, 0, 0] [2, 1, 3, 16] [1, 1, 1, 1] : tensor<2x5x3x16xf32> to tensor<2x1x3x16xf32>
# CHECK-NEXT:              %extracted_slice_32 = tensor.extract_slice %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:              %c0_33 = arith.constant 0 : index
# CHECK-NEXT:              %c3 = arith.constant 3 : index
# CHECK-NEXT:              %c1_34 = arith.constant 1 : index
# CHECK-NEXT:              %16 = scf.for %arg11 = %c0_33 to %c3 step %c1_34 iter_args(%arg12 = %extracted_slice_32) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                %extracted_slice_36 = tensor.extract_slice %extracted_slice_30[0, 0, 0, %arg11] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x2x1x3xf32> to tensor<1x2x1x1xf32>
# CHECK-NEXT:                %extracted_slice_37 = tensor.extract_slice %extracted_slice_31[0, 0, %arg11, 0] [2, 1, 1, 16] [1, 1, 1, 1] : tensor<2x1x3x16xf32> to tensor<2x1x1x16xf32>
# CHECK-NEXT:                %extracted_slice_38 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                %c0_39 = arith.constant 0 : index
# CHECK-NEXT:                %c16 = arith.constant 16 : index
# CHECK-NEXT:                %c1_40 = arith.constant 1 : index
# CHECK-NEXT:                %17 = scf.for %arg13 = %c0_39 to %c16 step %c1_40 iter_args(%arg14 = %extracted_slice_38) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                  %extracted_slice_42 = tensor.extract_slice %extracted_slice_36[0, 0, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x2x1x1xf32> to tensor<1x2x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_43 = tensor.extract_slice %extracted_slice_37[0, 0, 0, %arg13] [2, 1, 1, 1] [1, 1, 1, 1] : tensor<2x1x1x16xf32> to tensor<2x1x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_44 = tensor.extract_slice %arg14[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %18 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_42, %extracted_slice_43 : tensor<1x2x1x1xf32>, tensor<2x1x1x1xf32>) outs(%extracted_slice_44 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                    %19 = arith.mulf %in, %in_46 fastmath<fast> : f32
# CHECK-NEXT:                    %20 = arith.addf %out, %19 fastmath<fast> : f32
# CHECK-NEXT:                    linalg.yield %20 : f32
# CHECK-NEXT:                  } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %inserted_slice_45 = tensor.insert_slice %18 into %arg14[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_45 : tensor<1x1x1x16xf32>
# CHECK-NEXT:                } {"c/r0/f"}
# CHECK-NEXT:                %inserted_slice_41 = tensor.insert_slice %17 into %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_41 : tensor<1x1x1x16xf32>
# CHECK-NEXT:              } {"c/r0/c"}
# CHECK-NEXT:              %inserted_slice_35 = tensor.insert_slice %16 into %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_35 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"c/r0/s"}
# CHECK-NEXT:            %inserted_slice_21 = tensor.insert_slice %14 into %extracted_slice_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:            %extracted_slice_22 = tensor.extract_slice %13[0, 2, 0, 0] [1, 3, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x3x5x3xf32>
# CHECK-NEXT:            %extracted_slice_23 = tensor.extract_slice %extracted_slice_14[2, 0, 0, 0] [3, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<3x5x3x16xf32>
# CHECK-NEXT:            %extracted_slice_24 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %c0_25 = arith.constant 0 : index
# CHECK-NEXT:            %c5_26 = arith.constant 5 : index
# CHECK-NEXT:            %c1_27 = arith.constant 1 : index
# CHECK-NEXT:            %15 = scf.for %arg9 = %c0_25 to %c5_26 step %c1_27 iter_args(%arg10 = %extracted_slice_24) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_30 = tensor.extract_slice %extracted_slice_22[0, 0, %arg9, 0] [1, 3, 1, 3] [1, 1, 1, 1] : tensor<1x3x5x3xf32> to tensor<1x3x1x3xf32>
# CHECK-NEXT:              %extracted_slice_31 = tensor.extract_slice %extracted_slice_23[0, %arg9, 0, 0] [3, 1, 3, 16] [1, 1, 1, 1] : tensor<3x5x3x16xf32> to tensor<3x1x3x16xf32>
# CHECK-NEXT:              %extracted_slice_32 = tensor.extract_slice %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:              %c0_33 = arith.constant 0 : index
# CHECK-NEXT:              %c3 = arith.constant 3 : index
# CHECK-NEXT:              %c1_34 = arith.constant 1 : index
# CHECK-NEXT:              %16 = scf.for %arg11 = %c0_33 to %c3 step %c1_34 iter_args(%arg12 = %extracted_slice_32) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                %extracted_slice_36 = tensor.extract_slice %extracted_slice_30[0, 0, 0, %arg11] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<1x3x1x3xf32> to tensor<1x3x1x1xf32>
# CHECK-NEXT:                %extracted_slice_37 = tensor.extract_slice %extracted_slice_31[0, 0, %arg11, 0] [3, 1, 1, 16] [1, 1, 1, 1] : tensor<3x1x3x16xf32> to tensor<3x1x1x16xf32>
# CHECK-NEXT:                %extracted_slice_38 = tensor.extract_slice %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                %c0_39 = arith.constant 0 : index
# CHECK-NEXT:                %c16 = arith.constant 16 : index
# CHECK-NEXT:                %c1_40 = arith.constant 1 : index
# CHECK-NEXT:                %17 = scf.for %arg13 = %c0_39 to %c16 step %c1_40 iter_args(%arg14 = %extracted_slice_38) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                  %extracted_slice_42 = tensor.extract_slice %extracted_slice_36[0, 0, 0, 0] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<1x3x1x1xf32> to tensor<1x3x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_43 = tensor.extract_slice %extracted_slice_37[0, 0, 0, %arg13] [3, 1, 1, 1] [1, 1, 1, 1] : tensor<3x1x1x16xf32> to tensor<3x1x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_44 = tensor.extract_slice %arg14[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %18 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_42, %extracted_slice_43 : tensor<1x3x1x1xf32>, tensor<3x1x1x1xf32>) outs(%extracted_slice_44 : tensor<1x1x1x1xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                    %19 = arith.mulf %in, %in_46 fastmath<fast> : f32
# CHECK-NEXT:                    %20 = arith.addf %out, %19 fastmath<fast> : f32
# CHECK-NEXT:                    linalg.yield %20 : f32
# CHECK-NEXT:                  } -> tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %inserted_slice_45 = tensor.insert_slice %18 into %arg14[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_45 : tensor<1x1x1x16xf32>
# CHECK-NEXT:                } {"c/r1/f"}
# CHECK-NEXT:                %inserted_slice_41 = tensor.insert_slice %17 into %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_41 : tensor<1x1x1x16xf32>
# CHECK-NEXT:              } {"c/r1/c"}
# CHECK-NEXT:              %inserted_slice_35 = tensor.insert_slice %16 into %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_35 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"c/r1/s"}
# CHECK-NEXT:            %inserted_slice_28 = tensor.insert_slice %15 into %inserted_slice_21[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:            %inserted_slice_29 = tensor.insert_slice %inserted_slice_28 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_29 : tensor<1x1x4x16xf32>
# CHECK-NEXT:          } {"c/w"}
# CHECK-NEXT:          %inserted_slice_12 = tensor.insert_slice %10 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_12 : tensor<1x4x4x16xf32>
# CHECK-NEXT:        } {"c/h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:      } {"c/b"}
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
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
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x4x4x16xf32>)
# CHECK-NEXT:      %subview = memref.subview %alloc[0, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:        %1 = affine.apply #map(%arg3)
# CHECK-NEXT:        %subview_0 = memref.subview %subview[0, %1, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>> to memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_1) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %3 = affine.apply #map(%arg5)
# CHECK-NEXT:          %subview_3 = memref.subview %subview_0[0, 0, %3, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_3 : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %6 = linalg.index 1 : index
# CHECK-NEXT:            %7 = affine.apply #map2(%arg3)[%6]
# CHECK-NEXT:            %8 = linalg.index 2 : index
# CHECK-NEXT:            %9 = affine.apply #map2(%arg5)[%8]
# CHECK-NEXT:            %10 = linalg.index 3 : index
# CHECK-NEXT:            %11 = arith.subi %7, %c2 : index
# CHECK-NEXT:            %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:            %13 = arith.cmpi slt, %11, %c8 : index
# CHECK-NEXT:            %14 = arith.subi %9, %c2 : index
# CHECK-NEXT:            %15 = arith.cmpi sge, %14, %c0 : index
# CHECK-NEXT:            %16 = arith.cmpi slt, %14, %c8 : index
# CHECK-NEXT:            %17 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:            %18 = arith.cmpi slt, %10, %c3 : index
# CHECK-NEXT:            %19 = arith.andi %12, %13 : i1
# CHECK-NEXT:            %20 = arith.andi %19, %15 : i1
# CHECK-NEXT:            %21 = arith.andi %20, %16 : i1
# CHECK-NEXT:            %22 = arith.andi %21, %17 : i1
# CHECK-NEXT:            %23 = arith.andi %22, %18 : i1
# CHECK-NEXT:            %24 = scf.if %23 -> (f32) {
# CHECK-NEXT:              %25 = memref.load %arg0[%c0, %11, %14, %10] : memref<1x8x8x3xf32>
# CHECK-NEXT:              scf.yield %25 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %24 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview_3[0, 0, 0, 0] [1, 2, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %arg1[0, 0, 0, 0] [2, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<2x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:          %4 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %subview_4) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_10 = memref.subview %subview_5[0, 0, %arg7, 0] [1, 2, 1, 3] [1, 1, 1, 1] : memref<1x2x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_6[0, %arg7, 0, 0] [2, 1, 3, 16] [1, 1, 1, 1] : memref<2x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<2x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %6 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[0, 0, 0, %arg9] [1, 2, 1, 1] [1, 1, 1, 1] : memref<1x2x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_13 = memref.subview %subview_11[0, 0, %arg9, 0] [2, 1, 1, 16] [1, 1, 1, 1] : memref<2x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<2x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %7 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_14 = memref.subview %subview_13[0, 0, 0, %arg11] [2, 1, 1, 1] [1, 1, 1, 1] : memref<2x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<2x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_12, %subview_14 : memref<1x2x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<2x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_15 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_17: f32, %out: f32):
# CHECK-NEXT:                  %8 = arith.mulf %in, %in_17 fastmath<fast> : f32
# CHECK-NEXT:                  %9 = arith.addf %out, %8 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %9 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_16 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_15, %subview_16 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"c/r0/f"}
# CHECK-NEXT:              scf.yield %7 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"c/r0/c"}
# CHECK-NEXT:            scf.yield %6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"c/r0/s"}
# CHECK-NEXT:          %subview_7 = memref.subview %subview_3[0, 2, 0, 0] [1, 3, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %arg1[2, 0, 0, 0] [3, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<3x5x3x16xf32, strided<[240, 48, 16, 1], offset: 480>>
# CHECK-NEXT:          %5 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %4) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_10 = memref.subview %subview_7[0, 0, %arg7, 0] [1, 3, 1, 3] [1, 1, 1, 1] : memref<1x3x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_8[0, %arg7, 0, 0] [3, 1, 3, 16] [1, 1, 1, 1] : memref<3x5x3x16xf32, strided<[240, 48, 16, 1], offset: 480>> to memref<3x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %6 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[0, 0, 0, %arg9] [1, 3, 1, 1] [1, 1, 1, 1] : memref<1x3x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_13 = memref.subview %subview_11[0, 0, %arg9, 0] [3, 1, 1, 16] [1, 1, 1, 1] : memref<3x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<3x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %7 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_14 = memref.subview %subview_13[0, 0, 0, %arg11] [3, 1, 1, 1] [1, 1, 1, 1] : memref<3x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<3x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_12, %subview_14 : memref<1x3x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<3x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_15 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_17: f32, %out: f32):
# CHECK-NEXT:                  %8 = arith.mulf %in, %in_17 fastmath<fast> : f32
# CHECK-NEXT:                  %9 = arith.addf %out, %8 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %9 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_16 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_15, %subview_16 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"c/r1/f"}
# CHECK-NEXT:              scf.yield %7 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"c/r1/c"}
# CHECK-NEXT:            scf.yield %6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"c/r1/s"}
# CHECK-NEXT:          %subview_9 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %5, %subview_9 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"c/w"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_2 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:      } {"c/h"}
# CHECK-NEXT:      memref.copy %0, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:    - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
