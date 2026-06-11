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
sch.interchange(["b", "h", "w", "r", "s", "c", "f"])
sch.fuse_producer_at("w", 0)
sch.vectorize(["f"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_nhwc_mini_mlir_tensor_fused",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
    #print_lowered_ir=True,
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {sym_name = "pad_conv2d_nhwc_mini"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %0 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %2 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %3 into %loops_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./w" : !transform.any_op
# CHECK-NEXT:      %5 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %5 into %loops_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %6 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op_3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %6 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./c" : !transform.any_op
# CHECK-NEXT:      %7 = transform.get_parent_op %tiled_linalg_op_12 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %7 {
# CHECK-NEXT:        transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %8 = transform.structured.match interface{LinalgOp} in %7 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%8) : (!transform.any_op) -> ()
# CHECK-NEXT:      %9 = transform.get_parent_op %new_containing_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %9 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %4 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %3) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %1[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:        %5 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x4x4x16xf32>) {
# CHECK-NEXT:          %6 = affine.apply #map(%arg5)
# CHECK-NEXT:          %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %6, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:          %7 = scf.for %arg7 = %c0 to %c4 step %c1 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:            %8 = affine.apply #map(%arg7)
# CHECK-NEXT:            %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %8, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:            %9 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_4 : tensor<1x5x5x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:            ^bb0(%out: f32):
# CHECK-NEXT:              %11 = affine.apply #map2(%arg3)[%c0]
# CHECK-NEXT:              %12 = linalg.index 1 : index
# CHECK-NEXT:              %13 = affine.apply #map3(%arg5)[%12]
# CHECK-NEXT:              %14 = linalg.index 2 : index
# CHECK-NEXT:              %15 = affine.apply #map3(%arg7)[%14]
# CHECK-NEXT:              %16 = linalg.index 3 : index
# CHECK-NEXT:              %17 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:              %18 = arith.cmpi slt, %11, %c1 : index
# CHECK-NEXT:              %19 = arith.subi %13, %c2 : index
# CHECK-NEXT:              %20 = arith.cmpi sge, %19, %c0 : index
# CHECK-NEXT:              %21 = arith.cmpi slt, %19, %c8 : index
# CHECK-NEXT:              %22 = arith.subi %15, %c2 : index
# CHECK-NEXT:              %23 = arith.cmpi sge, %22, %c0 : index
# CHECK-NEXT:              %24 = arith.cmpi slt, %22, %c8 : index
# CHECK-NEXT:              %25 = arith.cmpi sge, %16, %c0 : index
# CHECK-NEXT:              %26 = arith.cmpi slt, %16, %c3 : index
# CHECK-NEXT:              %27 = arith.andi %17, %18 : i1
# CHECK-NEXT:              %28 = arith.andi %27, %20 : i1
# CHECK-NEXT:              %29 = arith.andi %28, %21 : i1
# CHECK-NEXT:              %30 = arith.andi %29, %23 : i1
# CHECK-NEXT:              %31 = arith.andi %30, %24 : i1
# CHECK-NEXT:              %32 = arith.andi %31, %25 : i1
# CHECK-NEXT:              %33 = arith.andi %32, %26 : i1
# CHECK-NEXT:              %34 = scf.if %33 -> (f32) {
# CHECK-NEXT:                %extracted = tensor.extract %arg0[%11, %19, %22, %16] : tensor<1x8x8x3xf32>
# CHECK-NEXT:                scf.yield %extracted : f32
# CHECK-NEXT:              } else {
# CHECK-NEXT:                scf.yield %cst : f32
# CHECK-NEXT:              }
# CHECK-NEXT:              linalg.yield %34 : f32
# CHECK-NEXT:            } -> tensor<1x5x5x3xf32>
# CHECK-NEXT:            %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %10 = scf.for %arg9 = %c0 to %c5 step %c1 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_7 = tensor.extract_slice %9[0, %arg9, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x1x5x3xf32>
# CHECK-NEXT:              %extracted_slice_8 = tensor.extract_slice %arg1[%arg9, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<1x5x3x16xf32>
# CHECK-NEXT:              %11 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %arg10) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                %extracted_slice_9 = tensor.extract_slice %extracted_slice_7[0, 0, %arg11, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x5x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                %extracted_slice_10 = tensor.extract_slice %extracted_slice_8[0, %arg11, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x5x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                %12 = scf.for %arg13 = %c0 to %c3 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                  %extracted_slice_11 = tensor.extract_slice %extracted_slice_9[0, 0, 0, %arg13] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                  %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<16xf32>
# CHECK-NEXT:                  %extracted_slice_15 = tensor.extract_slice %arg14[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<16xf32>
# CHECK-NEXT:                  %extracted_slice_16 = tensor.extract_slice %extracted_slice_13[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<f32>
# CHECK-NEXT:                  %extracted = tensor.extract %extracted_slice_16[] : tensor<f32>
# CHECK-NEXT:                  %13 = vector.broadcast %extracted : f32 to vector<16xf32>
# CHECK-NEXT:                  %14 = vector.transfer_read %extracted_slice_14[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:                  %15 = vector.transfer_read %extracted_slice_15[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:                  %16 = arith.mulf %13, %14 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                  %17 = arith.addf %15, %16 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                  %18 = vector.transfer_write %17, %extracted_slice_15[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
# CHECK-NEXT:                  %inserted_slice_17 = tensor.insert_slice %18 into %arg14[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_17 : tensor<1x1x1x16xf32>
# CHECK-NEXT:                } {"./c"}
# CHECK-NEXT:                scf.yield %12 : tensor<1x1x1x16xf32>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:              scf.yield %11 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:            %inserted_slice_6 = tensor.insert_slice %10 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_6 : tensor<1x1x4x16xf32>
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %inserted_slice_3 = tensor.insert_slice %7 into %arg6[0, %arg5, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_3 : tensor<1x4x4x16xf32>
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1x4x4x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {sym_name = "pad_conv2d_nhwc_mini"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %0 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map1 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x5x5x3xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x4x4x16xf32>)
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca : memref<1x5x5x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %4 = linalg.index 1 : index
# CHECK-NEXT:            %5 = affine.apply #map1(%arg3)[%4]
# CHECK-NEXT:            %6 = linalg.index 2 : index
# CHECK-NEXT:            %7 = affine.apply #map1(%arg5)[%6]
# CHECK-NEXT:            %8 = linalg.index 3 : index
# CHECK-NEXT:            %9 = arith.subi %5, %c2 : index
# CHECK-NEXT:            %10 = arith.cmpi sge, %9, %c0 : index
# CHECK-NEXT:            %11 = arith.cmpi slt, %9, %c8 : index
# CHECK-NEXT:            %12 = arith.subi %7, %c2 : index
# CHECK-NEXT:            %13 = arith.cmpi sge, %12, %c0 : index
# CHECK-NEXT:            %14 = arith.cmpi slt, %12, %c8 : index
# CHECK-NEXT:            %15 = arith.cmpi sge, %8, %c0 : index
# CHECK-NEXT:            %16 = arith.cmpi slt, %8, %c3 : index
# CHECK-NEXT:            %17 = arith.andi %10, %11 : i1
# CHECK-NEXT:            %18 = arith.andi %17, %13 : i1
# CHECK-NEXT:            %19 = arith.andi %18, %14 : i1
# CHECK-NEXT:            %20 = arith.andi %19, %15 : i1
# CHECK-NEXT:            %21 = arith.andi %20, %16 : i1
# CHECK-NEXT:            %22 = scf.if %21 -> (f32) {
# CHECK-NEXT:              %23 = memref.load %arg0[%c0, %9, %12, %8] : memref<1x8x8x3xf32>
# CHECK-NEXT:              scf.yield %23 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %22 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_1 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %subview_1) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_3 = memref.subview %alloca[0, %arg7, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32> to memref<1x1x5x3xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_4 = memref.subview %arg1[%arg7, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg9 = %c0 to %c5 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_5 = memref.subview %subview_3[0, 0, %arg9, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_6 = memref.subview %subview_4[0, %arg9, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_7 = memref.subview %subview_5[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_8 = memref.subview %subview_6[0, 0, %arg11, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_9 = memref.subview %subview_7[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1x1xf32, strided<[75, 15], offset: ?>>
# CHECK-NEXT:                %subview_10 = memref.subview %subview_8[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_11 = memref.subview %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_12 = memref.subview %subview_9[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[75, 15], offset: ?>> to memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %6 = memref.load %subview_12[] : memref<f32, strided<[], offset: ?>>
# CHECK-NEXT:                %7 = vector.broadcast %6 : f32 to vector<16xf32>
# CHECK-NEXT:                %8 = vector.transfer_read %subview_10[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %9 = vector.transfer_read %subview_11[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %10 = arith.mulf %7, %8 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                %11 = arith.addf %9, %10 fastmath<fast> : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %11, %subview_11[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_13 = memref.subview %arg12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_11, %subview_13 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"./c"}
# CHECK-NEXT:              scf.yield %5 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"./s"}
# CHECK-NEXT:            scf.yield %4 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./r"}
# CHECK-NEXT:          %subview_2 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_2 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_0 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_0 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %1, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
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
