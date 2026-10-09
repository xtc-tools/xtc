# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir, mlir-target=llvmir
# TODO: update xtc-translate to account for mlir-scope

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    c = O.conv2d(p, b, stride=(SH, SW), name="conv")
    O.relu(c, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="conv")
sch.interchange(["b", "h", "w", "r", "s", "c", "f"])
sch.parallelize(["h","w"])
sch.fuse_producer_at("w", 0)
sch.fuse_consumer_at("w")
sch.vectorize(["f"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_relu_mlir_tensor_fused",
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
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x12x12x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %7 = linalg.index 0 : index
# CHECK-NEXT:        %8 = linalg.index 1 : index
# CHECK-NEXT:        %9 = linalg.index 2 : index
# CHECK-NEXT:        %10 = linalg.index 3 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %11 = arith.subi %7, %c0_2 : index
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:        %13 = arith.cmpi slt, %11, %c1 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %14 = arith.subi %8, %c2 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %15 = arith.cmpi sge, %14, %c0 : index
# CHECK-NEXT:        %16 = arith.cmpi slt, %14, %c8 : index
# CHECK-NEXT:        %c2_3 = arith.constant 2 : index
# CHECK-NEXT:        %17 = arith.subi %9, %c2_3 : index
# CHECK-NEXT:        %c8_4 = arith.constant 8 : index
# CHECK-NEXT:        %18 = arith.cmpi sge, %17, %c0 : index
# CHECK-NEXT:        %19 = arith.cmpi slt, %17, %c8_4 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %20 = arith.subi %10, %c0_5 : index
# CHECK-NEXT:        %c3 = arith.constant 3 : index
# CHECK-NEXT:        %21 = arith.cmpi sge, %20, %c0 : index
# CHECK-NEXT:        %22 = arith.cmpi slt, %20, %c3 : index
# CHECK-NEXT:        %23 = arith.andi %12, %13 : i1
# CHECK-NEXT:        %24 = arith.andi %23, %15 : i1
# CHECK-NEXT:        %25 = arith.andi %24, %16 : i1
# CHECK-NEXT:        %26 = arith.andi %25, %18 : i1
# CHECK-NEXT:        %27 = arith.andi %26, %19 : i1
# CHECK-NEXT:        %28 = arith.andi %27, %21 : i1
# CHECK-NEXT:        %29 = arith.andi %28, %22 : i1
# CHECK-NEXT:        %30 = scf.if %29 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%11, %14, %17, %20] : tensor<1x8x8x3xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %30 : f32
# CHECK-NEXT:      } -> tensor<1x12x12x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%1, %arg1 : tensor<1x12x12x3xf32>, tensor<5x5x3x16xf32>) outs(%3 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.mulf %in, %in_2 fastmath<fast> : f32
# CHECK-NEXT:        %8 = arith.addf %out, %7 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %8 : f32
# CHECK-NEXT:      } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %5 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x4x4x16xf32>, f32) outs(%5 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %7 : f32
# CHECK-NEXT:      } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %2 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./h" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %3 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %4 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./w" : !transform.any_op
# CHECK-NEXT:      %5 = transform.structured.match attributes {__xtc_id_pad_} in %new_containing_op_1 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %5 into %forall_op_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %6 = transform.structured.match attributes {__xtc_id_conv_} in %new_containing_op_1 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %6 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./c" : !transform.any_op
# CHECK-NEXT:      %7 = transform.get_parent_op %tiled_linalg_op_10 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %7 {
# CHECK-NEXT:        transform.apply_patterns.xtc.fold_unit_extent_dims_via_slices_for_vectorization
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %8 = transform.structured.match interface{LinalgOp} in %7 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%8) : (!transform.any_op) -> ()
# CHECK-NEXT:      %9 = transform.get_parent_op %new_containing_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %9 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %10 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops:3 = transform.xtc.fuse_consumer %10 into %new_containing_op, %new_containing_op_1, %new_containing_op_5 : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops#0 "./b" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#1 "./h" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#2 "./w" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1) -> (d1, d0)>
# CHECK-NEXT:  #map6 = affine_map<(d0, d1) -> (d0)>
# CHECK-NEXT:  #map7 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: tensor<1x8x8x3xf32> {llvm.noalias}, %arg1: tensor<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<1x12x12x3xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %3 = linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%2 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      %4 = tensor.empty() : tensor<1x4x4x16xf32>
# CHECK-NEXT:      %5:2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %3, %arg5 = %4) -> (tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %1[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : tensor<1x12x12x3xf32> to tensor<1x11x11x3xf32>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x4x4x16xf32>
# CHECK-NEXT:        %7:2 = scf.forall (%arg6) in (4) shared_outs(%arg7 = %extracted_slice_0, %arg8 = %extracted_slice_1) -> (tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) {
# CHECK-NEXT:          %9 = affine.apply #map(%arg6)
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %9, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : tensor<1x11x11x3xf32> to tensor<1x5x11x3xf32>
# CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %arg7[0, %arg6, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:          %extracted_slice_5 = tensor.extract_slice %arg8[0, %arg6, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:          %10:2 = scf.forall (%arg9) in (4) shared_outs(%arg10 = %extracted_slice_4, %arg11 = %extracted_slice_5) -> (tensor<1x1x4x16xf32>, tensor<1x1x4x16xf32>) {
# CHECK-NEXT:            %12 = affine.apply #map(%arg9)
# CHECK-NEXT:            %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, 0, %12, 0] [1, 5, 5, 3] [1, 1, 1, 1] : tensor<1x5x11x3xf32> to tensor<1x5x5x3xf32>
# CHECK-NEXT:            %13 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%extracted_slice_6 : tensor<1x5x5x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:            ^bb0(%out: f32):
# CHECK-NEXT:              %16 = affine.apply #map2(%arg3)[%c0]
# CHECK-NEXT:              %17 = linalg.index 1 : index
# CHECK-NEXT:              %18 = affine.apply #map3(%arg6)[%17]
# CHECK-NEXT:              %19 = linalg.index 2 : index
# CHECK-NEXT:              %20 = affine.apply #map3(%arg9)[%19]
# CHECK-NEXT:              %21 = linalg.index 3 : index
# CHECK-NEXT:              %22 = arith.cmpi sge, %16, %c0 : index
# CHECK-NEXT:              %23 = arith.cmpi slt, %16, %c1 : index
# CHECK-NEXT:              %24 = arith.subi %18, %c2 : index
# CHECK-NEXT:              %25 = arith.cmpi sge, %24, %c0 : index
# CHECK-NEXT:              %26 = arith.cmpi slt, %24, %c8 : index
# CHECK-NEXT:              %27 = arith.subi %20, %c2 : index
# CHECK-NEXT:              %28 = arith.cmpi sge, %27, %c0 : index
# CHECK-NEXT:              %29 = arith.cmpi slt, %27, %c8 : index
# CHECK-NEXT:              %30 = arith.cmpi sge, %21, %c0 : index
# CHECK-NEXT:              %31 = arith.cmpi slt, %21, %c3 : index
# CHECK-NEXT:              %32 = arith.andi %22, %23 : i1
# CHECK-NEXT:              %33 = arith.andi %32, %25 : i1
# CHECK-NEXT:              %34 = arith.andi %33, %26 : i1
# CHECK-NEXT:              %35 = arith.andi %34, %28 : i1
# CHECK-NEXT:              %36 = arith.andi %35, %29 : i1
# CHECK-NEXT:              %37 = arith.andi %36, %30 : i1
# CHECK-NEXT:              %38 = arith.andi %37, %31 : i1
# CHECK-NEXT:              %39 = scf.if %38 -> (f32) {
# CHECK-NEXT:                %extracted = tensor.extract %arg0[%16, %24, %27, %21] : tensor<1x8x8x3xf32>
# CHECK-NEXT:                scf.yield %extracted : f32
# CHECK-NEXT:              } else {
# CHECK-NEXT:                scf.yield %cst : f32
# CHECK-NEXT:              }
# CHECK-NEXT:              linalg.yield %39 : f32
# CHECK-NEXT:            } -> tensor<1x5x5x3xf32>
# CHECK-NEXT:            %extracted_slice_7 = tensor.extract_slice %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %14 = scf.for %arg12 = %c0 to %c5 step %c1 iter_args(%arg13 = %extracted_slice_7) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_9 = tensor.extract_slice %13[0, %arg12, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : tensor<1x5x5x3xf32> to tensor<1x1x5x3xf32>
# CHECK-NEXT:              %extracted_slice_10 = tensor.extract_slice %arg1[%arg12, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : tensor<5x5x3x16xf32> to tensor<1x5x3x16xf32>
# CHECK-NEXT:              %16 = scf.for %arg14 = %c0 to %c5 step %c1 iter_args(%arg15 = %arg13) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                %extracted_slice_11 = tensor.extract_slice %extracted_slice_9[0, 0, %arg14, 0] [1, 1, 1, 3] [1, 1, 1, 1] : tensor<1x1x5x3xf32> to tensor<1x1x1x3xf32>
# CHECK-NEXT:                %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, %arg14, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x5x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                %17 = scf.for %arg16 = %c0 to %c3 step %c1 iter_args(%arg17 = %arg15) -> (tensor<1x1x1x16xf32>) {
# CHECK-NEXT:                  %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, 0, 0, %arg16] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x3xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                  %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, %arg16, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                  %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> to tensor<1xf32>
# CHECK-NEXT:                  %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:                  %extracted_slice_17 = tensor.extract_slice %arg17[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<16xf32>
# CHECK-NEXT:                  %18 = vector.transfer_read %extracted_slice_15[%c0], %0 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
# CHECK-NEXT:                  %19 = vector.transfer_read %extracted_slice_16[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x16xf32>
# CHECK-NEXT:                  %20 = vector.transfer_read %extracted_slice_17[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:                  %21 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %18, %19, %20 : vector<1xf32>, vector<1x16xf32> into vector<16xf32>
# CHECK-NEXT:                  %22 = vector.transfer_write %21, %extracted_slice_17[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
# CHECK-NEXT:                  %inserted_slice_18 = tensor.insert_slice %22 into %arg17[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<16xf32> into tensor<1x1x1x16xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_18 : tensor<1x1x1x16xf32>
# CHECK-NEXT:                } {"./c"}
# CHECK-NEXT:                scf.yield %17 : tensor<1x1x1x16xf32>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:              scf.yield %16 : tensor<1x1x1x16xf32>
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:            %extracted_slice_8 = tensor.extract_slice %arg11[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:            %15 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %cst : tensor<1x1x1x16xf32>, f32) outs(%extracted_slice_8 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_9: f32, %out: f32):
# CHECK-NEXT:              %16 = arith.maximumf %in, %in_9 : f32
# CHECK-NEXT:              linalg.yield %16 : f32
# CHECK-NEXT:            } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:            scf.forall.in_parallel {
# CHECK-NEXT:              tensor.parallel_insert_slice %14 into %arg10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:              tensor.parallel_insert_slice %15 into %arg11[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:            }
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:          %11 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10#0, %cst : tensor<1x1x4x16xf32>, f32) outs(%extracted_slice_5 : tensor<1x1x4x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_6: f32, %out: f32):
# CHECK-NEXT:            %12 = arith.maximumf %in, %in_6 : f32
# CHECK-NEXT:            linalg.yield %12 : f32
# CHECK-NEXT:          } -> tensor<1x1x4x16xf32>
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %10#0 into %arg7[0, %arg6, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:            tensor.parallel_insert_slice %10#1 into %arg8[0, %arg6, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        %8 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7#0, %cst : tensor<1x4x4x16xf32>, f32) outs(%extracted_slice_1 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_3: f32, %out: f32):
# CHECK-NEXT:          %9 = arith.maximumf %in, %in_3 : f32
# CHECK-NEXT:          linalg.yield %9 : f32
# CHECK-NEXT:        } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %7#0 into %arg4[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:        %inserted_slice_2 = tensor.insert_slice %7#1 into %arg5[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : tensor<1x4x4x16xf32> into tensor<1x4x4x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice, %inserted_slice_2 : tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %6 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5#0, %cst : tensor<1x4x4x16xf32>, f32) outs(%4 : tensor<1x4x4x16xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %7 = arith.maximumf %in, %in_0 : f32
# CHECK-NEXT:        linalg.yield %7 : f32
# CHECK-NEXT:      } -> tensor<1x4x4x16xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %5#1 in restrict writable %arg2 : (tensor<1x4x4x16xf32>, memref<1x4x4x16xf32>) -> ()
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
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x4x4x16xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%alloca : memref<1x4x4x16xf32>)
# CHECK-NEXT:      scf.forall (%arg3) in (4) {
# CHECK-NEXT:        %subview = memref.subview %alloca[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg2[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.forall (%arg4) in (4) {
# CHECK-NEXT:          %alloca_2 = memref.alloca() {alignment = 256 : i64} : memref<1x5x5x3xf32>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloca_2 : memref<1x5x5x3xf32>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %2 = linalg.index 1 : index
# CHECK-NEXT:            %3 = affine.apply #map1(%arg3)[%2]
# CHECK-NEXT:            %4 = linalg.index 2 : index
# CHECK-NEXT:            %5 = affine.apply #map1(%arg4)[%4]
# CHECK-NEXT:            %6 = linalg.index 3 : index
# CHECK-NEXT:            %7 = arith.subi %3, %c2 : index
# CHECK-NEXT:            %8 = arith.cmpi sge, %7, %c0 : index
# CHECK-NEXT:            %9 = arith.cmpi slt, %7, %c8 : index
# CHECK-NEXT:            %10 = arith.subi %5, %c2 : index
# CHECK-NEXT:            %11 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:            %12 = arith.cmpi slt, %10, %c8 : index
# CHECK-NEXT:            %13 = arith.cmpi sge, %6, %c0 : index
# CHECK-NEXT:            %14 = arith.cmpi slt, %6, %c3 : index
# CHECK-NEXT:            %15 = arith.andi %8, %9 : i1
# CHECK-NEXT:            %16 = arith.andi %15, %11 : i1
# CHECK-NEXT:            %17 = arith.andi %16, %12 : i1
# CHECK-NEXT:            %18 = arith.andi %17, %13 : i1
# CHECK-NEXT:            %19 = arith.andi %18, %14 : i1
# CHECK-NEXT:            %20 = scf.if %19 -> (f32) {
# CHECK-NEXT:              %21 = memref.load %arg0[%c0, %7, %10, %6] : memref<1x8x8x3xf32>
# CHECK-NEXT:              scf.yield %21 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %20 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_3 = memref.subview %subview[0, 0, %arg4, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %alloca_4 = memref.alloca() {alignment = 256 : i64} : memref<1x1x1x16xf32>
# CHECK-NEXT:          memref.copy %subview_3, %alloca_4 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32>
# CHECK-NEXT:          %1 = scf.for %arg5 = %c0 to %c5 step %c1 iter_args(%arg6 = %alloca_4) -> (memref<1x1x1x16xf32>) {
# CHECK-NEXT:            %subview_7 = memref.subview %alloca_2[0, %arg5, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32> to memref<1x1x5x3xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_8 = memref.subview %arg1[%arg5, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %2 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %arg6) -> (memref<1x1x1x16xf32>) {
# CHECK-NEXT:              %subview_9 = memref.subview %subview_7[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_10 = memref.subview %subview_8[0, %arg7, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %3 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32>) {
# CHECK-NEXT:                %subview_11 = memref.subview %subview_9[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[75, 15, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_12 = memref.subview %subview_10[0, 0, %arg9, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_13 = memref.subview %subview_11[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[75, 15, 3, 1], offset: ?>> to memref<1xf32, strided<[75], offset: ?>>
# CHECK-NEXT:                %subview_14 = memref.subview %subview_12[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x16xf32, strided<[240, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32> to memref<16xf32, strided<[1]>>
# CHECK-NEXT:                %4 = vector.transfer_read %subview_13[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[75], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                %5 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[240, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %6 = vector.transfer_read %subview_15[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1]>>, vector<16xf32>
# CHECK-NEXT:                %7 = vector.extract %5[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %8 = vector.extract %4[0] : f32 from vector<1xf32>
# CHECK-NEXT:                %9 = vector.broadcast %8 : f32 to vector<16xf32>
# CHECK-NEXT:                %10 = vector.fma %7, %9, %6 : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %10, %subview_15[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1]>>
# CHECK-NEXT:                %subview_16 = memref.subview %arg10[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32> to memref<16xf32, strided<[1]>>
# CHECK-NEXT:                memref.copy %subview_15, %subview_16 : memref<16xf32, strided<[1]>> to memref<16xf32, strided<[1]>>
# CHECK-NEXT:                scf.yield %arg10 : memref<1x1x1x16xf32>
# CHECK-NEXT:              } {"./c"}
# CHECK-NEXT:              scf.yield %3 : memref<1x1x1x16xf32>
# CHECK-NEXT:            } {"./s"}
# CHECK-NEXT:            scf.yield %2 : memref<1x1x1x16xf32>
# CHECK-NEXT:          } {"./r"}
# CHECK-NEXT:          %subview_5 = memref.subview %subview_0[0, 0, %arg4, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %cst : memref<1x1x1x16xf32>, f32) outs(%subview_5 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_7: f32, %out: f32):
# CHECK-NEXT:            %2 = arith.maximumf %in, %in_7 : f32
# CHECK-NEXT:            linalg.yield %2 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_6 = memref.subview %subview_0[0, 0, %arg4, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_5, %subview_6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %subview_0, %subview_1 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %arg2, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
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
# CHECK-NEXT:    - %4 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:    - %4: relu(%3) {name = 'relu'} : [1x4x4x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
