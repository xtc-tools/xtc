# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")
bias = O.tensor((J,), dtype, name="bias")

with O.graph(name="matmul_add") as gb:
    m = O.matmul(a, b, name="matmul")
    O.add(m, bias, name="add")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="matmul")
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["i", "j", "i1", "j1", "k"])
sch.fuse_consumer_at("j1")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_add_mlir_tensor_fused",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_add(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: tensor<32xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:      %2 = linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:      %3 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<4x32xf32>, tensor<32xf32>) outs(%3 : tensor<4x32xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.addf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %5 : f32
# CHECK-NEXT:      } -> tensor<4x32xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_matmul_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_matmul_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./j1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_add_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_consumer, %new_loops:4 = transform.xtc.fuse_consumer %2 into %loops_3, %loops_5, %loops_7, %loops_9 {in_place} : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %new_loops#0 "./i" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#1 "./j" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#2 "./i1" : !transform.any_op
# CHECK-NEXT:      transform.annotate %new_loops#3 "./j1" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_add(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: tensor<32xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0) -> (tensor<4x32xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_3 = arith.constant 1 : index
# CHECK-NEXT:        %5 = scf.for %arg6 = %c0_2 to %c32 step %c1_3 iter_args(%arg7 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %6 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%extracted_slice_4 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_5 = tensor.insert_slice %6 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_5 : tensor<1x32xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %2 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %3:2 = scf.for %arg4 = %c0_0 to %c4_1 step %c2 iter_args(%arg5 = %1, %arg6 = %2) -> (tensor<4x32xf32>, tensor<4x32xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [2, 512] [1, 1] : tensor<4x512xf32> to tensor<2x512xf32>
# CHECK-NEXT:        %extracted_slice_2 = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x32xf32> to tensor<512x32xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg5[%arg4, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c16 = arith.constant 16 : index
# CHECK-NEXT:        %extracted_slice_5 = tensor.extract_slice %arg2[0] [32] [1] : tensor<32xf32> to tensor<32xf32>
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %arg5[%arg4, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
# CHECK-NEXT:        %5:2 = scf.for %arg7 = %c0_4 to %c32 step %c16 iter_args(%arg8 = %extracted_slice_3, %arg9 = %extracted_slice_6) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, 0] [2, 512] [1, 1] : tensor<2x512xf32> to tensor<2x512xf32>
# CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %extracted_slice_2[0, %arg7] [512, 16] [1, 1] : tensor<512x32xf32> to tensor<512x16xf32>
# CHECK-NEXT:          %extracted_slice_9 = tensor.extract_slice %arg8[0, %arg7] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
# CHECK-NEXT:          %c0_10 = arith.constant 0 : index
# CHECK-NEXT:          %c2_11 = arith.constant 2 : index
# CHECK-NEXT:          %c1_12 = arith.constant 1 : index
# CHECK-NEXT:          %extracted_slice_13 = tensor.extract_slice %extracted_slice_5[%arg7] [16] [1] : tensor<32xf32> to tensor<16xf32>
# CHECK-NEXT:          %extracted_slice_14 = tensor.extract_slice %arg8[0, %arg7] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
# CHECK-NEXT:          %7:2 = scf.for %arg10 = %c0_10 to %c2_11 step %c1_12 iter_args(%arg11 = %extracted_slice_9, %arg12 = %extracted_slice_14) -> (tensor<2x16xf32>, tensor<2x16xf32>) {
# CHECK-NEXT:            %extracted_slice_16 = tensor.extract_slice %extracted_slice_7[%arg10, 0] [1, 512] [1, 1] : tensor<2x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:            %extracted_slice_17 = tensor.extract_slice %extracted_slice_8[0, 0] [512, 16] [1, 1] : tensor<512x16xf32> to tensor<512x16xf32>
# CHECK-NEXT:            %extracted_slice_18 = tensor.extract_slice %arg11[%arg10, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:            %c0_19 = arith.constant 0 : index
# CHECK-NEXT:            %c16_20 = arith.constant 16 : index
# CHECK-NEXT:            %c1_21 = arith.constant 1 : index
# CHECK-NEXT:            %extracted_slice_22 = tensor.extract_slice %extracted_slice_13[0] [16] [1] : tensor<16xf32> to tensor<16xf32>
# CHECK-NEXT:            %extracted_slice_23 = tensor.extract_slice %arg11[%arg10, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:            %9:2 = scf.for %arg13 = %c0_19 to %c16_20 step %c1_21 iter_args(%arg14 = %extracted_slice_18, %arg15 = %extracted_slice_23) -> (tensor<1x16xf32>, tensor<1x16xf32>) {
# CHECK-NEXT:              %extracted_slice_25 = tensor.extract_slice %extracted_slice_16[0, 0] [1, 512] [1, 1] : tensor<1x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:              %extracted_slice_26 = tensor.extract_slice %extracted_slice_17[0, %arg13] [512, 1] [1, 1] : tensor<512x16xf32> to tensor<512x1xf32>
# CHECK-NEXT:              %extracted_slice_27 = tensor.extract_slice %arg14[0, %arg13] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:              %c0_28 = arith.constant 0 : index
# CHECK-NEXT:              %c512 = arith.constant 512 : index
# CHECK-NEXT:              %c1_29 = arith.constant 1 : index
# CHECK-NEXT:              %11 = scf.for %arg16 = %c0_28 to %c512 step %c1_29 iter_args(%arg17 = %extracted_slice_27) -> (tensor<1x1xf32>) {
# CHECK-NEXT:                %extracted_slice_33 = tensor.extract_slice %extracted_slice_25[0, %arg16] [1, 1] [1, 1] : tensor<1x512xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_34 = tensor.extract_slice %extracted_slice_26[%arg16, 0] [1, 1] [1, 1] : tensor<512x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_35 = tensor.extract_slice %arg17[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %13 = linalg.matmul {__xtc_id_matmul_} ins(%extracted_slice_33, %extracted_slice_34 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_35 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:                %inserted_slice_36 = tensor.insert_slice %13 into %arg17[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_36 : tensor<1x1xf32>
# CHECK-NEXT:              } {"./k"}
# CHECK-NEXT:              %extracted_slice_30 = tensor.extract_slice %extracted_slice_22[%arg13] [1] [1] : tensor<16xf32> to tensor<1xf32>
# CHECK-NEXT:              %extracted_slice_31 = tensor.extract_slice %arg14[0, %arg13] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:              %12 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %extracted_slice_30 : tensor<1x1xf32>, tensor<1xf32>) outs(%11 : tensor<1x1xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_33: f32, %out: f32):
# CHECK-NEXT:                %13 = arith.addf %in, %in_33 fastmath<fast> : f32
# CHECK-NEXT:                linalg.yield %13 : f32
# CHECK-NEXT:              } -> tensor<1x1xf32>
# CHECK-NEXT:              %inserted_slice_32 = tensor.insert_slice %12 into %arg14[0, %arg13] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_32, %arg15 : tensor<1x16xf32>, tensor<1x16xf32>
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9#0, %extracted_slice_22 : tensor<1x16xf32>, tensor<16xf32>) outs(%extracted_slice_23 : tensor<1x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:            ^bb0(%in: f32, %in_25: f32, %out: f32):
# CHECK-NEXT:              %11 = arith.addf %in, %in_25 fastmath<fast> : f32
# CHECK-NEXT:              linalg.yield %11 : f32
# CHECK-NEXT:            } -> tensor<1x16xf32>
# CHECK-NEXT:            %inserted_slice_24 = tensor.insert_slice %9#0 into %arg11[%arg10, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_24, %arg12 : tensor<2x16xf32>, tensor<2x16xf32>
# CHECK-NEXT:          } {"./i1"}
# CHECK-NEXT:          %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7#0, %extracted_slice_13 : tensor<2x16xf32>, tensor<16xf32>) outs(%extracted_slice_14 : tensor<2x16xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_16: f32, %out: f32):
# CHECK-NEXT:            %9 = arith.addf %in, %in_16 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %9 : f32
# CHECK-NEXT:          } -> tensor<2x16xf32>
# CHECK-NEXT:          %inserted_slice_15 = tensor.insert_slice %7#0 into %arg8[0, %arg7] [2, 16] [1, 1] : tensor<2x16xf32> into tensor<2x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_15, %arg9 : tensor<2x32xf32>, tensor<2x32xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5#0, %extracted_slice_5 : tensor<2x32xf32>, tensor<32xf32>) outs(%extracted_slice_6 : tensor<2x32xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:        ^bb0(%in: f32, %in_7: f32, %out: f32):
# CHECK-NEXT:          %7 = arith.addf %in, %in_7 fastmath<fast> : f32
# CHECK-NEXT:          linalg.yield %7 : f32
# CHECK-NEXT:        } -> tensor<2x32xf32>
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5#0 into %arg5[%arg4, 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        scf.yield %inserted_slice, %arg6 : tensor<4x32xf32>, tensor<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3#0, %arg2 : tensor<4x32xf32>, tensor<32xf32>) outs(%2 : tensor<4x32xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %5 = arith.addf %in, %in_2 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %5 : f32
# CHECK-NEXT:      } -> tensor<4x32xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %3#0 in restrict writable %arg3 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
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
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_add(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<32xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %0 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (memref<4x32xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_1 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:          %subview_2 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_1, %subview_2 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %1 = scf.for %arg4 = %c0 to %c4 step %c2 iter_args(%arg5 = %0) -> (memref<4x32xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg4, 0] [2, 512] [1, 1] : memref<4x512xf32> to memref<2x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg5[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %subview_0) -> (memref<2x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_2 = memref.subview %arg1[0, %arg6] [512, 16] [1, 1] : memref<512x32xf32> to memref<512x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %arg7[0, %arg6] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %arg2[%arg6] [16] [1] : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %subview_3) -> (memref<2x16xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_6 = memref.subview %subview[%arg8, 0] [1, 512] [1, 1] : memref<2x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %arg9[%arg8, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %subview_7) -> (memref<1x16xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_9 = memref.subview %subview_2[0, %arg10] [512, 1] [1, 1] : memref<512x16xf32, strided<[32, 1], offset: ?>> to memref<512x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %subview_10 = memref.subview %arg11[0, %arg10] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %5 = scf.for %arg12 = %c0 to %c512 step %c1 iter_args(%arg13 = %subview_10) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_13 = memref.subview %subview_6[0, %arg12] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                %subview_14 = memref.subview %subview_9[%arg12, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                linalg.matmul {__xtc_id_matmul_} ins(%subview_13, %subview_14 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%arg13 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:                scf.yield %arg13 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              } {"./k"}
# CHECK-NEXT:              %subview_11 = memref.subview %subview_4[%arg10] [1] [1] : memref<16xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:              linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %subview_11 : memref<1x1xf32, strided<[32, 1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>) outs(%5 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:              ^bb0(%in: f32, %in_13: f32, %out: f32):
# CHECK-NEXT:                %6 = arith.addf %in, %in_13 fastmath<fast> : f32
# CHECK-NEXT:                linalg.yield %6 : f32
# CHECK-NEXT:              }
# CHECK-NEXT:              %subview_12 = memref.subview %arg11[0, %arg10] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %5, %subview_12 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg11 : memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %subview_8 = memref.subview %arg9[%arg8, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            memref.copy %4, %subview_8 : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            scf.yield %arg9 : memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {"./i1"}
# CHECK-NEXT:          %subview_5 = memref.subview %arg7[0, %arg6] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_5 : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg7 : memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_1 = memref.subview %arg5[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_1 : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      memref.copy %1, %arg3 : memref<4x32xf32> to memref<4x32xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul_add
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    - %2 : 32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %3: matmul(%0, %1) {name = 'matmul'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:    - %4: add(%3, %2) {name = 'add'} : [4x32xfloat32, 32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
