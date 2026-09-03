# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad(a, padding=(0, 2), name="A_pad")
    p2 = O.pad(b, padding=(0, 2), name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad(m_pad, padding=(0, 2), name="C")
graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="gen_pad_tuple_matmul_unpad_mlir_tensor",
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
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: tensor<14x14xf32> {llvm.noalias}, %arg1: tensor<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<16x16xf32>) attrs =  {__xtc_id_A_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %8 = linalg.index 0 : index
# CHECK-NEXT:        %9 = linalg.index 1 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %10 = arith.subi %8, %c0_2 : index
# CHECK-NEXT:        %c14 = arith.constant 14 : index
# CHECK-NEXT:        %11 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:        %12 = arith.cmpi slt, %10, %c14 : index
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %13 = arith.subi %9, %c0_3 : index
# CHECK-NEXT:        %c14_4 = arith.constant 14 : index
# CHECK-NEXT:        %14 = arith.cmpi sge, %13, %c0 : index
# CHECK-NEXT:        %15 = arith.cmpi slt, %13, %c14_4 : index
# CHECK-NEXT:        %16 = arith.andi %11, %12 : i1
# CHECK-NEXT:        %17 = arith.andi %16, %14 : i1
# CHECK-NEXT:        %18 = arith.andi %17, %15 : i1
# CHECK-NEXT:        %19 = scf.if %18 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg0[%10, %13] : tensor<14x14xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %19 : f32
# CHECK-NEXT:      } -> tensor<16x16xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<16x16xf32>) attrs =  {__xtc_id_B_pad_} {
# CHECK-NEXT:      ^bb0(%out: f32):
# CHECK-NEXT:        %8 = linalg.index 0 : index
# CHECK-NEXT:        %9 = linalg.index 1 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %10 = arith.subi %8, %c0_2 : index
# CHECK-NEXT:        %c14 = arith.constant 14 : index
# CHECK-NEXT:        %11 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:        %12 = arith.cmpi slt, %10, %c14 : index
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %13 = arith.subi %9, %c0_3 : index
# CHECK-NEXT:        %c14_4 = arith.constant 14 : index
# CHECK-NEXT:        %14 = arith.cmpi sge, %13, %c0 : index
# CHECK-NEXT:        %15 = arith.cmpi slt, %13, %c14_4 : index
# CHECK-NEXT:        %16 = arith.andi %11, %12 : i1
# CHECK-NEXT:        %17 = arith.andi %16, %14 : i1
# CHECK-NEXT:        %18 = arith.andi %17, %15 : i1
# CHECK-NEXT:        %19 = scf.if %18 -> (f32) {
# CHECK-NEXT:          %extracted = tensor.extract %arg1[%10, %13] : tensor<14x14xf32>
# CHECK-NEXT:          scf.yield %extracted : f32
# CHECK-NEXT:        } else {
# CHECK-NEXT:          scf.yield %cst_0 : f32
# CHECK-NEXT:        }
# CHECK-NEXT:        linalg.yield %19 : f32
# CHECK-NEXT:      } -> tensor<16x16xf32>
# CHECK-NEXT:      %4 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %5 = linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_1 : f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:      %6 = linalg.matmul {__xtc_id_matmul_padded_} ins(%1, %3 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:      %7 = tensor.empty() : tensor<14x14xf32>
# CHECK-NEXT:      %extracted_slice = tensor.extract_slice %6[0, 0] [14, 14] [1, 1] {__xtc_id_C_} : tensor<16x16xf32> to tensor<14x14xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %extracted_slice in restrict writable %arg2 : (tensor<14x14xf32>, memref<14x14xf32>) -> ()
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_A_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_B_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_matmul_padded_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0)[s0] -> (d0 + s0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: tensor<14x14xf32> {llvm.noalias}, %arg1: tensor<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %0) -> (tensor<16x16xf32>) {
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:        %c0_9 = arith.constant 0 : index
# CHECK-NEXT:        %c16_10 = arith.constant 16 : index
# CHECK-NEXT:        %c1_11 = arith.constant 1 : index
# CHECK-NEXT:        %8 = scf.for %arg5 = %c0_9 to %c16_10 step %c1_11 iter_args(%arg6 = %extracted_slice_8) -> (tensor<1x16xf32>) {
# CHECK-NEXT:          %extracted_slice_12 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_12 : tensor<1x1xf32>) attrs =  {__xtc_id_A_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %10 = linalg.index 0 : index
# CHECK-NEXT:            %11 = affine.apply #map1(%arg3)[%10]
# CHECK-NEXT:            %12 = linalg.index 1 : index
# CHECK-NEXT:            %13 = affine.apply #map1(%arg5)[%12]
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %c0_15 = arith.constant 0 : index
# CHECK-NEXT:            %14 = arith.subi %11, %c0_15 : index
# CHECK-NEXT:            %c14 = arith.constant 14 : index
# CHECK-NEXT:            %15 = arith.cmpi sge, %14, %c0_14 : index
# CHECK-NEXT:            %16 = arith.cmpi slt, %14, %c14 : index
# CHECK-NEXT:            %c0_16 = arith.constant 0 : index
# CHECK-NEXT:            %17 = arith.subi %13, %c0_16 : index
# CHECK-NEXT:            %c14_17 = arith.constant 14 : index
# CHECK-NEXT:            %18 = arith.cmpi sge, %17, %c0_14 : index
# CHECK-NEXT:            %19 = arith.cmpi slt, %17, %c14_17 : index
# CHECK-NEXT:            %20 = arith.andi %15, %16 : i1
# CHECK-NEXT:            %21 = arith.andi %20, %18 : i1
# CHECK-NEXT:            %22 = arith.andi %21, %19 : i1
# CHECK-NEXT:            %23 = scf.if %22 -> (f32) {
# CHECK-NEXT:              %extracted = tensor.extract %arg0[%14, %17] : tensor<14x14xf32>
# CHECK-NEXT:              scf.yield %extracted : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %23 : f32
# CHECK-NEXT:          } -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_13 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_13 : tensor<1x16xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %2 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0_1 = arith.constant 0 : index
# CHECK-NEXT:      %c16_2 = arith.constant 16 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %3 = scf.for %arg3 = %c0_1 to %c16_2 step %c1_3 iter_args(%arg4 = %2) -> (tensor<16x16xf32>) {
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:        %c0_9 = arith.constant 0 : index
# CHECK-NEXT:        %c16_10 = arith.constant 16 : index
# CHECK-NEXT:        %c1_11 = arith.constant 1 : index
# CHECK-NEXT:        %8 = scf.for %arg5 = %c0_9 to %c16_10 step %c1_11 iter_args(%arg6 = %extracted_slice_8) -> (tensor<1x16xf32>) {
# CHECK-NEXT:          %extracted_slice_12 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_12 : tensor<1x1xf32>) attrs =  {__xtc_id_B_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %10 = linalg.index 0 : index
# CHECK-NEXT:            %11 = affine.apply #map1(%arg3)[%10]
# CHECK-NEXT:            %12 = linalg.index 1 : index
# CHECK-NEXT:            %13 = affine.apply #map1(%arg5)[%12]
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %c0_15 = arith.constant 0 : index
# CHECK-NEXT:            %14 = arith.subi %11, %c0_15 : index
# CHECK-NEXT:            %c14 = arith.constant 14 : index
# CHECK-NEXT:            %15 = arith.cmpi sge, %14, %c0_14 : index
# CHECK-NEXT:            %16 = arith.cmpi slt, %14, %c14 : index
# CHECK-NEXT:            %c0_16 = arith.constant 0 : index
# CHECK-NEXT:            %17 = arith.subi %13, %c0_16 : index
# CHECK-NEXT:            %c14_17 = arith.constant 14 : index
# CHECK-NEXT:            %18 = arith.cmpi sge, %17, %c0_14 : index
# CHECK-NEXT:            %19 = arith.cmpi slt, %17, %c14_17 : index
# CHECK-NEXT:            %20 = arith.andi %15, %16 : i1
# CHECK-NEXT:            %21 = arith.andi %20, %18 : i1
# CHECK-NEXT:            %22 = arith.andi %21, %19 : i1
# CHECK-NEXT:            %23 = scf.if %22 -> (f32) {
# CHECK-NEXT:              %extracted = tensor.extract %arg1[%14, %17] : tensor<14x14xf32>
# CHECK-NEXT:              scf.yield %extracted : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst_0 : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %23 : f32
# CHECK-NEXT:          } -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_13 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_13 : tensor<1x16xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %4 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %5 = linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_4 : f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:      %c0_5 = arith.constant 0 : index
# CHECK-NEXT:      %c16_6 = arith.constant 16 : index
# CHECK-NEXT:      %c1_7 = arith.constant 1 : index
# CHECK-NEXT:      %6 = scf.for %arg3 = %c0_5 to %c16_6 step %c1_7 iter_args(%arg4 = %5) -> (tensor<16x16xf32>) {
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %1[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:        %extracted_slice_9 = tensor.extract_slice %3[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> to tensor<16x16xf32>
# CHECK-NEXT:        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:        %c0_11 = arith.constant 0 : index
# CHECK-NEXT:        %c16_12 = arith.constant 16 : index
# CHECK-NEXT:        %c1_13 = arith.constant 1 : index
# CHECK-NEXT:        %8 = scf.for %arg5 = %c0_11 to %c16_12 step %c1_13 iter_args(%arg6 = %extracted_slice_10) -> (tensor<1x16xf32>) {
# CHECK-NEXT:          %extracted_slice_14 = tensor.extract_slice %extracted_slice_8[0, 0] [1, 16] [1, 1] : tensor<1x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:          %extracted_slice_15 = tensor.extract_slice %extracted_slice_9[0, %arg5] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
# CHECK-NEXT:          %extracted_slice_16 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %c0_17 = arith.constant 0 : index
# CHECK-NEXT:          %c16_18 = arith.constant 16 : index
# CHECK-NEXT:          %c1_19 = arith.constant 1 : index
# CHECK-NEXT:          %9 = scf.for %arg7 = %c0_17 to %c16_18 step %c1_19 iter_args(%arg8 = %extracted_slice_16) -> (tensor<1x1xf32>) {
# CHECK-NEXT:            %extracted_slice_21 = tensor.extract_slice %extracted_slice_14[0, %arg7] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:            %extracted_slice_22 = tensor.extract_slice %extracted_slice_15[%arg7, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:            %extracted_slice_23 = tensor.extract_slice %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:            %10 = linalg.matmul {__xtc_id_matmul_padded_} ins(%extracted_slice_21, %extracted_slice_22 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_23 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:            %inserted_slice_24 = tensor.insert_slice %10 into %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_24 : tensor<1x1xf32>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          %inserted_slice_20 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_20 : tensor<1x16xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %7 = tensor.empty() : tensor<14x14xf32>
# CHECK-NEXT:      %extracted_slice = tensor.extract_slice %6[0, 0] [14, 14] [1, 1] {__xtc_id_C_} : tensor<16x16xf32> to tensor<14x14xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %extracted_slice in restrict writable %arg2 : (tensor<14x14xf32>, memref<14x14xf32>) -> ()
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
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: memref<14x14xf32> {llvm.noalias}, %arg1: memref<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c14 = arith.constant 14 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca_0) -> (memref<16x16xf32>) {
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_2) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview_4 : memref<1x1xf32, strided<[16, 1], offset: ?>>) attrs =  {__xtc_id_A_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %4 = arith.cmpi sge, %arg3, %c0 : index
# CHECK-NEXT:            %5 = arith.cmpi slt, %arg3, %c14 : index
# CHECK-NEXT:            %6 = arith.cmpi sge, %arg5, %c0 : index
# CHECK-NEXT:            %7 = arith.cmpi slt, %arg5, %c14 : index
# CHECK-NEXT:            %8 = arith.andi %4, %5 : i1
# CHECK-NEXT:            %9 = arith.andi %8, %6 : i1
# CHECK-NEXT:            %10 = arith.andi %9, %7 : i1
# CHECK-NEXT:            %11 = scf.if %10 -> (f32) {
# CHECK-NEXT:              %12 = memref.load %arg0[%arg3, %arg5] : memref<14x14xf32>
# CHECK-NEXT:              scf.yield %12 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %11 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_5 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_4, %subview_5 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_3 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %3, %subview_3 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %alloca_1 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca_1) -> (memref<16x16xf32>) {
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_2) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview_4 : memref<1x1xf32, strided<[16, 1], offset: ?>>) attrs =  {__xtc_id_B_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %4 = arith.cmpi sge, %arg3, %c0 : index
# CHECK-NEXT:            %5 = arith.cmpi slt, %arg3, %c14 : index
# CHECK-NEXT:            %6 = arith.cmpi sge, %arg5, %c0 : index
# CHECK-NEXT:            %7 = arith.cmpi slt, %arg5, %c14 : index
# CHECK-NEXT:            %8 = arith.andi %4, %5 : i1
# CHECK-NEXT:            %9 = arith.andi %8, %6 : i1
# CHECK-NEXT:            %10 = arith.andi %9, %7 : i1
# CHECK-NEXT:            %11 = scf.if %10 -> (f32) {
# CHECK-NEXT:              %12 = memref.load %arg1[%arg3, %arg5] : memref<14x14xf32>
# CHECK-NEXT:              scf.yield %12 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %11 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_5 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_4, %subview_5 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_3 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %3, %subview_3 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst : f32) outs(%alloca : memref<16x16xf32>)
# CHECK-NEXT:      %2 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca) -> (memref<16x16xf32>) {
# CHECK-NEXT:        %subview_2 = memref.subview %0[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_3) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_5 = memref.subview %1[0, %arg5] [16, 1] [1, 1] : memref<16x16xf32> to memref<16x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %4 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_6) -> (memref<1x1xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_8 = memref.subview %subview_2[0, %arg7] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_5[%arg7, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            linalg.matmul {__xtc_id_matmul_padded_} ins(%subview_8, %subview_9 : memref<1x1xf32, strided<[16, 1], offset: ?>>, memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%arg8 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:            scf.yield %arg8 : memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %4, %subview_7 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_4 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %3, %subview_4 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %subview = memref.subview %2[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      memref.copy %subview, %arg2 : memref<14x14xf32, strided<[16, 1]>> to memref<14x14xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: pad_matmul_unpad
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 14x14xfloat32
# CHECK-NEXT:    - %1 : 14x14xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %5 : 14x14xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad(%0, padding=(0, 2), constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %3: pad(%1, padding=(0, 2), constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %4: matmul(%2, %3) {name = 'matmul_padded'} : [16x16xfloat32, 16x16xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %5: unpad(%4, padding=(0, 2)) {name = 'C'} : [16x16xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
