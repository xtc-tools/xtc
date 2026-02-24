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
    dump_file="gen_pad_tuple_matmul_unpad_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_matmul_unpad(%arg0: tensor<14x14xf32> {llvm.noalias}, %arg1: tensor<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:     %inserted_slice = tensor.insert_slice %arg0 into %1[0, 0] [14, 14] [1, 1] {__xtc_id_A_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %2 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %3 = linalg.fill {__xtc_id_B_pad_0_} ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:     %inserted_slice_1 = tensor.insert_slice %arg1 into %3[0, 0] [14, 14] [1, 1] {__xtc_id_B_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %4 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %5 = linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_2 : f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:     %6 = linalg.matmul {__xtc_id_matmul_padded_} ins(%inserted_slice, %inserted_slice_1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:     %7 = tensor.empty() : tensor<14x14xf32>
# CHECK-NEXT:     %extracted_slice = tensor.extract_slice %6[0, 0] [14, 14] [1, 1] {__xtc_id_C_} : tensor<16x16xf32> to tensor<14x14xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %extracted_slice in restrict writable %arg2 : (tensor<14x14xf32>, memref<14x14xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_A_pad_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_B_pad_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_matmul_padded_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./j" : !transform.any_op
# CHECK-NEXT:     %3 = transform.structured.match attributes {__xtc_id_matmul_padded_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %3 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./k" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_matmul_unpad(%arg0: tensor<14x14xf32> {llvm.noalias}, %arg1: tensor<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %0) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %inserted_slice = tensor.insert_slice %arg0 into %1[0, 0] [14, 14] [1, 1] {__xtc_id_A_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %2 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_1 = arith.constant 0 : index
# CHECK-NEXT:     %c16_2 = arith.constant 16 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0_1 to %c16_2 step %c1_3 iter_args(%arg4 = %2) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_B_pad_0_} ins(%cst_0 : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %inserted_slice_4 = tensor.insert_slice %arg1 into %3[0, 0] [14, 14] [1, 1] {__xtc_id_B_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %4 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_5 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_6 = arith.constant 0 : index
# CHECK-NEXT:     %c16_7 = arith.constant 16 : index
# CHECK-NEXT:     %c1_8 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg3 = %c0_6 to %c16_7 step %c1_8 iter_args(%arg4 = %4) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_5 : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_9 = arith.constant 0 : index
# CHECK-NEXT:     %c16_10 = arith.constant 16 : index
# CHECK-NEXT:     %c1_11 = arith.constant 1 : index
# CHECK-NEXT:     %6 = scf.for %arg3 = %c0_9 to %c16_10 step %c1_11 iter_args(%arg4 = %5) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %inserted_slice[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %extracted_slice_13 = tensor.extract_slice %inserted_slice_4[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> to tensor<16x16xf32>
# CHECK-NEXT:       %extracted_slice_14 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_15 = arith.constant 0 : index
# CHECK-NEXT:       %c16_16 = arith.constant 16 : index
# CHECK-NEXT:       %c1_17 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_15 to %c16_16 step %c1_17 iter_args(%arg6 = %extracted_slice_14) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_19 = tensor.extract_slice %extracted_slice_12[0, 0] [1, 16] [1, 1] : tensor<1x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:         %extracted_slice_20 = tensor.extract_slice %extracted_slice_13[0, %arg5] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
# CHECK-NEXT:         %extracted_slice_21 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_22 = arith.constant 0 : index
# CHECK-NEXT:         %c16_23 = arith.constant 16 : index
# CHECK-NEXT:         %c1_24 = arith.constant 1 : index
# CHECK-NEXT:         %9 = scf.for %arg7 = %c0_22 to %c16_23 step %c1_24 iter_args(%arg8 = %extracted_slice_21) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_26 = tensor.extract_slice %extracted_slice_19[0, %arg7] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_27 = tensor.extract_slice %extracted_slice_20[%arg7, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_28 = tensor.extract_slice %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %10 = linalg.matmul {__xtc_id_matmul_padded_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_28 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_29 = tensor.insert_slice %10 into %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_29 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_25 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_25 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_18 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_18 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %7 = tensor.empty() : tensor<14x14xf32>
# CHECK-NEXT:     %extracted_slice = tensor.extract_slice %6[0, 0] [14, 14] [1, 1] {__xtc_id_C_} : tensor<16x16xf32> to tensor<14x14xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %extracted_slice in restrict writable %arg2 : (tensor<14x14xf32>, memref<14x14xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_matmul_unpad(%arg0: tensor<14x14xf32> {llvm.noalias}, %arg1: tensor<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %0) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %inserted_slice = tensor.insert_slice %arg0 into %1[0, 0] [14, 14] [1, 1] {__xtc_id_A_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %2 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_1 = arith.constant 0 : index
# CHECK-NEXT:     %c16_2 = arith.constant 16 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0_1 to %c16_2 step %c1_3 iter_args(%arg4 = %2) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_B_pad_0_} ins(%cst_0 : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %inserted_slice_4 = tensor.insert_slice %arg1 into %3[0, 0] [14, 14] [1, 1] {__xtc_id_B_pad_} : tensor<14x14xf32> into tensor<16x16xf32>
# CHECK-NEXT:     %4 = tensor.empty() : tensor<16x16xf32>
# CHECK-NEXT:     %cst_5 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_6 = arith.constant 0 : index
# CHECK-NEXT:     %c16_7 = arith.constant 16 : index
# CHECK-NEXT:     %c1_8 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg3 = %c0_6 to %c16_7 step %c1_8 iter_args(%arg4 = %4) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_13 = arith.constant 0 : index
# CHECK-NEXT:       %c16_14 = arith.constant 16 : index
# CHECK-NEXT:       %c1_15 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_13 to %c16_14 step %c1_15 iter_args(%arg6 = %extracted_slice_12) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %9 = linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_5 : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_18 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_18 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_16 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_16 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_9 = arith.constant 0 : index
# CHECK-NEXT:     %c16_10 = arith.constant 16 : index
# CHECK-NEXT:     %c1_11 = arith.constant 1 : index
# CHECK-NEXT:     %6 = scf.for %arg3 = %c0_9 to %c16_10 step %c1_11 iter_args(%arg4 = %5) -> (tensor<16x16xf32>) {
# CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %inserted_slice[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %extracted_slice_13 = tensor.extract_slice %inserted_slice_4[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> to tensor<16x16xf32>
# CHECK-NEXT:       %extracted_slice_14 = tensor.extract_slice %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:       %c0_15 = arith.constant 0 : index
# CHECK-NEXT:       %c16_16 = arith.constant 16 : index
# CHECK-NEXT:       %c1_17 = arith.constant 1 : index
# CHECK-NEXT:       %8 = scf.for %arg5 = %c0_15 to %c16_16 step %c1_17 iter_args(%arg6 = %extracted_slice_14) -> (tensor<1x16xf32>) {
# CHECK-NEXT:         %extracted_slice_19 = tensor.extract_slice %extracted_slice_12[0, 0] [1, 16] [1, 1] : tensor<1x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:         %extracted_slice_20 = tensor.extract_slice %extracted_slice_13[0, %arg5] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
# CHECK-NEXT:         %extracted_slice_21 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_22 = arith.constant 0 : index
# CHECK-NEXT:         %c16_23 = arith.constant 16 : index
# CHECK-NEXT:         %c1_24 = arith.constant 1 : index
# CHECK-NEXT:         %9 = scf.for %arg7 = %c0_22 to %c16_23 step %c1_24 iter_args(%arg8 = %extracted_slice_21) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_26 = tensor.extract_slice %extracted_slice_19[0, %arg7] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_27 = tensor.extract_slice %extracted_slice_20[%arg7, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_28 = tensor.extract_slice %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %10 = linalg.matmul {__xtc_id_matmul_padded_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_28 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_29 = tensor.insert_slice %10 into %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_29 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_25 = tensor.insert_slice %9 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_25 : tensor<1x16xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice_18 = tensor.insert_slice %8 into %arg4[%arg3, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
# CHECK-NEXT:       scf.yield %inserted_slice_18 : tensor<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %7 = tensor.empty() : tensor<14x14xf32>
# CHECK-NEXT:     %extracted_slice = tensor.extract_slice %6[0, 0] [14, 14] [1, 1] {__xtc_id_C_} : tensor<16x16xf32> to tensor<14x14xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %extracted_slice in restrict writable %arg2 : (tensor<14x14xf32>, memref<14x14xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @pad_matmul_unpad(%arg0: memref<14x14xf32> {llvm.noalias}, %arg1: memref<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:     %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca_0) -> (memref<16x16xf32>) {
# CHECK-NEXT:       %subview_4 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_4) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_6 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:         %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_6, %subview_7 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_5 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_5 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %subview = memref.subview %0[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:     memref.copy %arg0, %subview : memref<14x14xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:     %alloca_1 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca_1) -> (memref<16x16xf32>) {
# CHECK-NEXT:       %subview_4 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_4) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_6 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_B_pad_0_} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:         %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_6, %subview_7 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_5 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_5 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %subview_2 = memref.subview %1[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:     memref.copy %arg1, %subview_2 : memref<14x14xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %alloca) -> (memref<16x16xf32>) {
# CHECK-NEXT:       %subview_4 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_4) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_6 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:         %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_6, %subview_7 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_5 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_5 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %2) -> (memref<16x16xf32>) {
# CHECK-NEXT:       %subview_4 = memref.subview %0[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       %subview_5 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %subview_5) -> (memref<1x16xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_7 = memref.subview %1[0, %arg5] [16, 1] [1, 1] : memref<16x16xf32> to memref<16x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         %subview_8 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         %5 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %subview_8) -> (memref<1x1xf32, strided<[16, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_10 = memref.subview %subview_4[0, %arg7] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:           %subview_11 = memref.subview %subview_7[%arg7, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:           linalg.matmul {__xtc_id_matmul_padded_} ins(%subview_10, %subview_11 : memref<1x1xf32, strided<[16, 1], offset: ?>>, memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%arg8 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %subview_9 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %5, %subview_9 : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_6 = memref.subview %arg4[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_6 : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<16x16xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %subview_3 = memref.subview %3[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:     memref.copy %subview_3, %arg2 : memref<14x14xf32, strided<[16, 1]>> to memref<14x14xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: pad_matmul_unpad
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 14x14xfloat32
# CHECK-NEXT:   - %1 : 14x14xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %5 : 14x14xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: pad(%0, padding=(0, 2), constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:   - %3: pad(%1, padding=(0, 2), constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:   - %4: matmul(%2, %3) {name = 'matmul_padded'} : [16x16xfloat32, 16x16xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:   - %5: unpad(%4, padding=(0, 2)) {name = 'C'} : [16x16xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
