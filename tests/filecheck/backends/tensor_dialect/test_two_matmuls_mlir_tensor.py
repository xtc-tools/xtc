# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")
c = O.tensor((J, I), dtype, name="C")

with O.graph(name="matmul") as gb:
    d = O.matmul(a, b, name="D")
    O.matmul(c, d, name="E")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node = "E")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="two_matmul_mlir_tensor",
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
# CHECK-NEXT:   func.func @matmul(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: tensor<32x4xf32> {llvm.noalias}, %arg3: memref<32x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_D_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %2 = linalg.matmul {__xtc_id_D_} ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %3 = tensor.empty() : tensor<32x32xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %4 = linalg.fill {__xtc_id_E_0_} ins(%cst_0 : f32) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
# CHECK-NEXT:     %5 = linalg.matmul {__xtc_id_E_} ins(%arg2, %2 : tensor<32x4xf32>, tensor<4x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_D_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_D_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./k" : !transform.any_op
# CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_E_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./j" : !transform.any_op
# CHECK-NEXT:     %3 = transform.structured.match attributes {__xtc_id_E_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %3 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./k" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: tensor<32x4xf32> {llvm.noalias}, %arg3: memref<32x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c32_10 = arith.constant 32 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_9 to %c32_10 step %c1_11 iter_args(%arg7 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %7 = linalg.fill {__xtc_id_D_0_} ins(%cst : f32) outs(%extracted_slice_12 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_13 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_0 = arith.constant 0 : index
# CHECK-NEXT:     %c4_1 = arith.constant 4 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg4 = %c0_0 to %c4_1 step %c1_2 iter_args(%arg5 = %1) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x32xf32> to tensor<512x32xf32>
# CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_11 = arith.constant 0 : index
# CHECK-NEXT:       %c32_12 = arith.constant 32 : index
# CHECK-NEXT:       %c1_13 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_11 to %c32_12 step %c1_13 iter_args(%arg7 = %extracted_slice_10) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0] [1, 512] [1, 1] : tensor<1x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %extracted_slice_9[0, %arg6] [512, 1] [1, 1] : tensor<512x32xf32> to tensor<512x1xf32>
# CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_17 = arith.constant 0 : index
# CHECK-NEXT:         %c512 = arith.constant 512 : index
# CHECK-NEXT:         %c1_18 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg8 = %c0_17 to %c512 step %c1_18 iter_args(%arg9 = %extracted_slice_16) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, %arg8] [1, 1] [1, 1] : tensor<1x512xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[%arg8, 0] [1, 1] [1, 1] : tensor<512x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %8 = linalg.matmul {__xtc_id_D_} ins(%extracted_slice_20, %extracted_slice_21 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_22 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_23 = tensor.insert_slice %8 into %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_23 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_19 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_19 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %3 = tensor.empty() : tensor<32x32xf32>
# CHECK-NEXT:     %cst_3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_4 = arith.constant 0 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1_5 = arith.constant 1 : index
# CHECK-NEXT:     %4 = scf.for %arg4 = %c0_4 to %c32 step %c1_5 iter_args(%arg5 = %3) -> (tensor<32x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c32_10 = arith.constant 32 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_9 to %c32_10 step %c1_11 iter_args(%arg7 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %7 = linalg.fill {__xtc_id_E_0_} ins(%cst_3 : f32) outs(%extracted_slice_12 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_13 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_6 = arith.constant 0 : index
# CHECK-NEXT:     %c32_7 = arith.constant 32 : index
# CHECK-NEXT:     %c1_8 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg4 = %c0_6 to %c32_7 step %c1_8 iter_args(%arg5 = %4) -> (tensor<32x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg2[%arg4, 0] [1, 4] [1, 1] : tensor<32x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %2[0, 0] [4, 32] [1, 1] : tensor<4x32xf32> to tensor<4x32xf32>
# CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_11 = arith.constant 0 : index
# CHECK-NEXT:       %c32_12 = arith.constant 32 : index
# CHECK-NEXT:       %c1_13 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_11 to %c32_12 step %c1_13 iter_args(%arg7 = %extracted_slice_10) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %extracted_slice_9[0, %arg6] [4, 1] [1, 1] : tensor<4x32xf32> to tensor<4x1xf32>
# CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_17 = arith.constant 0 : index
# CHECK-NEXT:         %c4_18 = arith.constant 4 : index
# CHECK-NEXT:         %c1_19 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg8 = %c0_17 to %c4_18 step %c1_19 iter_args(%arg9 = %extracted_slice_16) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %extracted_slice_14[0, %arg8] [1, 1] [1, 1] : tensor<1x4xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %extracted_slice_15[%arg8, 0] [1, 1] [1, 1] : tensor<4x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_23 = tensor.extract_slice %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %8 = linalg.matmul {__xtc_id_E_} ins(%extracted_slice_21, %extracted_slice_22 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_23 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_24 = tensor.insert_slice %8 into %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_24 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_20 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_20 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: tensor<32x4xf32> {llvm.noalias}, %arg3: memref<32x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c32_10 = arith.constant 32 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_9 to %c32_10 step %c1_11 iter_args(%arg7 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %7 = linalg.fill {__xtc_id_D_0_} ins(%cst : f32) outs(%extracted_slice_12 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_13 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_0 = arith.constant 0 : index
# CHECK-NEXT:     %c4_1 = arith.constant 4 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg4 = %c0_0 to %c4_1 step %c1_2 iter_args(%arg5 = %1) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x32xf32> to tensor<512x32xf32>
# CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_11 = arith.constant 0 : index
# CHECK-NEXT:       %c32_12 = arith.constant 32 : index
# CHECK-NEXT:       %c1_13 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_11 to %c32_12 step %c1_13 iter_args(%arg7 = %extracted_slice_10) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0] [1, 512] [1, 1] : tensor<1x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %extracted_slice_9[0, %arg6] [512, 1] [1, 1] : tensor<512x32xf32> to tensor<512x1xf32>
# CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_17 = arith.constant 0 : index
# CHECK-NEXT:         %c512 = arith.constant 512 : index
# CHECK-NEXT:         %c1_18 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg8 = %c0_17 to %c512 step %c1_18 iter_args(%arg9 = %extracted_slice_16) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, %arg8] [1, 1] [1, 1] : tensor<1x512xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[%arg8, 0] [1, 1] [1, 1] : tensor<512x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %8 = linalg.matmul {__xtc_id_D_} ins(%extracted_slice_20, %extracted_slice_21 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_22 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_23 = tensor.insert_slice %8 into %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_23 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_19 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_19 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %3 = tensor.empty() : tensor<32x32xf32>
# CHECK-NEXT:     %cst_3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_4 = arith.constant 0 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1_5 = arith.constant 1 : index
# CHECK-NEXT:     %4 = scf.for %arg4 = %c0_4 to %c32 step %c1_5 iter_args(%arg5 = %3) -> (tensor<32x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c32_10 = arith.constant 32 : index
# CHECK-NEXT:       %c1_11 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_9 to %c32_10 step %c1_11 iter_args(%arg7 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %7 = linalg.fill {__xtc_id_E_0_} ins(%cst_3 : f32) outs(%extracted_slice_12 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_13 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_6 = arith.constant 0 : index
# CHECK-NEXT:     %c32_7 = arith.constant 32 : index
# CHECK-NEXT:     %c1_8 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg4 = %c0_6 to %c32_7 step %c1_8 iter_args(%arg5 = %4) -> (tensor<32x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg2[%arg4, 0] [1, 4] [1, 1] : tensor<32x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %2[0, 0] [4, 32] [1, 1] : tensor<4x32xf32> to tensor<4x32xf32>
# CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_11 = arith.constant 0 : index
# CHECK-NEXT:       %c32_12 = arith.constant 32 : index
# CHECK-NEXT:       %c1_13 = arith.constant 1 : index
# CHECK-NEXT:       %6 = scf.for %arg6 = %c0_11 to %c32_12 step %c1_13 iter_args(%arg7 = %extracted_slice_10) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %extracted_slice_9[0, %arg6] [4, 1] [1, 1] : tensor<4x32xf32> to tensor<4x1xf32>
# CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_17 = arith.constant 0 : index
# CHECK-NEXT:         %c4_18 = arith.constant 4 : index
# CHECK-NEXT:         %c1_19 = arith.constant 1 : index
# CHECK-NEXT:         %7 = scf.for %arg8 = %c0_17 to %c4_18 step %c1_19 iter_args(%arg9 = %extracted_slice_16) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_21 = tensor.extract_slice %extracted_slice_14[0, %arg8] [1, 1] [1, 1] : tensor<1x4xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %extracted_slice_15[%arg8, 0] [1, 1] [1, 1] : tensor<4x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_23 = tensor.extract_slice %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %8 = linalg.matmul {__xtc_id_E_} ins(%extracted_slice_21, %extracted_slice_22 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_23 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_24 = tensor.insert_slice %8 into %arg9[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_24 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_20 = tensor.insert_slice %7 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_20 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<32x4xf32> {llvm.noalias}, %arg3: memref<32x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c512 = arith.constant 512 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:     %0 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %alloca) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_D_0_} ins(%cst : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:         %subview_2 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_1, %subview_2 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[%arg4, 0] [1, 512] [1, 1] : memref<4x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview_0) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %arg1[0, %arg6] [512, 1] [1, 1] : memref<512x32xf32> to memref<512x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %5 = scf.for %arg8 = %c0 to %c512 step %c1 iter_args(%arg9 = %subview_3) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_5 = memref.subview %subview[0, %arg8] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_2[%arg8, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           linalg.matmul {__xtc_id_D_} ins(%subview_5, %subview_6 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%arg9 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:           scf.yield %arg9 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %5, %subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_1 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %2 = scf.for %arg4 = %c0 to %c32 step %c1 iter_args(%arg5 = %arg3) -> (memref<32x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_E_0_} ins(%cst : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:         %subview_2 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_1, %subview_2 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg5 : memref<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %3 = scf.for %arg4 = %c0 to %c32 step %c1 iter_args(%arg5 = %2) -> (memref<32x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg2[%arg4, 0] [1, 4] [1, 1] : memref<32x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview_0) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %1[0, %arg6] [4, 1] [1, 1] : memref<4x32xf32> to memref<4x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %5 = scf.for %arg8 = %c0 to %c4 step %c1 iter_args(%arg9 = %subview_3) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_5 = memref.subview %subview[0, %arg8] [1, 1] [1, 1] : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x1xf32, strided<[4, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_2[%arg8, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           linalg.matmul {__xtc_id_E_} ins(%subview_5, %subview_6 : memref<1x1xf32, strided<[4, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%arg9 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:           scf.yield %arg9 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %5, %subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_1 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg5 : memref<32x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     memref.copy %3, %arg3 : memref<32x32xf32> to memref<32x32xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: matmul
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 4x512xfloat32
# CHECK-NEXT:   - %1 : 512x32xfloat32
# CHECK-NEXT:   - %2 : 32x4xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %4 : 32x32xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %3: matmul(%0, %1) {name = 'D'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:   - %4: matmul(%2, %3) {name = 'E'} : [32x4xfloat32, 4x32xfloat32] -> [32x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
