# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir_tensor",
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
# CHECK-NEXT:   func.func @matmul(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %2 = linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./k" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_3 = arith.constant 0 : index
# CHECK-NEXT:       %c32 = arith.constant 32 : index
# CHECK-NEXT:       %c1_4 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_3 to %c32 step %c1_4 iter_args(%arg6 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %4 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_5 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_6 = tensor.insert_slice %4 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_6 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_0 = arith.constant 0 : index
# CHECK-NEXT:     %c4_1 = arith.constant 4 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0_0 to %c4_1 step %c1_2 iter_args(%arg4 = %1) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x32xf32> to tensor<512x32xf32>
# CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %c0_5 = arith.constant 0 : index
# CHECK-NEXT:       %c32 = arith.constant 32 : index
# CHECK-NEXT:       %c1_6 = arith.constant 1 : index
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0_5 to %c32 step %c1_6 iter_args(%arg6 = %extracted_slice_4) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, 0] [1, 512] [1, 1] : tensor<1x512xf32> to tensor<1x512xf32>
# CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[0, %arg5] [512, 1] [1, 1] : tensor<512x32xf32> to tensor<512x1xf32>
# CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %c0_10 = arith.constant 0 : index
# CHECK-NEXT:         %c512 = arith.constant 512 : index
# CHECK-NEXT:         %c1_11 = arith.constant 1 : index
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0_10 to %c512 step %c1_11 iter_args(%arg8 = %extracted_slice_9) -> (tensor<1x1xf32>) {
# CHECK-NEXT:           %extracted_slice_13 = tensor.extract_slice %extracted_slice_7[0, %arg7] [1, 1] [1, 1] : tensor<1x512xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_14 = tensor.extract_slice %extracted_slice_8[%arg7, 0] [1, 1] [1, 1] : tensor<512x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_15 = tensor.extract_slice %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %5 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_13, %extracted_slice_14 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_15 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:           %inserted_slice_16 = tensor.insert_slice %5 into %arg8[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_16 : tensor<1x1xf32>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %inserted_slice_12 = tensor.insert_slice %4 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_12 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c512 = arith.constant 512 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_1, %subview_2 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[%arg3, 0] [1, 512] [1, 1] : memref<4x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_2 = memref.subview %arg1[0, %arg5] [512, 1] [1, 1] : memref<512x32xf32> to memref<512x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %3 = scf.for %arg7 = %c0 to %c512 step %c1 iter_args(%arg8 = %subview_3) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_5 = memref.subview %subview[0, %arg7] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_2[%arg7, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           linalg.matmul {__xtc_id_C_} ins(%subview_5, %subview_6 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%arg8 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %3, %subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_1 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     memref.copy %1, %arg2 : memref<4x32xf32> to memref<4x32xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: matmul
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 4x512xfloat32
# CHECK-NEXT:   - %1 : 512x32xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %2 : 4x32xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
