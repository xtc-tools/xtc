# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul_relu") as gb:
    m = O.matmul(a, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="matmul")
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %2 = linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %3 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %cst_0 : tensor<4x32xf32>, f32) outs(%3 : tensor<4x32xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:       %5 = arith.maximumf %in, %in_1 : f32
# CHECK-NEXT:       linalg.yield %5 : f32
# CHECK-NEXT:     } -> tensor<4x32xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_matmul_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_matmul_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./k" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./i1" : !transform.any_op
# CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_8) : (!transform.any_op) -> ()
# CHECK-NEXT:     transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:     %2 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     %3 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %3 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./i" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<1x16xf32>
# CHECK-NEXT:     %0 = ub.poison : f32
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c512 = arith.constant 512 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %extracted_slice) -> (tensor<1x32xf32>) {
# CHECK-NEXT:         %extracted_slice_4 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:         %7 = linalg.fill {__xtc_id_matmul_0_} ins(%cst_0 : f32) outs(%extracted_slice_4 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:         %inserted_slice_5 = tensor.insert_slice %7 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_5 : tensor<1x32xf32>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %2) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[0, %arg3] [4, 1] [1, 1] : tensor<4x512xf32> to tensor<4x1xf32>
# CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %arg1[%arg3, 0] [1, 32] [1, 1] : tensor<512x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %6 = scf.for %arg5 = %c0 to %c4 step %c2 iter_args(%arg6 = %arg4) -> (tensor<4x32xf32>) {
# CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %extracted_slice[%arg5, 0] [2, 1] [1, 1] : tensor<4x1xf32> to tensor<2x1xf32>
# CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %arg6[%arg5, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
# CHECK-NEXT:         %7 = scf.for %arg7 = %c0 to %c32 step %c16 iter_args(%arg8 = %extracted_slice_6) -> (tensor<2x32xf32>) {
# CHECK-NEXT:           %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, %arg7] [1, 16] [1, 1] : tensor<1x32xf32> to tensor<1x16xf32>
# CHECK-NEXT:           %extracted_slice_8 = tensor.extract_slice %arg8[0, %arg7] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
# CHECK-NEXT:           %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%c0, 0] [1, 1] [1, 1] : tensor<2x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_10 = tensor.extract_slice %extracted_slice_8[%c0, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:           %8 = vector.transfer_read %extracted_slice_9[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x1xf32>
# CHECK-NEXT:           %9 = vector.transfer_read %extracted_slice_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x16xf32>
# CHECK-NEXT:           %10 = vector.transfer_read %extracted_slice_10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x16xf32>
# CHECK-NEXT:           %11 = vector.extract %9[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %12 = vector.extract %8[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:           %13 = vector.broadcast %12 : f32 to vector<16xf32>
# CHECK-NEXT:           %14 = vector.extract %10[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %15 = vector.fma %13, %11, %14 : vector<16xf32>
# CHECK-NEXT:           %16 = vector.insert %15, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:           %17 = vector.transfer_write %16, %extracted_slice_10[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, tensor<1x16xf32>
# CHECK-NEXT:           %inserted_slice_11 = tensor.insert_slice %17 into %extracted_slice_8[%c0, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
# CHECK-NEXT:           %extracted_slice_12 = tensor.extract_slice %extracted_slice_5[%c1, 0] [1, 1] [1, 1] : tensor<2x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:           %extracted_slice_13 = tensor.extract_slice %inserted_slice_11[%c1, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
# CHECK-NEXT:           %18 = vector.transfer_read %extracted_slice_12[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x1xf32>
# CHECK-NEXT:           %19 = vector.transfer_read %extracted_slice_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x16xf32>
# CHECK-NEXT:           %20 = vector.transfer_read %extracted_slice_13[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x16xf32>
# CHECK-NEXT:           %21 = vector.extract %19[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %22 = vector.extract %18[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:           %23 = vector.broadcast %22 : f32 to vector<16xf32>
# CHECK-NEXT:           %24 = vector.extract %20[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %25 = vector.fma %23, %21, %24 : vector<16xf32>
# CHECK-NEXT:           %26 = vector.insert %25, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:           %27 = vector.transfer_write %26, %extracted_slice_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, tensor<1x16xf32>
# CHECK-NEXT:           %inserted_slice_14 = tensor.insert_slice %27 into %inserted_slice_11[%c1, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
# CHECK-NEXT:           %inserted_slice_15 = tensor.insert_slice %inserted_slice_14 into %arg8[0, %arg7] [2, 16] [1, 1] : tensor<2x16xf32> into tensor<2x32xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_15 : tensor<2x32xf32>
# CHECK-NEXT:         } {"./j"}
# CHECK-NEXT:         %inserted_slice = tensor.insert_slice %7 into %arg6[%arg5, 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:         scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:       } {"./i"}
# CHECK-NEXT:       scf.yield %6 : tensor<4x32xf32>
# CHECK-NEXT:     } {"./k"}
# CHECK-NEXT:     %4 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %c0_1 = arith.constant 0 : index
# CHECK-NEXT:     %c4_2 = arith.constant 4 : index
# CHECK-NEXT:     %c1_3 = arith.constant 1 : index
# CHECK-NEXT:     %5 = scf.for %arg3 = %c0_1 to %c4_2 step %c1_3 iter_args(%arg4 = %4) -> (tensor<4x32xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %3[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:       %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %cst_0 : tensor<1x32xf32>, f32) outs(%extracted_slice_4 : tensor<1x32xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:       ^bb0(%in: f32, %in_5: f32, %out: f32):
# CHECK-NEXT:         %7 = arith.maximumf %in, %in_5 : f32
# CHECK-NEXT:         linalg.yield %7 : f32
# CHECK-NEXT:       } -> tensor<1x32xf32>
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %6 into %arg4[%arg3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = ub.poison : f32
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c512 = arith.constant 512 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %alloca) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %subview) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %subview_1, %subview_2 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %4, %subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %1) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %4 = scf.for %arg5 = %c0 to %c4 step %c2 iter_args(%arg6 = %arg4) -> (memref<4x32xf32>) {
# CHECK-NEXT:         %subview_1 = memref.subview %subview[%arg5, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[%arg5, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %5 = scf.for %arg7 = %c0 to %c32 step %c16 iter_args(%arg8 = %subview_2) -> (memref<2x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_4 = memref.subview %subview_0[0, %arg7] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_5 = memref.subview %arg8[0, %arg7] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_1[0, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_7 = memref.subview %subview_5[0, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %6 = vector.transfer_read %subview_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:           %7 = vector.transfer_read %subview_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:           %8 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:           %9 = vector.extract %7[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %10 = vector.extract %6[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:           %11 = vector.broadcast %10 : f32 to vector<16xf32>
# CHECK-NEXT:           %12 = vector.extract %8[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %13 = vector.fma %11, %9, %12 : vector<16xf32>
# CHECK-NEXT:           %14 = vector.broadcast %13 : vector<16xf32> to vector<1x16xf32>
# CHECK-NEXT:           vector.transfer_write %14, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_8 = memref.subview %subview_5[0, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_7, %subview_8 : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_9 = memref.subview %subview_1[1, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_10 = memref.subview %subview_5[1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %15 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:           %16 = vector.transfer_read %subview_10[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:           %17 = vector.extract %15[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:           %18 = vector.broadcast %17 : f32 to vector<16xf32>
# CHECK-NEXT:           %19 = vector.extract %16[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:           %20 = vector.fma %18, %9, %19 : vector<16xf32>
# CHECK-NEXT:           %21 = vector.broadcast %20 : vector<16xf32> to vector<1x16xf32>
# CHECK-NEXT:           vector.transfer_write %21, %subview_10[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_11 = memref.subview %subview_5[1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_10, %subview_11 : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_12 = memref.subview %arg8[0, %arg7] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %subview_5, %subview_12 : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         } {"./j"}
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[%arg5, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %5, %subview_3 : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<4x32xf32>
# CHECK-NEXT:       } {"./i"}
# CHECK-NEXT:       scf.yield %4 : memref<4x32xf32>
# CHECK-NEXT:     } {"./k"}
# CHECK-NEXT:     %3 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<4x32xf32>) {
# CHECK-NEXT:       %subview = memref.subview %2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%subview, %cst : memref<1x32xf32, strided<[32, 1], offset: ?>>, f32) outs(%subview_0 : memref<1x32xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:       ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:         %4 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:         linalg.yield %4 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %subview_0, %subview_1 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<4x32xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     memref.copy %3, %arg2 : memref<4x32xf32> to memref<4x32xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: matmul_relu
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 4x512xfloat32
# CHECK-NEXT:   - %1 : 512x32xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %3 : 4x32xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'matmul'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:   - %3: relu(%2) {name = 'relu'} : [4x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
