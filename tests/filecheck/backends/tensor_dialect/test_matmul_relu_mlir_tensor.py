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
# CHECK: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT: module {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %2 = linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>
# CHECK-NEXT:     %3 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:     %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<4x32xf32> into tensor<128xf32>
# CHECK-NEXT:     %4 = tensor.empty() : tensor<128xf32>
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapsed, %cst_0 : tensor<128xf32>, f32) outs(%4 : tensor<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:       %6 = arith.maximumf %in, %in_1 : f32
# CHECK-NEXT:       linalg.yield %6 : f32
# CHECK-NEXT:     } -> tensor<128xf32>
# CHECK-NEXT:     %expanded = tensor.expand_shape %5 [[0, 1]] output_shape [4, 32] : tensor<128xf32> into tensor<4x32xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %expanded in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT: module {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:     linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:     %collapse_shape = memref.collapse_shape %alloca [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:     %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<128xf32>
# CHECK-NEXT:     %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapse_shape, %cst_1 : memref<128xf32>, f32) outs(%alloca_0 : memref<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:       %0 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:       linalg.yield %0 : f32
# CHECK-NEXT:     }
# CHECK-NEXT:     %expand_shape = memref.expand_shape %alloca_0 [[0, 1]] output_shape [4, 32] : memref<128xf32> into memref<4x32xf32>
# CHECK-NEXT:     memref.copy %expand_shape, %arg2 : memref<4x32xf32> to memref<4x32xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:     linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:     %collapse_shape = memref.collapse_shape %alloca [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:     %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<128xf32>
# CHECK-NEXT:     %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapse_shape, %cst_1 : memref<128xf32>, f32) outs(%alloca_0 : memref<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:       %0 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:       linalg.yield %0 : f32
# CHECK-NEXT:     }
# CHECK-NEXT:     %expand_shape = memref.expand_shape %alloca_0 [[0, 1]] output_shape [4, 32] : memref<128xf32> into memref<4x32xf32>
# CHECK-NEXT:     memref.copy %expand_shape, %arg2 : memref<4x32xf32> to memref<4x32xf32>
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
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./k" : !transform.any_op
# CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./i" : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:       %subview = memref.subview %alloca[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %c0_7 = arith.constant 0 : index
# CHECK-NEXT:       %c32 = arith.constant 32 : index
# CHECK-NEXT:       %c1_8 = arith.constant 1 : index
# CHECK-NEXT:       scf.for %arg4 = %c0_7 to %c32 step %c1_8 {
# CHECK-NEXT:         %subview_9 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%subview_9 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %c0_0 = arith.constant 0 : index
# CHECK-NEXT:     %c4_1 = arith.constant 4 : index
# CHECK-NEXT:     %c1_2 = arith.constant 1 : index
# CHECK-NEXT:     scf.for %arg3 = %c0_0 to %c4_1 step %c1_2 {
# CHECK-NEXT:       %subview = memref.subview %arg0[%arg3, 0] [1, 512] [1, 1] : memref<4x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:       %subview_7 = memref.subview %arg1[0, 0] [512, 32] [1, 1] : memref<512x32xf32> to memref<512x32xf32, strided<[32, 1]>>
# CHECK-NEXT:       %subview_8 = memref.subview %alloca[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       %c0_9 = arith.constant 0 : index
# CHECK-NEXT:       %c32 = arith.constant 32 : index
# CHECK-NEXT:       %c1_10 = arith.constant 1 : index
# CHECK-NEXT:       scf.for %arg4 = %c0_9 to %c32 step %c1_10 {
# CHECK-NEXT:         %subview_11 = memref.subview %subview[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:         %subview_12 = memref.subview %subview_7[0, %arg4] [512, 1] [1, 1] : memref<512x32xf32, strided<[32, 1]>> to memref<512x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %subview_13 = memref.subview %subview_8[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %c0_14 = arith.constant 0 : index
# CHECK-NEXT:         %c512 = arith.constant 512 : index
# CHECK-NEXT:         %c1_15 = arith.constant 1 : index
# CHECK-NEXT:         scf.for %arg5 = %c0_14 to %c512 step %c1_15 {
# CHECK-NEXT:           %subview_16 = memref.subview %subview_11[0, %arg5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_17 = memref.subview %subview_12[%arg5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           %subview_18 = memref.subview %subview_13[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           linalg.matmul {__xtc_id_matmul_} ins(%subview_16, %subview_17 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%subview_18 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %collapse_shape = memref.collapse_shape %alloca [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:     %alloca_3 = memref.alloca() {alignment = 256 : i64} : memref<128xf32>
# CHECK-NEXT:     %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0_5 = arith.constant 0 : index
# CHECK-NEXT:     %c128 = arith.constant 128 : index
# CHECK-NEXT:     %c1_6 = arith.constant 1 : index
# CHECK-NEXT:     scf.for %arg3 = %c0_5 to %c128 step %c1_6 {
# CHECK-NEXT:       %subview = memref.subview %collapse_shape[%arg3] [1] [1] : memref<128xf32> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       %subview_7 = memref.subview %alloca_3[%arg3] [1] [1] : memref<128xf32> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%subview, %cst_4 : memref<1xf32, strided<[1], offset: ?>>, f32) outs(%subview_7 : memref<1xf32, strided<[1], offset: ?>>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:       ^bb0(%in: f32, %in_8: f32, %out: f32):
# CHECK-NEXT:         %0 = arith.maximumf %in, %in_8 : f32
# CHECK-NEXT:         linalg.yield %0 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     %expand_shape = memref.expand_shape %alloca_3 [[0, 1]] output_shape [4, 32] : memref<128xf32> into memref<4x32xf32>
# CHECK-NEXT:     memref.copy %expand_shape, %arg2 : memref<4x32xf32> to memref<4x32xf32>
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
