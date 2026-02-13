# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.artifacts import get_operation
from xtc.artifacts import get_operation

op = get_operation("conv2d", "ResNet18_01")
N, H, W, F, R, S, C = [op["dims"][k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
SH, SW = [op["params"][k] for k in ["SH", "SW"]]
dtype = "float32"

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_r181(%arg0: tensor<1x230x230x3xf32> {llvm.noalias}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x112x112x64xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%0 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
# CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%1 : tensor<1x112x112x64xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:       %3 = arith.mulf %in, %in_0 : f32
# CHECK-NEXT:       %4 = arith.addf %out, %3 : f32
# CHECK-NEXT:       linalg.yield %4 : f32
# CHECK-NEXT:     } -> tensor<1x112x112x64xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x112x112x64xf32>, memref<1x112x112x64xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_O_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 4, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_21 "./w1" : !transform.any_op
# CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_20) : (!transform.any_op) -> ()
# CHECK-NEXT:     transform.loop.unroll %loops_21 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:     transform.loop.unroll %loops_19 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:     %2 = transform.get_parent_op %loops_7 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_r181(%arg0: tensor<1x230x230x3xf32> {llvm.noalias}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c6 = arith.constant 6 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c7 = arith.constant 7 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c64 = arith.constant 64 : index
# CHECK-NEXT:     %c112 = arith.constant 112 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x112x112x64xf32>
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %0) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:         %extracted_slice_0 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c112 step %c1 iter_args(%arg8 = %extracted_slice_0) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:           %extracted_slice_2 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0 to %c64 step %c1 iter_args(%arg10 = %extracted_slice_2) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:             %extracted_slice_4 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %6 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_5 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_5 : tensor<1x1x1x64xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_3 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_3 : tensor<1x1x112x64xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_1 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_1 : tensor<1x112x112x64xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %1) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : tensor<1x230x230x3xf32> to tensor<1x229x229x3xf32>
# CHECK-NEXT:       %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:         %4 = affine.apply #map(%arg5)
# CHECK-NEXT:         %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %4, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : tensor<1x229x229x3xf32> to tensor<1x7x229x3xf32>
# CHECK-NEXT:         %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:         %5 = scf.for %arg7 = %c0 to %c112 step %c4 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:           %6 = affine.apply #map(%arg7)
# CHECK-NEXT:           %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %6, 0] [1, 7, 13, 3] [1, 1, 1, 1] : tensor<1x7x229x3xf32> to tensor<1x7x13x3xf32>
# CHECK-NEXT:           %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x4x64xf32>
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x4x64xf32>) {
# CHECK-NEXT:             %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0, %arg9] [7, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x64xf32> to tensor<7x7x3x16xf32>
# CHECK-NEXT:             %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x64xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:             %8 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:               %extracted_slice_10 = tensor.extract_slice %extracted_slice_4[0, %arg11, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : tensor<1x7x13x3xf32> to tensor<1x1x13x3xf32>
# CHECK-NEXT:               %extracted_slice_11 = tensor.extract_slice %extracted_slice_7[%arg11, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x16xf32> to tensor<1x7x3x16xf32>
# CHECK-NEXT:               %9 = scf.for %arg13 = %c0 to %c7 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                 %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x1x13x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:                 %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x7x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                 %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_17 = tensor.extract_slice %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %10 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_16, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_17 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_18 = tensor.insert_slice %10 into %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_19 = tensor.extract_slice %extracted_slice_14[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_20 = tensor.extract_slice %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %11 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_19, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_20 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_21 = tensor.insert_slice %11 into %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_22 = tensor.extract_slice %extracted_slice_14[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_23 = tensor.extract_slice %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_22, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_23 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_24 = tensor.insert_slice %12 into %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_25 = tensor.extract_slice %extracted_slice_14[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_26 = tensor.extract_slice %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %13 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_25, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_26 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_27 = tensor.insert_slice %13 into %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_28 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_29 = tensor.extract_slice %extracted_slice_13[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_30 = tensor.extract_slice %extracted_slice_28[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_31 = tensor.extract_slice %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_30, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_31 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_32 = tensor.insert_slice %14 into %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_33 = tensor.extract_slice %extracted_slice_28[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %15 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_33, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_34 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_35 = tensor.insert_slice %15 into %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %extracted_slice_28[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_37 = tensor.extract_slice %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %16 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_36, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_37 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_38 = tensor.insert_slice %16 into %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_28[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %17 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_39, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_40 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_41 = tensor.insert_slice %17 into %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_42 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_43 = tensor.extract_slice %extracted_slice_13[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_44 = tensor.extract_slice %extracted_slice_42[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_45 = tensor.extract_slice %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %18 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_45 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_46 = tensor.insert_slice %18 into %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_47 = tensor.extract_slice %extracted_slice_42[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_48 = tensor.extract_slice %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %19 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_47, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_48 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_49 = tensor.insert_slice %19 into %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_50 = tensor.extract_slice %extracted_slice_42[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_51 = tensor.extract_slice %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %20 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_50, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_51 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_52 = tensor.insert_slice %20 into %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_53 = tensor.extract_slice %extracted_slice_42[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_54 = tensor.extract_slice %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %21 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_53, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_54 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_55 = tensor.insert_slice %21 into %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_55 : tensor<1x1x4x16xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               scf.yield %9 : tensor<1x1x4x16xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_9 = tensor.insert_slice %8 into %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x64xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_9 : tensor<1x1x4x64xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_6 = tensor.insert_slice %7 into %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x4x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_6 : tensor<1x1x112x64xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_3 = tensor.insert_slice %5 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_3 : tensor<1x112x112x64xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x112x112x64xf32>, memref<1x112x112x64xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump Before Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_r181(%arg0: tensor<1x230x230x3xf32> {llvm.noalias}, %arg1: tensor<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c6 = arith.constant 6 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c7 = arith.constant 7 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c64 = arith.constant 64 : index
# CHECK-NEXT:     %c112 = arith.constant 112 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = tensor.empty() : tensor<1x112x112x64xf32>
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %0) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:         %extracted_slice_0 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c112 step %c1 iter_args(%arg8 = %extracted_slice_0) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:           %extracted_slice_2 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x1x64xf32>
# CHECK-NEXT:           %5 = scf.for %arg9 = %c0 to %c64 step %c1 iter_args(%arg10 = %extracted_slice_2) -> (tensor<1x1x1x64xf32>) {
# CHECK-NEXT:             %extracted_slice_4 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:             %6 = linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%extracted_slice_4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
# CHECK-NEXT:             %inserted_slice_5 = tensor.insert_slice %6 into %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x1x64xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_5 : tensor<1x1x1x64xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_3 = tensor.insert_slice %5 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_3 : tensor<1x1x112x64xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_1 = tensor.insert_slice %4 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_1 : tensor<1x112x112x64xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %1) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : tensor<1x230x230x3xf32> to tensor<1x229x229x3xf32>
# CHECK-NEXT:       %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x112x112x64xf32>
# CHECK-NEXT:       %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %extracted_slice_0) -> (tensor<1x112x112x64xf32>) {
# CHECK-NEXT:         %4 = affine.apply #map(%arg5)
# CHECK-NEXT:         %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, %4, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : tensor<1x229x229x3xf32> to tensor<1x7x229x3xf32>
# CHECK-NEXT:         %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> to tensor<1x1x112x64xf32>
# CHECK-NEXT:         %5 = scf.for %arg7 = %c0 to %c112 step %c4 iter_args(%arg8 = %extracted_slice_2) -> (tensor<1x1x112x64xf32>) {
# CHECK-NEXT:           %6 = affine.apply #map(%arg7)
# CHECK-NEXT:           %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0, %6, 0] [1, 7, 13, 3] [1, 1, 1, 1] : tensor<1x7x229x3xf32> to tensor<1x7x13x3xf32>
# CHECK-NEXT:           %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> to tensor<1x1x4x64xf32>
# CHECK-NEXT:           %7 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %extracted_slice_5) -> (tensor<1x1x4x64xf32>) {
# CHECK-NEXT:             %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0, %arg9] [7, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x64xf32> to tensor<7x7x3x16xf32>
# CHECK-NEXT:             %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x64xf32> to tensor<1x1x4x16xf32>
# CHECK-NEXT:             %8 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %extracted_slice_8) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:               %extracted_slice_10 = tensor.extract_slice %extracted_slice_4[0, %arg11, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : tensor<1x7x13x3xf32> to tensor<1x1x13x3xf32>
# CHECK-NEXT:               %extracted_slice_11 = tensor.extract_slice %extracted_slice_7[%arg11, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : tensor<7x7x3x16xf32> to tensor<1x7x3x16xf32>
# CHECK-NEXT:               %9 = scf.for %arg13 = %c0 to %c7 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x4x16xf32>) {
# CHECK-NEXT:                 %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : tensor<1x1x13x3xf32> to tensor<1x1x7x3xf32>
# CHECK-NEXT:                 %extracted_slice_13 = tensor.extract_slice %extracted_slice_11[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : tensor<1x7x3x16xf32> to tensor<1x1x3x16xf32>
# CHECK-NEXT:                 %extracted_slice_14 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_15 = tensor.extract_slice %extracted_slice_13[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_16 = tensor.extract_slice %extracted_slice_14[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_17 = tensor.extract_slice %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %10 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_16, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_17 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_18 = tensor.insert_slice %10 into %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_19 = tensor.extract_slice %extracted_slice_14[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_20 = tensor.extract_slice %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %11 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_19, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_20 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_21 = tensor.insert_slice %11 into %inserted_slice_18[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_22 = tensor.extract_slice %extracted_slice_14[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_23 = tensor.extract_slice %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_22, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_23 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_24 = tensor.insert_slice %12 into %inserted_slice_21[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_25 = tensor.extract_slice %extracted_slice_14[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_26 = tensor.extract_slice %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %13 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_25, %extracted_slice_15 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_26 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_27 = tensor.insert_slice %13 into %inserted_slice_24[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_28 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_29 = tensor.extract_slice %extracted_slice_13[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_30 = tensor.extract_slice %extracted_slice_28[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_31 = tensor.extract_slice %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_30, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_31 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_32 = tensor.insert_slice %14 into %inserted_slice_27[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_33 = tensor.extract_slice %extracted_slice_28[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_34 = tensor.extract_slice %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %15 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_33, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_34 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_35 = tensor.insert_slice %15 into %inserted_slice_32[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_36 = tensor.extract_slice %extracted_slice_28[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_37 = tensor.extract_slice %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %16 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_36, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_37 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_38 = tensor.insert_slice %16 into %inserted_slice_35[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_39 = tensor.extract_slice %extracted_slice_28[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_40 = tensor.extract_slice %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %17 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_39, %extracted_slice_29 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_40 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_41 = tensor.insert_slice %17 into %inserted_slice_38[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_42 = tensor.extract_slice %extracted_slice_12[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : tensor<1x1x7x3xf32> to tensor<1x1x7x1xf32>
# CHECK-NEXT:                 %extracted_slice_43 = tensor.extract_slice %extracted_slice_13[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x3x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %extracted_slice_44 = tensor.extract_slice %extracted_slice_42[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_45 = tensor.extract_slice %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %18 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_44, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_45 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_46 = tensor.insert_slice %18 into %inserted_slice_41[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_47 = tensor.extract_slice %extracted_slice_42[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_48 = tensor.extract_slice %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %19 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_47, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_48 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_49 = tensor.insert_slice %19 into %inserted_slice_46[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_50 = tensor.extract_slice %extracted_slice_42[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_51 = tensor.extract_slice %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %20 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_50, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_51 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_52 = tensor.insert_slice %20 into %inserted_slice_49[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 %extracted_slice_53 = tensor.extract_slice %extracted_slice_42[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x7x1xf32> to tensor<1x1x1x1xf32>
# CHECK-NEXT:                 %extracted_slice_54 = tensor.extract_slice %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> to tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %21 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_53, %extracted_slice_43 : tensor<1x1x1x1xf32>, tensor<1x1x1x16xf32>) outs(%extracted_slice_54 : tensor<1x1x1x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                   %22 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                   %23 = arith.addf %out, %22 : f32
# CHECK-NEXT:                   linalg.yield %23 : f32
# CHECK-NEXT:                 } -> tensor<1x1x1x16xf32>
# CHECK-NEXT:                 %inserted_slice_55 = tensor.insert_slice %21 into %inserted_slice_52[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x16xf32> into tensor<1x1x4x16xf32>
# CHECK-NEXT:                 scf.yield %inserted_slice_55 : tensor<1x1x4x16xf32>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               scf.yield %9 : tensor<1x1x4x16xf32>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %inserted_slice_9 = tensor.insert_slice %8 into %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : tensor<1x1x4x16xf32> into tensor<1x1x4x64xf32>
# CHECK-NEXT:             scf.yield %inserted_slice_9 : tensor<1x1x4x64xf32>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %inserted_slice_6 = tensor.insert_slice %7 into %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : tensor<1x1x4x64xf32> into tensor<1x1x112x64xf32>
# CHECK-NEXT:           scf.yield %inserted_slice_6 : tensor<1x1x112x64xf32>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %inserted_slice_3 = tensor.insert_slice %5 into %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : tensor<1x1x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:         scf.yield %inserted_slice_3 : tensor<1x112x112x64xf32>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x112x112x64xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1x112x112x64xf32>, memref<1x112x112x64xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias}, %arg1: memref<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:     %c6 = arith.constant 6 : index
# CHECK-NEXT:     %c3 = arith.constant 3 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c7 = arith.constant 7 : index
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c64 = arith.constant 64 : index
# CHECK-NEXT:     %c112 = arith.constant 112 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x112x112x64xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %subview) -> (memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:         %subview_1 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         %3 = scf.for %arg7 = %c0 to %c112 step %c1 iter_args(%arg8 = %subview_1) -> (memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:           %subview_3 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           %4 = scf.for %arg9 = %c0 to %c64 step %c1 iter_args(%arg10 = %subview_3) -> (memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:             %subview_5 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_5 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>)
# CHECK-NEXT:             %subview_6 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             memref.copy %subview_5, %subview_6 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             scf.yield %arg10 : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %subview_4 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %4, %subview_4 : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %subview_2 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %3, %subview_2 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_0 : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %0) -> (memref<1x112x112x64xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : memref<1x230x230x3xf32> to memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       %2 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %subview_0) -> (memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:         %3 = affine.apply #map(%arg5)
# CHECK-NEXT:         %subview_2 = memref.subview %subview[0, %3, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:         %subview_3 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %c112 step %c4 iter_args(%arg8 = %subview_3) -> (memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:           %5 = affine.apply #map(%arg7)
# CHECK-NEXT:           %subview_5 = memref.subview %subview_2[0, 0, %5, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           %6 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %subview_6) -> (memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:             %subview_8 = memref.subview %arg1[0, 0, 0, %arg9] [7, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x64xf32> to memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:             %subview_9 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             %7 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %subview_9) -> (memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:               %subview_11 = memref.subview %subview_5[0, %arg11, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:               %subview_12 = memref.subview %subview_8[%arg11, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:               %8 = scf.for %arg13 = %c0 to %c7 step %c1 iter_args(%arg14 = %arg12) -> (memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) {
# CHECK-NEXT:                 %subview_13 = memref.subview %subview_11[0, 0, %arg13, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_14 = memref.subview %subview_12[0, %arg13, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_15 = memref.subview %subview_13[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_16 = memref.subview %subview_14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_17 = memref.subview %subview_15[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_18 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_17, %subview_16 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_18 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_19 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_18, %subview_19 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_20 = memref.subview %subview_15[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_21 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_20, %subview_16 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_21 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_22 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_21, %subview_22 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_23 = memref.subview %subview_15[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_24 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_23, %subview_16 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_24 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_25 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_24, %subview_25 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_26 = memref.subview %subview_15[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_27 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_26, %subview_16 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_27 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_28 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_27, %subview_28 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_29 = memref.subview %subview_13[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_30 = memref.subview %subview_14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_31 = memref.subview %subview_29[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_32 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_31, %subview_30 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_32 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_33 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_32, %subview_33 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_34 = memref.subview %subview_29[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_35 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_34, %subview_30 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_35 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_36 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_35, %subview_36 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_37 = memref.subview %subview_29[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_38 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_37, %subview_30 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_38 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_39 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_38, %subview_39 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_40 = memref.subview %subview_29[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_41 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_40, %subview_30 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_41 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_42 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_41, %subview_42 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_43 = memref.subview %subview_13[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_44 = memref.subview %subview_14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_45 = memref.subview %subview_43[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_46 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_45, %subview_44 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_46 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_47 = memref.subview %arg14[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_46, %subview_47 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_48 = memref.subview %subview_43[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_49 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_48, %subview_44 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_49 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_50 = memref.subview %arg14[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_49, %subview_50 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_51 = memref.subview %subview_43[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_52 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_51, %subview_44 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_52 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_53 = memref.subview %arg14[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_52, %subview_53 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 %subview_54 = memref.subview %subview_43[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                 %subview_55 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_54, %subview_44 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_55 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                 ^bb0(%in: f32, %in_57: f32, %out: f32):
# CHECK-NEXT:                   %9 = arith.mulf %in, %in_57 : f32
# CHECK-NEXT:                   %10 = arith.addf %out, %9 : f32
# CHECK-NEXT:                   linalg.yield %10 : f32
# CHECK-NEXT:                 }
# CHECK-NEXT:                 %subview_56 = memref.subview %arg14[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 memref.copy %subview_55, %subview_56 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                 scf.yield %arg14 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:               } {"./s"}
# CHECK-NEXT:               scf.yield %8 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             } {"./r"}
# CHECK-NEXT:             %subview_10 = memref.subview %arg10[0, 0, 0, %arg9] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             memref.copy %7, %subview_10 : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:             scf.yield %arg10 : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           } {"./f"}
# CHECK-NEXT:           %subview_7 = memref.subview %arg8[0, 0, %arg7, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           memref.copy %6, %subview_7 : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:           scf.yield %arg8 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         } {"./w"}
# CHECK-NEXT:         %subview_4 = memref.subview %arg6[0, %arg5, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         memref.copy %4, %subview_4 : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:         scf.yield %arg6 : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       } {"./h"}
# CHECK-NEXT:       %subview_1 = memref.subview %arg4[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       memref.copy %2, %subview_1 : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg4 : memref<1x112x112x64xf32>
# CHECK-NEXT:     } {"./b"}
# CHECK-NEXT:     memref.copy %1, %arg2 : memref<1x112x112x64xf32> to memref<1x112x112x64xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: conv2d_nhwc_r181
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:   - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %2 : 1x112x112x64xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'O'} : [1x230x230x3xfloat32, 7x7x3x64xfloat32] -> [1x112x112x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
