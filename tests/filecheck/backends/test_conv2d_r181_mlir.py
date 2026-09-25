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

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias}, %arg1: memref<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%arg2 : memref<1x112x112x64xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x230x230x3xf32>, memref<7x7x3x64xf32>) outs(%arg2 : memref<1x112x112x64xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.mulf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        %1 = arith.addf %out, %0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %1 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_O_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 4, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./w1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %tiled_linalg_op_20 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.xtc.fold_unit_extent_dims_via_slices_for_vectorization
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match interface{LinalgOp} in %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%3) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.loop.unroll %loops_21 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_19 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_7 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias}, %arg1: memref<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c6 = arith.constant 6 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c7 = arith.constant 7 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c112 = arith.constant 112 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c112 step %c1 {
# CHECK-NEXT:          %subview_0 = memref.subview %subview[0, %arg4, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c112 step %c1 {
# CHECK-NEXT:            %subview_1 = memref.subview %subview_0[0, 0, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c64 step %c1 {
# CHECK-NEXT:              %subview_2 = memref.subview %subview_1[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_2 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>)
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : memref<1x230x230x3xf32> to memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0, 0, 0] [7, 7, 3, 64] [1, 1, 1, 1] : memref<7x7x3x64xf32> to memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c112 step %c1 {
# CHECK-NEXT:          %1 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %1, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c112 step %c4 {
# CHECK-NEXT:            %2 = affine.apply #map(%arg5)
# CHECK-NEXT:            %subview_4 = memref.subview %subview_2[0, 0, %2, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_5 = memref.subview %subview_3[0, 0, %arg5, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c64 step %c16 {
# CHECK-NEXT:              %subview_6 = memref.subview %subview_0[0, 0, 0, %arg6] [7, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>> to memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:              %subview_7 = memref.subview %subview_5[0, 0, 0, %arg6] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:              scf.for %arg7 = %c0 to %c7 step %c1 {
# CHECK-NEXT:                %subview_8 = memref.subview %subview_4[0, %arg7, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_9 = memref.subview %subview_6[%arg7, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                scf.for %arg8 = %c0 to %c7 step %c1 {
# CHECK-NEXT:                  %subview_10 = memref.subview %subview_8[0, 0, %arg8, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_11 = memref.subview %subview_9[0, %arg8, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_12 = memref.subview %subview_10[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_13 = memref.subview %subview_11[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_14 = memref.subview %subview_12[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_15 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_16 = memref.subview %subview_14[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_17 = memref.subview %subview_13[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_18 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %3 = vector.transfer_read %subview_16[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %4 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %5 = vector.transfer_read %subview_18[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %6 = vector.extract %4[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %7 = vector.extract %3[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %8 = vector.broadcast %7 : f32 to vector<16xf32>
# CHECK-NEXT:                  %9 = vector.fma %6, %8, %5 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %9, %subview_18[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_19 = memref.subview %subview_12[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_20 = memref.subview %subview_7[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_21 = memref.subview %subview_19[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_22 = memref.subview %subview_13[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_23 = memref.subview %subview_20[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %10 = vector.transfer_read %subview_21[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %11 = vector.transfer_read %subview_22[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %12 = vector.transfer_read %subview_23[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %13 = vector.extract %11[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %14 = vector.extract %10[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %15 = vector.broadcast %14 : f32 to vector<16xf32>
# CHECK-NEXT:                  %16 = vector.fma %13, %15, %12 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %16, %subview_23[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_24 = memref.subview %subview_12[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_25 = memref.subview %subview_7[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_26 = memref.subview %subview_24[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_27 = memref.subview %subview_13[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_28 = memref.subview %subview_25[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %17 = vector.transfer_read %subview_26[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %18 = vector.transfer_read %subview_27[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %19 = vector.transfer_read %subview_28[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %20 = vector.extract %18[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %21 = vector.extract %17[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %22 = vector.broadcast %21 : f32 to vector<16xf32>
# CHECK-NEXT:                  %23 = vector.fma %20, %22, %19 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %23, %subview_28[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_29 = memref.subview %subview_12[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_30 = memref.subview %subview_7[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_31 = memref.subview %subview_29[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_32 = memref.subview %subview_13[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_33 = memref.subview %subview_30[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %24 = vector.transfer_read %subview_31[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %25 = vector.transfer_read %subview_32[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %26 = vector.transfer_read %subview_33[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %27 = vector.extract %25[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %28 = vector.extract %24[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %29 = vector.broadcast %28 : f32 to vector<16xf32>
# CHECK-NEXT:                  %30 = vector.fma %27, %29, %26 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %30, %subview_33[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_34 = memref.subview %subview_10[0, 0, 0, %c1] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_35 = memref.subview %subview_11[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_36 = memref.subview %subview_34[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_37 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_38 = memref.subview %subview_36[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_39 = memref.subview %subview_35[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_40 = memref.subview %subview_37[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %31 = vector.transfer_read %subview_38[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %32 = vector.transfer_read %subview_39[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %33 = vector.transfer_read %subview_40[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %34 = vector.extract %32[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %35 = vector.extract %31[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %36 = vector.broadcast %35 : f32 to vector<16xf32>
# CHECK-NEXT:                  %37 = vector.fma %34, %36, %33 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %37, %subview_40[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_41 = memref.subview %subview_34[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_42 = memref.subview %subview_7[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_43 = memref.subview %subview_41[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_44 = memref.subview %subview_35[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_45 = memref.subview %subview_42[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %38 = vector.transfer_read %subview_43[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %39 = vector.transfer_read %subview_44[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %40 = vector.transfer_read %subview_45[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %41 = vector.extract %39[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %42 = vector.extract %38[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %43 = vector.broadcast %42 : f32 to vector<16xf32>
# CHECK-NEXT:                  %44 = vector.fma %41, %43, %40 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %44, %subview_45[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_46 = memref.subview %subview_34[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_47 = memref.subview %subview_7[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_48 = memref.subview %subview_46[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_49 = memref.subview %subview_35[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_50 = memref.subview %subview_47[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %45 = vector.transfer_read %subview_48[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %46 = vector.transfer_read %subview_49[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %47 = vector.transfer_read %subview_50[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %48 = vector.extract %46[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %49 = vector.extract %45[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %50 = vector.broadcast %49 : f32 to vector<16xf32>
# CHECK-NEXT:                  %51 = vector.fma %48, %50, %47 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %51, %subview_50[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_51 = memref.subview %subview_34[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_52 = memref.subview %subview_7[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_53 = memref.subview %subview_51[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_54 = memref.subview %subview_35[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_55 = memref.subview %subview_52[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %52 = vector.transfer_read %subview_53[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %53 = vector.transfer_read %subview_54[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %54 = vector.transfer_read %subview_55[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %55 = vector.extract %53[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %56 = vector.extract %52[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %57 = vector.broadcast %56 : f32 to vector<16xf32>
# CHECK-NEXT:                  %58 = vector.fma %55, %57, %54 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %58, %subview_55[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_56 = memref.subview %subview_10[0, 0, 0, %c2] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_57 = memref.subview %subview_11[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_58 = memref.subview %subview_56[0, 0, %c0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_59 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_60 = memref.subview %subview_58[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_61 = memref.subview %subview_57[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_62 = memref.subview %subview_59[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %59 = vector.transfer_read %subview_60[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %60 = vector.transfer_read %subview_61[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %61 = vector.transfer_read %subview_62[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %62 = vector.extract %60[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %63 = vector.extract %59[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %64 = vector.broadcast %63 : f32 to vector<16xf32>
# CHECK-NEXT:                  %65 = vector.fma %62, %64, %61 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %65, %subview_62[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_63 = memref.subview %subview_56[0, 0, %c2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_64 = memref.subview %subview_7[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_65 = memref.subview %subview_63[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_66 = memref.subview %subview_57[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_67 = memref.subview %subview_64[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %66 = vector.transfer_read %subview_65[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %67 = vector.transfer_read %subview_66[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %68 = vector.transfer_read %subview_67[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %69 = vector.extract %67[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %70 = vector.extract %66[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %71 = vector.broadcast %70 : f32 to vector<16xf32>
# CHECK-NEXT:                  %72 = vector.fma %69, %71, %68 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %72, %subview_67[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_68 = memref.subview %subview_56[0, 0, %c4, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_69 = memref.subview %subview_7[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_70 = memref.subview %subview_68[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_71 = memref.subview %subview_57[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_72 = memref.subview %subview_69[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %73 = vector.transfer_read %subview_70[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %74 = vector.transfer_read %subview_71[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %75 = vector.transfer_read %subview_72[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %76 = vector.extract %74[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %77 = vector.extract %73[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %78 = vector.broadcast %77 : f32 to vector<16xf32>
# CHECK-NEXT:                  %79 = vector.fma %76, %78, %75 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %79, %subview_72[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %subview_73 = memref.subview %subview_56[0, 0, %c6, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_74 = memref.subview %subview_7[0, 0, %c3, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_75 = memref.subview %subview_73[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1xf32, strided<[158700], offset: ?>>
# CHECK-NEXT:                  %subview_76 = memref.subview %subview_57[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x16xf32, strided<[1344, 1], offset: ?>>
# CHECK-NEXT:                  %subview_77 = memref.subview %subview_74[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                  %80 = vector.transfer_read %subview_75[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[158700], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                  %81 = vector.transfer_read %subview_76[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[1344, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                  %82 = vector.transfer_read %subview_77[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                  %83 = vector.extract %81[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                  %84 = vector.extract %80[0] : f32 from vector<1xf32>
# CHECK-NEXT:                  %85 = vector.broadcast %84 : f32 to vector<16xf32>
# CHECK-NEXT:                  %86 = vector.fma %83, %85, %82 : vector<16xf32>
# CHECK-NEXT:                  vector.transfer_write %86, %subview_77[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv2d_nhwc_r181
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:    - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x112x112x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'O'} : [1x230x230x3xfloat32, 7x7x3x64xfloat32] -> [1x112x112x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
