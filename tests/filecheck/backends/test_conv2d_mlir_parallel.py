# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.interchange(["b", "h", "w", "r", "s", "c", "f"])
sch.parallelize(["h","w"])
sch.unroll({"c":3})
sch.vectorize(["f"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_mlir_parallel",
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
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x2x2x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x2x2x16xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x8x8x3xf32>, memref<5x5x3x16xf32>) outs(%arg2 : memref<1x2x2x16xf32>) attrs =  {__xtc_id_conv_} {
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_conv_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./f" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_op_8, %forall_op_9 = transform.structured.tile_using_forall %tiled_op tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_9 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_op_8 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./c" : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %tiled_linalg_op_14 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.xtc.fold_unit_extent_dims_via_slices_for_vectorization
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match interface{LinalgOp} in %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%3) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.loop.unroll %loops_15 {factor = 3 : i64} : !transform.any_op
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
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x2x2x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0, 0, 0] [1, 2, 2, 16] [1, 1, 1, 1] : memref<1x2x2x16xf32> to memref<1x2x2x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c2 step %c1 {
# CHECK-NEXT:          %subview_0 = memref.subview %subview[0, %arg4, 0, 0] [1, 1, 2, 16] [1, 1, 1, 1] : memref<1x2x2x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<1x1x2x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c2 step %c1 {
# CHECK-NEXT:            %subview_1 = memref.subview %subview_0[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x2x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c16 step %c1 {
# CHECK-NEXT:              %subview_2 = memref.subview %subview_1[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%subview_2 : memref<1x1x1x1xf32, strided<[64, 32, 16, 1], offset: ?>>)
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 7, 7, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32> to memref<1x7x7x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0, 0, 0] [1, 2, 2, 16] [1, 1, 1, 1] : memref<1x2x2x16xf32> to memref<1x2x2x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.forall (%arg4) in (2) {
# CHECK-NEXT:          %1 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %1, 0, 0] [1, 5, 7, 3] [1, 1, 1, 1] : memref<1x7x7x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x5x7x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4, 0, 0] [1, 1, 2, 16] [1, 1, 1, 1] : memref<1x2x2x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<1x1x2x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.forall (%arg5) in (2) {
# CHECK-NEXT:            %2 = affine.apply #map(%arg5)
# CHECK-NEXT:            %subview_4 = memref.subview %subview_2[0, 0, %2, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x7x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_5 = memref.subview %subview_3[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x2x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c5 step %c1 {
# CHECK-NEXT:              %subview_6 = memref.subview %subview_4[0, %arg6, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x5x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_7 = memref.subview %subview_0[%arg6, 0, 0, 0] [1, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              scf.for %arg7 = %c0 to %c5 step %c1 {
# CHECK-NEXT:                %subview_8 = memref.subview %subview_6[0, 0, %arg7, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_9 = memref.subview %subview_7[0, %arg7, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x5x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_10 = memref.subview %subview_8[0, 0, 0, %c0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_11 = memref.subview %subview_9[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_12 = memref.subview %subview_10[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1xf32, strided<[192], offset: ?>>
# CHECK-NEXT:                %subview_13 = memref.subview %subview_11[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x16xf32, strided<[240, 1], offset: ?>>
# CHECK-NEXT:                %subview_14 = memref.subview %subview_5[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %3 = vector.transfer_read %subview_12[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[192], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                %4 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[240, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %5 = vector.transfer_read %subview_14[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %6 = vector.extract %4[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %7 = vector.extract %3[0] : f32 from vector<1xf32>
# CHECK-NEXT:                %8 = vector.broadcast %7 : f32 to vector<16xf32>
# CHECK-NEXT:                %9 = vector.fma %6, %8, %5 : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %9, %subview_14[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %subview_8[0, 0, 0, %c1] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_16 = memref.subview %subview_9[0, 0, %c1, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_17 = memref.subview %subview_15[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1xf32, strided<[192], offset: ?>>
# CHECK-NEXT:                %subview_18 = memref.subview %subview_16[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x16xf32, strided<[240, 1], offset: ?>>
# CHECK-NEXT:                %subview_19 = memref.subview %subview_5[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %10 = vector.transfer_read %subview_17[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[192], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                %11 = vector.transfer_read %subview_18[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[240, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %12 = vector.transfer_read %subview_19[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %13 = vector.extract %11[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %14 = vector.extract %10[0] : f32 from vector<1xf32>
# CHECK-NEXT:                %15 = vector.broadcast %14 : f32 to vector<16xf32>
# CHECK-NEXT:                %16 = vector.fma %13, %15, %12 : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %16, %subview_19[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %subview_20 = memref.subview %subview_8[0, 0, 0, %c2] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_21 = memref.subview %subview_9[0, 0, %c2, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_22 = memref.subview %subview_20[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1xf32, strided<[192], offset: ?>>
# CHECK-NEXT:                %subview_23 = memref.subview %subview_21[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x16xf32, strided<[240, 1], offset: ?>>
# CHECK-NEXT:                %subview_24 = memref.subview %subview_5[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[64, 32, 16, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:                %17 = vector.transfer_read %subview_22[%c0], %0 {in_bounds = [true]} : memref<1xf32, strided<[192], offset: ?>>, vector<1xf32>
# CHECK-NEXT:                %18 = vector.transfer_read %subview_23[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[240, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %19 = vector.transfer_read %subview_24[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:                %20 = vector.extract %18[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %21 = vector.extract %17[0] : f32 from vector<1xf32>
# CHECK-NEXT:                %22 = vector.broadcast %21 : f32 to vector<16xf32>
# CHECK-NEXT:                %23 = vector.fma %20, %22, %19 : vector<16xf32>
# CHECK-NEXT:                vector.transfer_write %23, %subview_24[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:              } {"./s"}
# CHECK-NEXT:            } {"./r"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
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
# CHECK-NEXT:    - %2 : 1x2x2x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'conv'} : [1x8x8x3xfloat32, 5x5x3x16xfloat32] -> [1x2x2x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
