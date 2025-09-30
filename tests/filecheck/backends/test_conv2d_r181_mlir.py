# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Resnet18_01 size
N, H, W, F, R, S, C, SH, SW, dtype = 1, 224, 224, 64, 7, 7, 3, 2, 2, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

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
# CHECK-NEXT:        %0 = arith.mulf %in, %in_0 : f32
# CHECK-NEXT:        %1 = arith.addf %out, %0 : f32
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
# CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_O_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_20) : (!transform.any_op) -> ()
# CHECK-NEXT:      %3 = transform.get_parent_op %loops_7 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %4 = transform.structured.match attributes {"./w1"} in %3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_21 {factor = 4 : i64} : !transform.any_op
# CHECK-NEXT:      %5 = transform.structured.match attributes {"./c"} in %3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_19 {factor = 3 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias}, %arg1: memref<7x7x3x64xf32> {llvm.noalias}, %arg2: memref<1x112x112x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c3 = arith.constant 3 : index
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
# CHECK-NEXT:          %0 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %0, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c112 step %c4 {
# CHECK-NEXT:            %1 = affine.apply #map(%arg5)
# CHECK-NEXT:            %subview_4 = memref.subview %subview_2[0, 0, %1, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
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
# CHECK-NEXT:                  %c3_12 = arith.constant 3 : index
# CHECK-NEXT:                  %subview_13 = memref.subview %subview_10[0, 0, 0, %c0] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_14 = memref.subview %subview_11[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %c4_15 = arith.constant 4 : index
# CHECK-NEXT:                  %2 = affine.apply #map(%c0)
# CHECK-NEXT:                  %subview_16 = memref.subview %subview_13[0, 0, %2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_17 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_16, %subview_14 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_17 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c1_18 = arith.constant 1 : index
# CHECK-NEXT:                  %3 = arith.muli %c1, %c1_18 : index
# CHECK-NEXT:                  %4 = arith.addi %c0, %3 : index
# CHECK-NEXT:                  %5 = affine.apply #map(%4)
# CHECK-NEXT:                  %subview_19 = memref.subview %subview_13[0, 0, %5, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_20 = memref.subview %subview_7[0, 0, %4, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_19, %subview_14 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_20 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c2 = arith.constant 2 : index
# CHECK-NEXT:                  %6 = arith.muli %c1, %c2 : index
# CHECK-NEXT:                  %7 = arith.addi %c0, %6 : index
# CHECK-NEXT:                  %8 = affine.apply #map(%7)
# CHECK-NEXT:                  %subview_21 = memref.subview %subview_13[0, 0, %8, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_22 = memref.subview %subview_7[0, 0, %7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_21, %subview_14 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_22 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c3_23 = arith.constant 3 : index
# CHECK-NEXT:                  %9 = arith.muli %c1, %c3_23 : index
# CHECK-NEXT:                  %10 = arith.addi %c0, %9 : index
# CHECK-NEXT:                  %11 = affine.apply #map(%10)
# CHECK-NEXT:                  %subview_24 = memref.subview %subview_13[0, 0, %11, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_25 = memref.subview %subview_7[0, 0, %10, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_24, %subview_14 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_25 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c1_26 = arith.constant 1 : index
# CHECK-NEXT:                  %12 = arith.muli %c1, %c1_26 : index
# CHECK-NEXT:                  %13 = arith.addi %c0, %12 : index
# CHECK-NEXT:                  %subview_27 = memref.subview %subview_10[0, 0, 0, %13] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_28 = memref.subview %subview_11[0, 0, %13, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %c4_29 = arith.constant 4 : index
# CHECK-NEXT:                  %14 = affine.apply #map(%c0)
# CHECK-NEXT:                  %subview_30 = memref.subview %subview_27[0, 0, %14, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_31 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_30, %subview_28 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_31 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c1_32 = arith.constant 1 : index
# CHECK-NEXT:                  %15 = arith.muli %c1, %c1_32 : index
# CHECK-NEXT:                  %16 = arith.addi %c0, %15 : index
# CHECK-NEXT:                  %17 = affine.apply #map(%16)
# CHECK-NEXT:                  %subview_33 = memref.subview %subview_27[0, 0, %17, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_34 = memref.subview %subview_7[0, 0, %16, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_33, %subview_28 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_34 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c2_35 = arith.constant 2 : index
# CHECK-NEXT:                  %18 = arith.muli %c1, %c2_35 : index
# CHECK-NEXT:                  %19 = arith.addi %c0, %18 : index
# CHECK-NEXT:                  %20 = affine.apply #map(%19)
# CHECK-NEXT:                  %subview_36 = memref.subview %subview_27[0, 0, %20, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_37 = memref.subview %subview_7[0, 0, %19, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_36, %subview_28 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_37 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c3_38 = arith.constant 3 : index
# CHECK-NEXT:                  %21 = arith.muli %c1, %c3_38 : index
# CHECK-NEXT:                  %22 = arith.addi %c0, %21 : index
# CHECK-NEXT:                  %23 = affine.apply #map(%22)
# CHECK-NEXT:                  %subview_39 = memref.subview %subview_27[0, 0, %23, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_40 = memref.subview %subview_7[0, 0, %22, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_39, %subview_28 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_40 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c2_41 = arith.constant 2 : index
# CHECK-NEXT:                  %24 = arith.muli %c1, %c2_41 : index
# CHECK-NEXT:                  %25 = arith.addi %c0, %24 : index
# CHECK-NEXT:                  %subview_42 = memref.subview %subview_10[0, 0, 0, %25] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_43 = memref.subview %subview_11[0, 0, %25, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %c4_44 = arith.constant 4 : index
# CHECK-NEXT:                  %26 = affine.apply #map(%c0)
# CHECK-NEXT:                  %subview_45 = memref.subview %subview_42[0, 0, %26, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_46 = memref.subview %subview_7[0, 0, %c0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_45, %subview_43 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_46 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c1_47 = arith.constant 1 : index
# CHECK-NEXT:                  %27 = arith.muli %c1, %c1_47 : index
# CHECK-NEXT:                  %28 = arith.addi %c0, %27 : index
# CHECK-NEXT:                  %29 = affine.apply #map(%28)
# CHECK-NEXT:                  %subview_48 = memref.subview %subview_42[0, 0, %29, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_49 = memref.subview %subview_7[0, 0, %28, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_48, %subview_43 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_49 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c2_50 = arith.constant 2 : index
# CHECK-NEXT:                  %30 = arith.muli %c1, %c2_50 : index
# CHECK-NEXT:                  %31 = arith.addi %c0, %30 : index
# CHECK-NEXT:                  %32 = affine.apply #map(%31)
# CHECK-NEXT:                  %subview_51 = memref.subview %subview_42[0, 0, %32, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_52 = memref.subview %subview_7[0, 0, %31, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_51, %subview_43 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_52 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
# CHECK-NEXT:                  %c3_53 = arith.constant 3 : index
# CHECK-NEXT:                  %33 = arith.muli %c1, %c3_53 : index
# CHECK-NEXT:                  %34 = arith.addi %c0, %33 : index
# CHECK-NEXT:                  %35 = affine.apply #map(%34)
# CHECK-NEXT:                  %subview_54 = memref.subview %subview_42[0, 0, %35, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_55 = memref.subview %subview_7[0, 0, %34, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_54, %subview_43 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_55 : memref<1x1x1x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                  ^bb0(%in: f32, %in_56: f32, %out: f32):
# CHECK-NEXT:                    %36 = arith.mulf %in, %in_56 : f32
# CHECK-NEXT:                    %37 = arith.addf %out, %36 : f32
# CHECK-NEXT:                    linalg.yield %37 : f32
# CHECK-NEXT:                  }
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
