# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axis=(1, 2), name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_nhwc_mini_mlir",
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
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_pad_0_} ins(%cst : f32) outs(%alloca : memref<1x12x12x3xf32>)
# CHECK-NEXT:      %subview = memref.subview %alloca[0, 2, 2, 0] [1, 8, 8, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: 78>>
# CHECK-NEXT:      linalg.copy {__xtc_id_pad_} ins(%arg0 : memref<1x8x8x3xf32>) outs(%subview : memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: 78>>)
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst_0 : f32) outs(%arg2 : memref<1x4x4x16xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%alloca, %arg1 : memref<1x12x12x3xf32>, memref<5x5x3x16xf32>) outs(%arg2 : memref<1x4x4x16xf32>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.mulf %in, %in_1 : f32
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_pad_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./d1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./d2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./d3" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./d1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./d2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./d3" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_conv_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./f" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_conv_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_22, %loops_23 = transform.structured.tile_using_for %3 tile_sizes [1, 0, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_23 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_24, %loops_25 = transform.structured.tile_using_for %tiled_linalg_op_22 tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_25 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_26, %loops_27 = transform.structured.tile_using_for %tiled_linalg_op_24 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_27 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_28, %loops_29 = transform.structured.tile_using_for %tiled_linalg_op_26 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_29 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_30, %loops_31 = transform.structured.tile_using_for %tiled_linalg_op_28 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_31 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_32, %loops_33 = transform.structured.tile_using_for %tiled_linalg_op_30 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_33 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_34, %loops_35 = transform.structured.tile_using_for %tiled_linalg_op_32 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_35 "./c" : !transform.any_op
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
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1_0 {
# CHECK-NEXT:        %subview_11 = memref.subview %alloca[%arg3, 0, 0, 0] [1, 12, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x12x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:        %c0_12 = arith.constant 0 : index
# CHECK-NEXT:        %c12 = arith.constant 12 : index
# CHECK-NEXT:        %c1_13 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_12 to %c12 step %c1_13 {
# CHECK-NEXT:          %subview_14 = memref.subview %subview_11[0, %arg4, 0, 0] [1, 1, 12, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %c0_15 = arith.constant 0 : index
# CHECK-NEXT:          %c12_16 = arith.constant 12 : index
# CHECK-NEXT:          %c1_17 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_15 to %c12_16 step %c1_17 {
# CHECK-NEXT:            %subview_18 = memref.subview %subview_14[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x12x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %c0_19 = arith.constant 0 : index
# CHECK-NEXT:            %c3 = arith.constant 3 : index
# CHECK-NEXT:            %c1_20 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_19 to %c3 step %c1_20 {
# CHECK-NEXT:              %subview_21 = memref.subview %subview_18[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_pad_0_} ins(%cst : f32) outs(%subview_21 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>)
# CHECK-NEXT:            } {"./d3"}
# CHECK-NEXT:          } {"./d2"}
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %subview = memref.subview %alloca[0, 2, 2, 0] [1, 8, 8, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: 78>>
# CHECK-NEXT:      %c0_1 = arith.constant 0 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_1 to %c1_2 step %c1_3 {
# CHECK-NEXT:        %subview_11 = memref.subview %arg0[%arg3, 0, 0, 0] [1, 8, 8, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32> to memref<1x8x8x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_12 = memref.subview %subview[%arg3, 0, 0, 0] [1, 8, 8, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: 78>> to memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:        %c0_13 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_14 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_13 to %c8 step %c1_14 {
# CHECK-NEXT:          %subview_15 = memref.subview %subview_11[0, %arg4, 0, 0] [1, 1, 8, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x8x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_16 = memref.subview %subview_12[0, %arg4, 0, 0] [1, 1, 8, 3] [1, 1, 1, 1] : memref<1x8x8x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x8x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %c0_17 = arith.constant 0 : index
# CHECK-NEXT:          %c8_18 = arith.constant 8 : index
# CHECK-NEXT:          %c1_19 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_17 to %c8_18 step %c1_19 {
# CHECK-NEXT:            %subview_20 = memref.subview %subview_15[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x8x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_21 = memref.subview %subview_16[0, 0, %arg5, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x8x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %c0_22 = arith.constant 0 : index
# CHECK-NEXT:            %c3 = arith.constant 3 : index
# CHECK-NEXT:            %c1_23 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_22 to %c3 step %c1_23 {
# CHECK-NEXT:              %subview_24 = memref.subview %subview_20[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[192, 24, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_25 = memref.subview %subview_21[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              linalg.copy {__xtc_id_pad_} ins(%subview_24 : memref<1x1x1x1xf32, strided<[192, 24, 3, 1], offset: ?>>) outs(%subview_25 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>)
# CHECK-NEXT:            } {"./d3"}
# CHECK-NEXT:          } {"./d2"}
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0_5 = arith.constant 0 : index
# CHECK-NEXT:      %c1_6 = arith.constant 1 : index
# CHECK-NEXT:      %c1_7 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_5 to %c1_6 step %c1_7 {
# CHECK-NEXT:        %subview_11 = memref.subview %arg2[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x4x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %c0_12 = arith.constant 0 : index
# CHECK-NEXT:        %c4 = arith.constant 4 : index
# CHECK-NEXT:        %c1_13 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_12 to %c4 step %c1_13 {
# CHECK-NEXT:          %subview_14 = memref.subview %subview_11[0, %arg4, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %c0_15 = arith.constant 0 : index
# CHECK-NEXT:          %c4_16 = arith.constant 4 : index
# CHECK-NEXT:          %c1_17 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_15 to %c4_16 step %c1_17 {
# CHECK-NEXT:            %subview_18 = memref.subview %subview_14[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            %c0_19 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_20 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_19 to %c16 step %c1_20 {
# CHECK-NEXT:              %subview_21 = memref.subview %subview_18[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_conv_0_} ins(%cst_4 : f32) outs(%subview_21 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>)
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %c0_8 = arith.constant 0 : index
# CHECK-NEXT:      %c1_9 = arith.constant 1 : index
# CHECK-NEXT:      %c1_10 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_8 to %c1_9 step %c1_10 {
# CHECK-NEXT:        %subview_11 = memref.subview %alloca[%arg3, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x11x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_12 = memref.subview %arg1[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:        %subview_13 = memref.subview %arg2[%arg3, 0, 0, 0] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x4x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %c0_14 = arith.constant 0 : index
# CHECK-NEXT:        %c4 = arith.constant 4 : index
# CHECK-NEXT:        %c1_15 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_14 to %c4 step %c1_15 {
# CHECK-NEXT:          %0 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_16 = memref.subview %subview_11[0, %0, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : memref<1x11x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_17 = memref.subview %subview_12[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:          %subview_18 = memref.subview %subview_13[0, %arg4, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %c0_19 = arith.constant 0 : index
# CHECK-NEXT:          %c4_20 = arith.constant 4 : index
# CHECK-NEXT:          %c1_21 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_19 to %c4_20 step %c1_21 {
# CHECK-NEXT:            %1 = affine.apply #map(%arg5)
# CHECK-NEXT:            %subview_22 = memref.subview %subview_16[0, 0, %1, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_23 = memref.subview %subview_17[0, 0, 0, 0] [5, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:            %subview_24 = memref.subview %subview_18[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            %c0_25 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_26 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_25 to %c16 step %c1_26 {
# CHECK-NEXT:              %subview_27 = memref.subview %subview_22[0, 0, 0, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_28 = memref.subview %subview_23[0, 0, 0, %arg6] [5, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %subview_29 = memref.subview %subview_24[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              %c0_30 = arith.constant 0 : index
# CHECK-NEXT:              %c5 = arith.constant 5 : index
# CHECK-NEXT:              %c1_31 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg7 = %c0_30 to %c5 step %c1_31 {
# CHECK-NEXT:                %subview_32 = memref.subview %subview_27[0, %arg7, 0, 0] [1, 1, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_33 = memref.subview %subview_28[%arg7, 0, 0, 0] [1, 5, 3, 1] [1, 1, 1, 1] : memref<5x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_34 = memref.subview %subview_29[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                %c0_35 = arith.constant 0 : index
# CHECK-NEXT:                %c5_36 = arith.constant 5 : index
# CHECK-NEXT:                %c1_37 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg8 = %c0_35 to %c5_36 step %c1_37 {
# CHECK-NEXT:                  %subview_38 = memref.subview %subview_32[0, 0, %arg8, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_39 = memref.subview %subview_33[0, %arg8, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x5x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                  %subview_40 = memref.subview %subview_34[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                  %c0_41 = arith.constant 0 : index
# CHECK-NEXT:                  %c3 = arith.constant 3 : index
# CHECK-NEXT:                  %c1_42 = arith.constant 1 : index
# CHECK-NEXT:                  scf.for %arg9 = %c0_41 to %c3 step %c1_42 {
# CHECK-NEXT:                    %subview_43 = memref.subview %subview_38[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:                    %subview_44 = memref.subview %subview_39[0, 0, %arg9, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                    %subview_45 = memref.subview %subview_40[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_43, %subview_44 : memref<1x1x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_45 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                    ^bb0(%in: f32, %in_46: f32, %out: f32):
# CHECK-NEXT:                      %2 = arith.mulf %in, %in_46 : f32
# CHECK-NEXT:                      %3 = arith.addf %out, %2 : f32
# CHECK-NEXT:                      linalg.yield %3 : f32
# CHECK-NEXT:                    }
# CHECK-NEXT:                  } {"./c"}
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
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:    - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding=(2, 2, 2, 2), axis=(1, 2), constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
