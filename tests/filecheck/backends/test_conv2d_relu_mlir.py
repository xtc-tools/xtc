# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="conv2d_nhwc_mini") as gb:
    c = O.conv2d(a, b, stride=(SH, SW), name="O")
    O.relu(c, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="relu")
sch.tile("i", {"i1": 16})
sch.interchange([ "i", "i1"])
sch.vectorize(["i1"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_relu_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map3 = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map4 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_mini(%arg0: memref<1x10x10x3xf32> {llvm.noalias}, %arg1: memref<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x8x8x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%alloca : memref<1x8x8x16xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x10x10x3xf32>, memref<3x3x3x16xf32>) outs(%alloca : memref<1x8x8x16xf32>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.mulf %in, %in_2 fastmath<fast> : f32
# CHECK-NEXT:        %1 = arith.addf %out, %0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %1 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2, 3]] : memref<1x8x8x16xf32> into memref<1024xf32>
# CHECK-NEXT:      %collapse_shape_0 = memref.collapse_shape %arg2 [[0, 1, 2, 3]] : memref<1x8x8x16xf32> into memref<1024xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel"]} ins(%collapse_shape, %cst_1 : memref<1024xf32>, f32) outs(%collapse_shape_0 : memref<1024xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
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
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./c" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %2 tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./i" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_20) : (!transform.any_op) -> ()
# CHECK-NEXT:      %3 = transform.get_parent_op %loops_21 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @conv2d_nhwc_mini(%arg0: memref<1x10x10x3xf32> {llvm.noalias}, %arg1: memref<3x3x3x16xf32> {llvm.noalias}, %arg2: memref<1x8x8x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<1x8x8x16xf32>
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %alloca[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x8x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c8 step %c1 {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %arg4, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c8 step %c1 {
# CHECK-NEXT:            %subview_3 = memref.subview %subview_2[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c16 step %c1 {
# CHECK-NEXT:              %subview_4 = memref.subview %subview_3[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_O_0_} ins(%cst_0 : f32) outs(%subview_4 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>)
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 10, 10, 3] [1, 1, 1, 1] : memref<1x10x10x3xf32> to memref<1x10x10x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[0, 0, 0, 0] [3, 3, 3, 16] [1, 1, 1, 1] : memref<3x3x3x16xf32> to memref<3x3x3x16xf32, strided<[144, 48, 16, 1]>>
# CHECK-NEXT:        %subview_3 = memref.subview %alloca[%arg3, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32> to memref<1x8x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c8 step %c1 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, %arg4, 0, 0] [1, 3, 10, 3] [1, 1, 1, 1] : memref<1x10x10x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview_3[0, %arg4, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x8x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c8 step %c1 {
# CHECK-NEXT:            %subview_6 = memref.subview %subview_4[0, 0, %arg5, 0] [1, 3, 3, 3] [1, 1, 1, 1] : memref<1x3x10x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_5[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x8x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c16 step %c1 {
# CHECK-NEXT:              %subview_8 = memref.subview %subview_2[0, 0, 0, %arg6] [3, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x16xf32, strided<[144, 48, 16, 1]>> to memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %subview_9 = memref.subview %subview_7[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1024, 128, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>
# CHECK-NEXT:              scf.for %arg7 = %c0 to %c3 step %c1 {
# CHECK-NEXT:                %subview_10 = memref.subview %subview_6[0, %arg7, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x3x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_11 = memref.subview %subview_8[%arg7, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : memref<3x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                scf.for %arg8 = %c0 to %c3 step %c1 {
# CHECK-NEXT:                  %subview_12 = memref.subview %subview_10[0, 0, %arg8, 0] [1, 1, 1, 3] [1, 1, 1, 1] : memref<1x1x3x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_13 = memref.subview %subview_11[0, %arg8, 0, 0] [1, 1, 3, 1] [1, 1, 1, 1] : memref<1x3x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                  scf.for %arg9 = %c0 to %c3 step %c1 {
# CHECK-NEXT:                    %subview_14 = memref.subview %subview_12[0, 0, 0, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x3xf32, strided<[300, 30, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>
# CHECK-NEXT:                    %subview_15 = memref.subview %subview_13[0, 0, %arg9, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x3x1xf32, strided<[144, 48, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_14, %subview_15 : memref<1x1x1x1xf32, strided<[300, 30, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[144, 48, 16, 1], offset: ?>>) outs(%subview_9 : memref<1x1x1x1xf32, strided<[1024, 128, 16, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                    ^bb0(%in: f32, %in_16: f32, %out: f32):
# CHECK-NEXT:                      %1 = arith.mulf %in, %in_16 fastmath<fast> : f32
# CHECK-NEXT:                      %2 = arith.addf %out, %1 fastmath<fast> : f32
# CHECK-NEXT:                      linalg.yield %2 : f32
# CHECK-NEXT:                    }
# CHECK-NEXT:                  } {"./c"}
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2, 3]] : memref<1x8x8x16xf32> into memref<1024xf32>
# CHECK-NEXT:      %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1, 2, 3]] : memref<1x8x8x16xf32> into memref<1024xf32>
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c16 {
# CHECK-NEXT:        %subview = memref.subview %collapse_shape[%arg3] [16] [1] : memref<1024xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %collapse_shape_1[%arg3] [16] [1] : memref<1024xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:        %2 = arith.maximumf %1, %cst : vector<16xf32>
# CHECK-NEXT:        vector.transfer_write %2, %subview_2[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'O'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [1x8x8x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
