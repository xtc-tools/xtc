# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.artifacts import get_operation
from xtc.artifacts import get_operation

from xtc.runtimes.accelerator.gpu import GPUDevice

# Create device
gpu = GPUDevice()
op = get_operation("conv2d", "ResNet18_01")
N, H, W, F, R, S, C = [op["dims"][k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
SH, SW = [op["params"][k] for k in ["SH", "SW"]]
dtype = "float32"

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, device=gpu)
b = O.tensor((R, S, C, F), dtype, device=gpu)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.gpu_block(["b"])
sch.gpu_thread(["w1", "f1"])
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_gpu_mlir",
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
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias, memref.on_device}, %arg1: memref<7x7x3x64xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1x112x112x64xf32> {llvm.noalias, memref.on_device}) {
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [1, 0, 0, 0, 0, 0, 0](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_op tile_sizes [0, 1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./h" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 0, 4, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./w" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 0, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./f" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 0, 0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./r" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 0, 0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./s" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 0, 0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./c" : !transform.any_op
# CHECK-NEXT:      %tiled_op_18, %forall_op_19 = transform.structured.tile_using_forall %tiled_linalg_op_16 tile_sizes [0, 0, 1, 1, 0, 0, 0](mapping = [#gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_19 "./w1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_nested_forall_to_threads %2 block_dims = [4, 16, 1] : (!transform.any_op) -> !transform.any_op
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
# CHECK-NEXT:    func.func @conv2d_nhwc_r181(%arg0: memref<1x230x230x3xf32> {llvm.noalias, memref.on_device}, %arg1: memref<7x7x3x64xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1x112x112x64xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1 step %c1_0 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c112 = arith.constant 112 : index
# CHECK-NEXT:        %c1_7 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_6 to %c112 step %c1_7 {
# CHECK-NEXT:          %subview_8 = memref.subview %subview[0, %arg4, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          %c0_9 = arith.constant 0 : index
# CHECK-NEXT:          %c112_10 = arith.constant 112 : index
# CHECK-NEXT:          %c1_11 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_9 to %c112_10 step %c1_11 {
# CHECK-NEXT:            %subview_12 = memref.subview %subview_8[0, 0, %arg5, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c64 = arith.constant 64 : index
# CHECK-NEXT:            %c1_14 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_13 to %c64 step %c1_14 {
# CHECK-NEXT:              %subview_15 = memref.subview %subview_12[0, 0, 0, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:              linalg.fill {__xtc_id_O_0_} ins(%cst : f32) outs(%subview_15 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>)
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_3, %arg10 = %c1_4, %arg11 = %c1_5) threads(%arg6, %arg7, %arg8) in (%arg12 = %c4, %arg13 = %c16, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %subview = memref.subview %arg0[%block_id_x, 0, 0, 0] [1, 229, 229, 3] [1, 1, 1, 1] : memref<1x230x230x3xf32> to memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg1[0, 0, 0, 0] [7, 7, 3, 64] [1, 1, 1, 1] : memref<7x7x3x64xf32> to memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>>
# CHECK-NEXT:        %subview_9 = memref.subview %arg2[%block_id_x, 0, 0, 0] [1, 112, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32> to memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:        %c0_10 = arith.constant 0 : index
# CHECK-NEXT:        %c112 = arith.constant 112 : index
# CHECK-NEXT:        %c1_11 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_10 to %c112 step %c1_11 {
# CHECK-NEXT:          %0 = affine.apply #map(%arg15)
# CHECK-NEXT:          %subview_12 = memref.subview %subview[0, %0, 0, 0] [1, 7, 229, 3] [1, 1, 1, 1] : memref<1x229x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_13 = memref.subview %subview_8[0, 0, 0, 0] [7, 7, 3, 64] [1, 1, 1, 1] : memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>> to memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>>
# CHECK-NEXT:          %subview_14 = memref.subview %subview_9[0, %arg15, 0, 0] [1, 1, 112, 64] [1, 1, 1, 1] : memref<1x112x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:          %c0_15 = arith.constant 0 : index
# CHECK-NEXT:          %c112_16 = arith.constant 112 : index
# CHECK-NEXT:          %c4_17 = arith.constant 4 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_15 to %c112_16 step %c4_17 {
# CHECK-NEXT:            %1 = affine.apply #map(%arg16)
# CHECK-NEXT:            %subview_18 = memref.subview %subview_12[0, 0, %1, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x229x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %subview_13[0, 0, 0, 0] [7, 7, 3, 64] [1, 1, 1, 1] : memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>> to memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>>
# CHECK-NEXT:            %subview_20 = memref.subview %subview_14[0, 0, %arg16, 0] [1, 1, 4, 64] [1, 1, 1, 1] : memref<1x1x112x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:            %c0_21 = arith.constant 0 : index
# CHECK-NEXT:            %c64 = arith.constant 64 : index
# CHECK-NEXT:            %c16_22 = arith.constant 16 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_21 to %c64 step %c16_22 {
# CHECK-NEXT:              %subview_23 = memref.subview %subview_18[0, 0, 0, 0] [1, 7, 13, 3] [1, 1, 1, 1] : memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_24 = memref.subview %subview_19[0, 0, 0, %arg17] [7, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x64xf32, strided<[1344, 192, 64, 1]>> to memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:              %subview_25 = memref.subview %subview_20[0, 0, 0, %arg17] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x64xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:              %c0_26 = arith.constant 0 : index
# CHECK-NEXT:              %c7 = arith.constant 7 : index
# CHECK-NEXT:              %c1_27 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg18 = %c0_26 to %c7 step %c1_27 {
# CHECK-NEXT:                %subview_28 = memref.subview %subview_23[0, %arg18, 0, 0] [1, 1, 13, 3] [1, 1, 1, 1] : memref<1x7x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                %subview_29 = memref.subview %subview_24[%arg18, 0, 0, 0] [1, 7, 3, 16] [1, 1, 1, 1] : memref<7x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                %subview_30 = memref.subview %subview_25[0, 0, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                %c0_31 = arith.constant 0 : index
# CHECK-NEXT:                %c7_32 = arith.constant 7 : index
# CHECK-NEXT:                %c1_33 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg19 = %c0_31 to %c7_32 step %c1_33 {
# CHECK-NEXT:                  %subview_34 = memref.subview %subview_28[0, 0, %arg19, 0] [1, 1, 7, 3] [1, 1, 1, 1] : memref<1x1x13x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                  %subview_35 = memref.subview %subview_29[0, %arg19, 0, 0] [1, 1, 3, 16] [1, 1, 1, 1] : memref<1x7x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_36 = memref.subview %subview_30[0, 0, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                  %c0_37 = arith.constant 0 : index
# CHECK-NEXT:                  %c3 = arith.constant 3 : index
# CHECK-NEXT:                  %c1_38 = arith.constant 1 : index
# CHECK-NEXT:                  scf.for %arg20 = %c0_37 to %c3 step %c1_38 {
# CHECK-NEXT:                    %subview_39 = memref.subview %subview_34[0, 0, 0, %arg20] [1, 1, 7, 1] [1, 1, 1, 1] : memref<1x1x7x3xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                    %subview_40 = memref.subview %subview_35[0, 0, %arg20, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x3x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                    %subview_41 = memref.subview %subview_36[0, 0, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                    %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:                    %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:                    %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:                    %2 = affine.apply #map(%thread_id_x)
# CHECK-NEXT:                    %subview_42 = memref.subview %subview_39[0, 0, %2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x7x1xf32, strided<[158700, 690, 3, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>
# CHECK-NEXT:                    %subview_43 = memref.subview %subview_40[0, 0, 0, %thread_id_y] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[1344, 192, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[1344, 192, 64, 1], offset: ?>>
# CHECK-NEXT:                    %subview_44 = memref.subview %subview_41[0, 0, %thread_id_x, %thread_id_y] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[802816, 7168, 64, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>
# CHECK-NEXT:                    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_42, %subview_43 : memref<1x1x1x1xf32, strided<[158700, 690, 3, 1], offset: ?>>, memref<1x1x1x1xf32, strided<[1344, 192, 64, 1], offset: ?>>) outs(%subview_44 : memref<1x1x1x1xf32, strided<[802816, 7168, 64, 1], offset: ?>>) attrs =  {__xtc_id_O_} {
# CHECK-NEXT:                    ^bb0(%in: f32, %in_45: f32, %out: f32):
# CHECK-NEXT:                      %3 = arith.mulf %in, %in_45 fastmath<fast> : f32
# CHECK-NEXT:                      %4 = arith.addf %out, %3 fastmath<fast> : f32
# CHECK-NEXT:                      linalg.yield %4 : f32
# CHECK-NEXT:                    }
# CHECK-NEXT:                    gpu.barrier
# CHECK-NEXT:                  } {"./c"}
# CHECK-NEXT:                } {"./s"}
# CHECK-NEXT:              } {"./r"}
# CHECK-NEXT:            } {"./f"}
# CHECK-NEXT:          } {"./w"}
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
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
