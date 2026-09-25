# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 1024, 1024, 512, "float32"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler = sch,
    node_name = "C",
    abstract_dims = ["I","J","K"],
    spec = {
        "I": {"gpu_block": 0},
        "J": {"gpu_block": 1},
        "K": {},
        "I#128": {"gpu_warp": 0},
        "J#32": {},
        "I#32": {"gpu_lane": 0},
        "J#4": {},

    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_descript_mlir_gpu_warp_lane",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias, memref.on_device}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<1024x1024xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<1024x512xf32>, memref<512x1024xf32>) outs(%arg2 : memref<1024x1024xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [128, 32, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "C/I" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C/K" : !transform.any_op
# CHECK-NEXT:      %tiled_op_4, %forall_op_5 = transform.structured.tile_using_forall %tiled_linalg_op_2 tile_sizes [32, 0, 0](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_5 "C/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_op_4 tile_sizes [0, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/J0" : !transform.any_op
# CHECK-NEXT:      %tiled_op_8, %forall_op_9 = transform.structured.tile_using_forall %tiled_linalg_op_6 tile_sizes [1, 0, 0](mapping = [#gpu.lane<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_9 "C/I1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_op_8 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "C/J1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_nested_forall_to_threads %2 block_dims = [128, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT:  #map2 = affine_map<()[s0] -> (s0 floordiv 32)>
# CHECK-NEXT:  #map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 128 + s2 * 128)>
# CHECK-NEXT:  #map4 = affine_map<()[s0] -> (s0 mod 32)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias, memref.on_device}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c1024_5 = arith.constant 1024 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c1024_5 step %c1_6 {
# CHECK-NEXT:          %subview_7 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_7 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c8, %arg10 = %c32, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %c128, %arg13 = %c1_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %1 = affine.apply #map1(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%0, 0] [128, 512] [1, 1] : memref<1024x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_6 = memref.subview %arg1[0, %1] [512, 32] [1, 1] : memref<512x1024xf32> to memref<512x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg2[%0, %1] [128, 32] [1, 1] : memref<1024x1024xf32> to memref<128x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_8 = arith.constant 0 : index
# CHECK-NEXT:        %c512 = arith.constant 512 : index
# CHECK-NEXT:        %c1_9 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_8 to %c512 step %c1_9 {
# CHECK-NEXT:          %subview_10 = memref.subview %subview[0, %arg15] [128, 1] [1, 1] : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_11 = memref.subview %subview_6[%arg15, 0] [1, 32] [1, 1] : memref<512x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %subview_12 = memref.subview %subview_7[0, 0] [128, 32] [1, 1] : memref<128x32xf32, strided<[1024, 1], offset: ?>> to memref<128x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:          %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:          %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:          %2 = affine.apply #map2()[%thread_id_x]
# CHECK-NEXT:          %3 = affine.apply #map1(%2)
# CHECK-NEXT:          %subview_13 = memref.subview %subview_10[%3, 0] [32, 1] [1, 1] : memref<128x1xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_14 = memref.subview %subview_11[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %subview_15 = memref.subview %subview_12[%3, 0] [32, 32] [1, 1] : memref<128x32xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %c0_16 = arith.constant 0 : index
# CHECK-NEXT:          %c32_17 = arith.constant 32 : index
# CHECK-NEXT:          %c4 = arith.constant 4 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_16 to %c32_17 step %c4 {
# CHECK-NEXT:            %subview_18 = memref.subview %subview_13[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %subview_14[0, %arg16] [1, 4] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %subview_20 = memref.subview %subview_15[0, %arg16] [32, 4] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<32x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %thread_id_x_21 = gpu.thread_id  x
# CHECK-NEXT:            %thread_id_y_22 = gpu.thread_id  y
# CHECK-NEXT:            %thread_id_z_23 = gpu.thread_id  z
# CHECK-NEXT:            %4 = affine.apply #map3()[%thread_id_x_21, %c0_4, %c0_4]
# CHECK-NEXT:            %5 = affine.apply #map4()[%thread_id_x_21]
# CHECK-NEXT:            %6 = affine.apply #map4()[%thread_id_x_21]
# CHECK-NEXT:            %c32_24 = arith.constant 32 : index
# CHECK-NEXT:            %7 = arith.cmpi ult, %5, %c32_24 : index
# CHECK-NEXT:            scf.if %7 {
# CHECK-NEXT:              %subview_25 = memref.subview %subview_18[%6, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_26 = memref.subview %subview_19[0, 0] [1, 4] [1, 1] : memref<1x4xf32, strided<[1024, 1], offset: ?>> to memref<1x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %subview_27 = memref.subview %subview_20[%6, 0] [1, 4] [1, 1] : memref<32x4xf32, strided<[1024, 1], offset: ?>> to memref<1x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %c0_28 = arith.constant 0 : index
# CHECK-NEXT:              %c4_29 = arith.constant 4 : index
# CHECK-NEXT:              %c1_30 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg17 = %c0_28 to %c4_29 step %c1_30 {
# CHECK-NEXT:                %subview_31 = memref.subview %subview_25[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                %subview_32 = memref.subview %subview_26[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                %subview_33 = memref.subview %subview_27[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                linalg.matmul {__xtc_id_C_} ins(%subview_31, %subview_32 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[1024, 1], offset: ?>>) outs(%subview_33 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:              } {"C/J1"}
# CHECK-NEXT:            }
# CHECK-NEXT:            gpu.barrier
# CHECK-NEXT:          } {"C/J0"}
# CHECK-NEXT:          gpu.barrier
# CHECK-NEXT:        } {"C/K"}
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1024x512xfloat32
# CHECK-NEXT:    - %1 : 512x1024xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1024x1024xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [1024x512xfloat32, 512x1024xfloat32] -> [1024x1024xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
