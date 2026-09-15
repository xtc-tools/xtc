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
        "I#32": {"gpu_warp": 0},
        "J#32": {},
        "I#4": {"gpu_lane": 0},
        "J#4": {"gpu_lane": 1},

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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [32, 32, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "C/I" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C/K" : !transform.any_op
# CHECK-NEXT:      %tiled_op_4, %forall_op_5 = transform.structured.tile_using_forall %tiled_linalg_op_2 tile_sizes [4, 0, 0](mapping = [#gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_5 "C/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_op_4 tile_sizes [0, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/J0" : !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_nested_forall_to_threads %2 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias, memref.on_device}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c1024_7 = arith.constant 1024 : index
# CHECK-NEXT:        %c1_8 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_6 to %c1024_7 step %c1_8 {
# CHECK-NEXT:          %subview_9 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_9 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c32_1 = arith.constant 32 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c32_3 = arith.constant 32 : index
# CHECK-NEXT:      %c32_4 = arith.constant 32 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c32_3, %arg10 = %c32_4, %arg11 = %c1_5) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c32_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%0, 0] [32, 512] [1, 1] : memref<1024x512xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg1[0, %1] [512, 32] [1, 1] : memref<512x1024xf32> to memref<512x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_9 = memref.subview %arg2[%0, %1] [32, 32] [1, 1] : memref<1024x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_10 = arith.constant 0 : index
# CHECK-NEXT:        %c512 = arith.constant 512 : index
# CHECK-NEXT:        %c1_11 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_10 to %c512 step %c1_11 {
# CHECK-NEXT:          %subview_12 = memref.subview %subview[0, %arg15] [32, 1] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_13 = memref.subview %subview_8[%arg15, 0] [1, 32] [1, 1] : memref<512x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %subview_14 = memref.subview %subview_9[0, 0] [32, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:          %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:          %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:          %c8 = arith.constant 8 : index
# CHECK-NEXT:          %2 = arith.cmpi ult, %thread_id_x, %c8 : index
# CHECK-NEXT:          %c1_15 = arith.constant 1 : index
# CHECK-NEXT:          %3 = arith.cmpi ult, %thread_id_y, %c1_15 : index
# CHECK-NEXT:          %4 = arith.andi %2, %3 : i1
# CHECK-NEXT:          scf.if %4 {
# CHECK-NEXT:            %5 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:            %subview_16 = memref.subview %subview_12[%5, 0] [4, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_17 = memref.subview %subview_13[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %subview_18 = memref.subview %subview_14[%5, 0] [4, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<4x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %c0_19 = arith.constant 0 : index
# CHECK-NEXT:            %c32_20 = arith.constant 32 : index
# CHECK-NEXT:            %c4 = arith.constant 4 : index
# CHECK-NEXT:            scf.for %arg16 = %c0_19 to %c32_20 step %c4 {
# CHECK-NEXT:              %subview_21 = memref.subview %subview_16[0, 0] [4, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_22 = memref.subview %subview_17[0, %arg16] [1, 4] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %subview_23 = memref.subview %subview_18[0, %arg16] [4, 4] [1, 1] : memref<4x32xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_21, %subview_22 : memref<4x1xf32, strided<[512, 1], offset: ?>>, memref<1x4xf32, strided<[1024, 1], offset: ?>>) outs(%subview_23 : memref<4x4xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:            } {"C/J0"}
# CHECK-NEXT:          }
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
