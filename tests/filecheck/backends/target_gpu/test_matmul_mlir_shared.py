# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 256, 256, 256, "float32"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler(default_node="C")
sch.tile("i", {"i1": 64, "i2": 4})
sch.tile("j", {"j1": 64, "j2": 4})
sch.tile("k", {"k1": 32})
sch.pack_at("k", 0, "shared")
sch.pack_at("k", 1, "shared", alignment=4)
sch.gpu_block(["i", "j"])
sch.gpu_thread(["i1", "j1"])
sch.interchange(["i", "j", "i1", "j1", "k", "k1", "i2", "j2"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_gpu_shared_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg1: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<256x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<256x256xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<256x256xf32>, memref<256x256xf32>) outs(%arg2 : memref<256x256xf32>)
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [64, 64, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [4, 4, 0](mapping = [#gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./i1" : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %tiled_op_2 {
# CHECK-NEXT:        transform.apply_patterns.memref.fold_memref_alias_ops
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op_2 tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./j2" : !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_nested_forall_to_threads %2 block_dims = [16, 16, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.structured.match attributes {"./k"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.xtc.promote_shared_memory %3, %4 {input_idx = 0 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %6 = transform.xtc.promote_shared_memory %3, %4 {alignment = 4 : i64, input_idx = 1 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0 + d1 * 16)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map2 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg1: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<256x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %c0_14 = arith.constant 0 : index
# CHECK-NEXT:        %c256_15 = arith.constant 256 : index
# CHECK-NEXT:        %c1_16 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_14 to %c256_15 step %c1_16 {
# CHECK-NEXT:          %subview_17 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_17 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c16_1 = arith.constant 16 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c4_3 = arith.constant 4 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      %c0_5 = arith.constant 0 : index
# CHECK-NEXT:      %c1_6 = arith.constant 1 : index
# CHECK-NEXT:      %c256_7 = arith.constant 256 : index
# CHECK-NEXT:      %c2048 = arith.constant 2048 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c31 = arith.constant 31 : index
# CHECK-NEXT:      %c0_8 = arith.constant 0 : index
# CHECK-NEXT:      %c1_9 = arith.constant 1 : index
# CHECK-NEXT:      %c256_10 = arith.constant 256 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c16_11 = arith.constant 16 : index
# CHECK-NEXT:      %c4_12 = arith.constant 4 : index
# CHECK-NEXT:      %c15 = arith.constant 15 : index
# CHECK-NEXT:      %c4_13 = arith.constant 4 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c4, %arg10 = %c4_3, %arg11 = %c1_4) threads(%arg6, %arg7, %arg8) in (%arg12 = %c16, %arg13 = %c16_1, %arg14 = %c1_2) workgroup(%arg15 : memref<64x32xf32, 3 : i32>, %arg16 : memref<32x64xf32, 3 : i32>) {
# CHECK-NEXT:        %0 = affine.apply #map(%arg6, %arg7)
# CHECK-NEXT:        %1 = affine.apply #map(%arg6, %arg7)
# CHECK-NEXT:        %c0_14 = arith.constant 0 : index
# CHECK-NEXT:        %c0_15 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %2 = affine.apply #map1(%block_id_x)
# CHECK-NEXT:        %3 = affine.apply #map1(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%2, 0] [64, 256] [1, 1] : memref<256x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_16 = memref.subview %arg1[0, %3] [256, 64] [1, 1] : memref<256x256xf32> to memref<256x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_17 = memref.subview %arg2[%2, %3] [64, 64] [1, 1] : memref<256x256xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %4 = affine.apply #map2(%thread_id_x)
# CHECK-NEXT:        %5 = affine.apply #map2(%thread_id_y)
# CHECK-NEXT:        %subview_18 = memref.subview %subview[%4, 0] [4, 256] [1, 1] : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<4x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_19 = memref.subview %subview_16[0, %5] [256, 4] [1, 1] : memref<256x64xf32, strided<[256, 1], offset: ?>> to memref<256x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_20 = memref.subview %subview_17[%4, %5] [4, 4] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %c0_21 = arith.constant 0 : index
# CHECK-NEXT:        %c256_22 = arith.constant 256 : index
# CHECK-NEXT:        %c32_23 = arith.constant 32 : index
# CHECK-NEXT:        scf.for %arg17 = %c0_21 to %c256_22 step %c32_23 {
# CHECK-NEXT:          %subview_24 = memref.subview %subview_18[0, %arg17] [4, 32] [1, 1] : memref<4x256xf32, strided<[256, 1], offset: ?>> to memref<4x32xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          gpu.barrier
# CHECK-NEXT:          scf.for %arg18 = %1 to %c2048 step %c256_7 {
# CHECK-NEXT:            %7 = arith.shrui %arg18, %c5 : index
# CHECK-NEXT:            %8 = arith.andi %arg18, %c31 : index
# CHECK-NEXT:            %9 = arith.addi %arg17, %8 : index
# CHECK-NEXT:            %10 = nvgpu.device_async_copy %subview[%7, %9], %arg15[%7, %8], 1 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x32xf32, 3 : i32>
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_25 = memref.subview %subview_19[%arg17, 0] [32, 4] [1, 1] : memref<256x4xf32, strided<[256, 1], offset: ?>> to memref<32x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg18 = %0 to %c512 step %c256_10 {
# CHECK-NEXT:            %7 = arith.shrui %arg18, %c4_12 : index
# CHECK-NEXT:            %8 = arith.andi %arg18, %c15 : index
# CHECK-NEXT:            %9 = arith.muli %8, %c4_13 : index
# CHECK-NEXT:            %10 = arith.addi %arg17, %7 : index
# CHECK-NEXT:            %11 = nvgpu.device_async_copy %subview_16[%10, %9], %arg16[%7, %9], 4 : memref<256x64xf32, strided<[256, 1], offset: ?>> to memref<32x64xf32, 3 : i32>
# CHECK-NEXT:          }
# CHECK-NEXT:          %6 = nvgpu.device_async_create_group 
# CHECK-NEXT:          nvgpu.device_async_wait %6 {numGroups = 0 : i32}
# CHECK-NEXT:          gpu.barrier
# CHECK-NEXT:          %subview_26 = memref.subview %subview_20[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %c0_27 = arith.constant 0 : index
# CHECK-NEXT:          %c32_28 = arith.constant 32 : index
# CHECK-NEXT:          %c1_29 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg18 = %c0_27 to %c32_28 step %c1_29 {
# CHECK-NEXT:            %subview_30 = memref.subview %subview_24[0, %arg18] [4, 1] [1, 1] : memref<4x32xf32, strided<[256, 1], offset: ?>> to memref<4x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_31 = memref.subview %subview_25[%arg18, 0] [1, 4] [1, 1] : memref<32x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_32 = memref.subview %subview_26[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %c0_33 = arith.constant 0 : index
# CHECK-NEXT:            %c4_34 = arith.constant 4 : index
# CHECK-NEXT:            %c1_35 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg19 = %c0_33 to %c4_34 step %c1_35 {
# CHECK-NEXT:              %subview_36 = memref.subview %subview_30[%arg19, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_37 = memref.subview %subview_31[0, 0] [1, 4] [1, 1] : memref<1x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_38 = memref.subview %subview_32[%arg19, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %c0_39 = arith.constant 0 : index
# CHECK-NEXT:              %c4_40 = arith.constant 4 : index
# CHECK-NEXT:              %c1_41 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg20 = %c0_39 to %c4_40 step %c1_41 {
# CHECK-NEXT:                %7 = arith.addi %c0_5, %4 : index
# CHECK-NEXT:                %8 = arith.addi %c0_5, %arg18 : index
# CHECK-NEXT:                %9 = arith.addi %7, %arg19 : index
# CHECK-NEXT:                %subview_42 = memref.subview %arg15[%9, %8] [1, 1] [1, 1] : memref<64x32xf32, 3 : i32> to memref<1x1xf32, strided<[32, 1], offset: ?>, 3 : i32>
# CHECK-NEXT:                %10 = arith.addi %c0_8, %5 : index
# CHECK-NEXT:                %11 = arith.addi %c0_8, %arg18 : index
# CHECK-NEXT:                %12 = arith.addi %10, %arg20 : index
# CHECK-NEXT:                %subview_43 = memref.subview %arg16[%11, %12] [1, 1] [1, 1] : memref<32x64xf32, 3 : i32> to memref<1x1xf32, strided<[64, 1], offset: ?>, 3 : i32>
# CHECK-NEXT:                %subview_44 = memref.subview %subview_38[0, %arg20] [1, 1] [1, 1] : memref<1x4xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                linalg.matmul {__xtc_id_C_} ins(%subview_42, %subview_43 : memref<1x1xf32, strided<[32, 1], offset: ?>, 3 : i32>, memref<1x1xf32, strided<[64, 1], offset: ?>, 3 : i32>) outs(%subview_44 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:              } {"./j2"}
# CHECK-NEXT:            } {"./i2"}
# CHECK-NEXT:          } {"./k1"}
# CHECK-NEXT:        } {"./k"}
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 256x256xfloat32
# CHECK-NEXT:    - %1 : 256x256xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 256x256xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [256x256xfloat32, 256x256xfloat32] -> [256x256xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
