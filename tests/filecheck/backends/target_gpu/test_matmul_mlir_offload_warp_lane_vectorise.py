# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.gpu import GPUDevice

# Create device
gpu = GPUDevice()

I, J, K, dtype = 1024, 1024, 512, "float32"
a = O.tensor((I, K), dtype, name="A", device=gpu) # A lives on the host
b = O.tensor((K, J), dtype, name="B", device=gpu) # B lives on the accelerator

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu) # C must live on the accelerator

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 8, "i2": 4})
sch.tile("j", {"j1": 128, "j2": 64, "j3": 4})
sch.tile("k", {"k2": 16})
# sch.unroll({"i2": 2})
sch.gpu_block(["j", "i"])
sch.gpu_warp(["j1"])
sch.gpu_lane(["j2", "i1"])
sch.interchange(["j", "i", "j1", "j2", "i1","k", "j3","i2", "k2"])
sch.vectorize(["j3","i2","k2"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="gpu_matmul_mlir_offload_tensor_vectorise",
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [8, 128, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [0, 64, 0](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./j1" : !transform.any_op
# CHECK-NEXT:      %tiled_op_4, %forall_op_5 = transform.structured.tile_using_forall %tiled_op_2 tile_sizes [4, 4, 0](mapping = [#gpu.lane<linear_dim_0>, #gpu.lane<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_5 "./j2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_op_4 tile_sizes [0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_6) : (!transform.any_op) -> ()
# CHECK-NEXT:      %2 = transform.get_parent_op %forall_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_nested_forall_to_threads %3 block_dims = [64, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 8)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map2 = affine_map<()[s0] -> (s0 floordiv 32)>
# CHECK-NEXT:  #map3 = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map4 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 64)>
# CHECK-NEXT:  #map5 = affine_map<()[s0] -> (s0 mod 32)>
# CHECK-NEXT:  #map6 = affine_map<()[s0] -> (s0 mod 2)>
# CHECK-NEXT:  #map7 = affine_map<()[s0] -> ((s0 mod 32) floordiv 2)>
# CHECK-NEXT:  #map8 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  #map9 = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map10 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map11 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias, memref.on_device}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c1024 step %c1 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_4 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c128, %arg10 = %c8, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %c64, %arg13 = %c1_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %2 = affine.apply #map1(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%1, 0] [8, 512] [1, 1] : memref<1024x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_6 = memref.subview %arg1[0, %2] [512, 128] [1, 1] : memref<512x1024xf32> to memref<512x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg2[%1, %2] [8, 128] [1, 1] : memref<1024x1024xf32> to memref<8x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %3 = affine.apply #map2()[%thread_id_x]
# CHECK-NEXT:        %4 = affine.apply #map3(%3)
# CHECK-NEXT:        %subview_8 = memref.subview %subview_6[0, %4] [512, 64] [1, 1] : memref<512x128xf32, strided<[1024, 1], offset: ?>> to memref<512x64xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_9 = memref.subview %subview_7[0, %4] [8, 64] [1, 1] : memref<8x128xf32, strided<[1024, 1], offset: ?>> to memref<8x64xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x_10 = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y_11 = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z_12 = gpu.thread_id  z
# CHECK-NEXT:        %5 = affine.apply #map4()[%thread_id_x_10, %c0_4, %c0_4]
# CHECK-NEXT:        %6 = affine.apply #map5()[%thread_id_x_10]
# CHECK-NEXT:        %7 = affine.apply #map6()[%thread_id_x_10]
# CHECK-NEXT:        %8 = affine.apply #map7()[%thread_id_x_10]
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %9 = arith.cmpi ult, %6, %c32 : index
# CHECK-NEXT:        scf.if %9 {
# CHECK-NEXT:          %10 = affine.apply #map8(%7)
# CHECK-NEXT:          %11 = affine.apply #map8(%8)
# CHECK-NEXT:          %subview_13 = memref.subview %subview[%10, 0] [4, 512] [1, 1] : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<4x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_14 = memref.subview %subview_8[0, %11] [512, 4] [1, 1] : memref<512x64xf32, strided<[1024, 1], offset: ?>> to memref<512x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %subview_15 = memref.subview %subview_9[%10, %11] [4, 4] [1, 1] : memref<8x64xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg15 = %c0 to %c512 step %c16 {
# CHECK-NEXT:            %subview_16 = memref.subview %subview_13[0, %arg15] [4, 16] [1, 1] : memref<4x512xf32, strided<[512, 1], offset: ?>> to memref<4x16xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_17 = memref.subview %subview_14[%arg15, 0] [16, 4] [1, 1] : memref<512x4xf32, strided<[1024, 1], offset: ?>> to memref<16x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %12 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x16xf32, strided<[512, 1], offset: ?>>, vector<4x16xf32>
# CHECK-NEXT:            %13 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<16x4xf32, strided<[1024, 1], offset: ?>>, vector<16x4xf32>
# CHECK-NEXT:            %14 = vector.transfer_read %subview_15[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x4xf32, strided<[1024, 1], offset: ?>>, vector<4x4xf32>
# CHECK-NEXT:            %15 = vector.contract {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %13, %14 : vector<4x16xf32>, vector<16x4xf32> into vector<4x4xf32>
# CHECK-NEXT:            vector.transfer_write %15, %subview_15[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:        }
# CHECK-NEXT:        gpu.barrier
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
# CHECK-NEXT:    - %0 : 1024x512xfloat32
# CHECK-NEXT:    - %1 : 512x1024xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1024x1024xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [1024x512xfloat32, 512x1024xfloat32] -> [1024x1024xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
