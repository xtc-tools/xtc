# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.gpu import GPUDevice

# Create device
gpu = GPUDevice()

I, J, K, dtype = 512, 512, 512, "float32"
a = O.tensor((I, K), dtype, name="A") # A lives on the host
b = O.tensor((K, J), dtype, name="B", device=gpu) # B lives on the accelerator

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu) # C must live on the accelerator

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 128, "i2": 32})
sch.tile("j", {"j1": 128, "j2": 32})
sch.tile("k", {"k1": 64})
sch.unroll({"i2": 2})
sch.gpu_block(["i", "j"])
sch.gpu_thread(["i1", "j1"])
sch.interchange(["i", "j", "i1", "j1","k", "k1", "i2", "j2"])
sched = sch.schedule()
sch.vectorize(["j2"])
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
# CHECK-NEXT:    func.func @matmul(%arg0: memref<512x512xf32> {llvm.noalias}, %arg1: memref<512x512xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x512xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<512x512xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<512x512xf32>, memref<512x512xf32>) outs(%arg2 : memref<512x512xf32>)
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [128, 128, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [32, 32, 0](mapping = [#gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op_2 tile_sizes [0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i2" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_8) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %forall_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_nested_forall_to_threads %3 block_dims = [4, 4, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<512x512xf32> {llvm.noalias}, %arg1: memref<512x512xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x512xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 512] [1, 1] : memref<512x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c512 step %c1 {
# CHECK-NEXT:          %subview_6 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[512, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c4_3 = arith.constant 4 : index
# CHECK-NEXT:      %c4_4 = arith.constant 4 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c4_3, %arg10 = %c4_4, %arg11 = %c1_5) threads(%arg6, %arg7, %arg8) in (%arg12 = %c4, %arg13 = %c4_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %2 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%1, 0] [128, 512] [1, 1] : memref<512x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg1[0, %2] [512, 128] [1, 1] : memref<512x512xf32> to memref<512x128xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_9 = memref.subview %arg2[%1, %2] [128, 128] [1, 1] : memref<512x512xf32> to memref<128x128xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %3 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %4 = affine.apply #map1(%thread_id_y)
# CHECK-NEXT:        %subview_10 = memref.subview %subview[%3, 0] [32, 512] [1, 1] : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<32x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_11 = memref.subview %subview_8[0, %4] [512, 32] [1, 1] : memref<512x128xf32, strided<[512, 1], offset: ?>> to memref<512x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_12 = memref.subview %subview_9[%3, %4] [32, 32] [1, 1] : memref<128x128xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg15 = %c0 to %c512 step %c64 {
# CHECK-NEXT:          %subview_13 = memref.subview %subview_10[0, %arg15] [32, 64] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_14 = memref.subview %subview_11[%arg15, 0] [64, 32] [1, 1] : memref<512x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg16 = %c0 to %c64 step %c1 {
# CHECK-NEXT:            %subview_15 = memref.subview %subview_13[0, %arg16] [32, 1] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_16 = memref.subview %subview_14[%arg16, 0] [1, 32] [1, 1] : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<1x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg17 = %c0 to %c32 step %c2 {
# CHECK-NEXT:              %subview_17 = memref.subview %subview_15[%arg17, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_18 = memref.subview %subview_12[%arg17, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<1x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %5 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:              %6 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[512, 1], offset: ?>>, vector<1x32xf32>
# CHECK-NEXT:              %7 = vector.transfer_read %subview_18[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[512, 1], offset: ?>>, vector<1x32xf32>
# CHECK-NEXT:              %8 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %7 : vector<1x1xf32>, vector<1x32xf32> into vector<1x32xf32>
# CHECK-NEXT:              vector.transfer_write %8, %subview_18[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %9 = arith.addi %arg17, %c1 : index
# CHECK-NEXT:              %subview_19 = memref.subview %subview_15[%9, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_20 = memref.subview %subview_12[%9, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<1x32xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %10 = vector.transfer_read %subview_19[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:              %11 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[512, 1], offset: ?>>, vector<1x32xf32>
# CHECK-NEXT:              %12 = vector.transfer_read %subview_20[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[512, 1], offset: ?>>, vector<1x32xf32>
# CHECK-NEXT:              %13 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %12 : vector<1x1xf32>, vector<1x32xf32> into vector<1x32xf32>
# CHECK-NEXT:              vector.transfer_write %13, %subview_20[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[512, 1], offset: ?>>
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
# CHECK-NEXT:    - %0 : 512x512xfloat32
# CHECK-NEXT:    - %1 : 512x512xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 512x512xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [512x512xfloat32, 512x512xfloat32] -> [512x512xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
