# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.gpu import GPUDevice

# Create device
gpu = GPUDevice()

I, J, K, dtype = 1024, 1024, 512, "float32"
a = O.tensor((I, K), dtype, name="A") # A lives on the host
b = O.tensor((K, J), dtype, name="B", device=gpu) # B lives on the accelerator

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu) # C must live on the accelerator

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 128, "i2": 32})
sch.tile("j", {"j1": 128, "j2": 32})
sch.tile("k", {"k1": 64})
# sch.unroll({"i2": 2})
sch.gpu_block(["i", "j"])
sch.gpu_thread(["i1", "j1"])
sch.interchange(["i", "j", "i1", "j1","k", "k1", "i2", "j2"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="gpu_matmul_mlir_offload_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<1024x512xf32> {llvm.noalias}, %arg1: tensor<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1024x1024xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
# CHECK-NEXT:      %2 = linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<512x1024xf32>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./i", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [4, 4, 1] : (!transform.any_op) -> !transform.any_op
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
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./j2" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<1024x512xf32> {llvm.noalias}, %arg1: tensor<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<1024x1024xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %0) -> (tensor<1024x1024xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
# CHECK-NEXT:        %c0_0 = arith.constant 0 : index
# CHECK-NEXT:        %c1024_1 = arith.constant 1024 : index
# CHECK-NEXT:        %c1_2 = arith.constant 1 : index
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0_0 to %c1024_1 step %c1_2 iter_args(%arg6 = %extracted_slice) -> (tensor<1x1024xf32>) {
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1024xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %4 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_3 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_4 = tensor.insert_slice %4 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1024xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_4 : tensor<1x1024xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<1024x1024xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<1024x1024xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %2 = scf.forall (%arg3, %arg4) in (8, 8) shared_outs(%arg5 = %1) -> (tensor<1024x1024xf32>) {
# CHECK-NEXT:        %3 = affine.apply #map(%arg3)
# CHECK-NEXT:        %4 = affine.apply #map(%arg4)
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%3, 0] [128, 512] [1, 1] : tensor<1024x512xf32> to tensor<128x512xf32>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 128] [1, 1] : tensor<512x1024xf32> to tensor<512x128xf32>
# CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg5[%3, %4] [128, 128] [1, 1] : tensor<1024x1024xf32> to tensor<128x128xf32>
# CHECK-NEXT:        %5 = scf.forall (%arg6, %arg7) in (4, 4) shared_outs(%arg8 = %extracted_slice_1) -> (tensor<128x128xf32>) {
# CHECK-NEXT:          %6 = affine.apply #map1(%arg6)
# CHECK-NEXT:          %7 = affine.apply #map1(%arg7)
# CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %extracted_slice[%6, 0] [32, 512] [1, 1] : tensor<128x512xf32> to tensor<32x512xf32>
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, %7] [512, 32] [1, 1] : tensor<512x128xf32> to tensor<512x32xf32>
# CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %arg8[%6, %7] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
# CHECK-NEXT:          %c0_5 = arith.constant 0 : index
# CHECK-NEXT:          %c512 = arith.constant 512 : index
# CHECK-NEXT:          %c64 = arith.constant 64 : index
# CHECK-NEXT:          %8 = scf.for %arg9 = %c0_5 to %c512 step %c64 iter_args(%arg10 = %extracted_slice_4) -> (tensor<32x32xf32>) {
# CHECK-NEXT:            %extracted_slice_6 = tensor.extract_slice %extracted_slice_2[0, %arg9] [32, 64] [1, 1] : tensor<32x512xf32> to tensor<32x64xf32>
# CHECK-NEXT:            %extracted_slice_7 = tensor.extract_slice %extracted_slice_3[%arg9, 0] [64, 32] [1, 1] : tensor<512x32xf32> to tensor<64x32xf32>
# CHECK-NEXT:            %extracted_slice_8 = tensor.extract_slice %arg10[0, 0] [32, 32] [1, 1] : tensor<32x32xf32> to tensor<32x32xf32>
# CHECK-NEXT:            %c0_9 = arith.constant 0 : index
# CHECK-NEXT:            %c64_10 = arith.constant 64 : index
# CHECK-NEXT:            %c1_11 = arith.constant 1 : index
# CHECK-NEXT:            %9 = scf.for %arg11 = %c0_9 to %c64_10 step %c1_11 iter_args(%arg12 = %extracted_slice_8) -> (tensor<32x32xf32>) {
# CHECK-NEXT:              %extracted_slice_12 = tensor.extract_slice %extracted_slice_6[0, %arg11] [32, 1] [1, 1] : tensor<32x64xf32> to tensor<32x1xf32>
# CHECK-NEXT:              %extracted_slice_13 = tensor.extract_slice %extracted_slice_7[%arg11, 0] [1, 32] [1, 1] : tensor<64x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:              %extracted_slice_14 = tensor.extract_slice %arg12[0, 0] [32, 32] [1, 1] : tensor<32x32xf32> to tensor<32x32xf32>
# CHECK-NEXT:              %c0_15 = arith.constant 0 : index
# CHECK-NEXT:              %c32 = arith.constant 32 : index
# CHECK-NEXT:              %c1_16 = arith.constant 1 : index
# CHECK-NEXT:              %10 = scf.for %arg13 = %c0_15 to %c32 step %c1_16 iter_args(%arg14 = %extracted_slice_14) -> (tensor<32x32xf32>) {
# CHECK-NEXT:                %extracted_slice_18 = tensor.extract_slice %extracted_slice_12[%arg13, 0] [1, 1] [1, 1] : tensor<32x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_19 = tensor.extract_slice %extracted_slice_13[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:                %extracted_slice_20 = tensor.extract_slice %arg14[%arg13, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:                %c0_21 = arith.constant 0 : index
# CHECK-NEXT:                %c32_22 = arith.constant 32 : index
# CHECK-NEXT:                %c1_23 = arith.constant 1 : index
# CHECK-NEXT:                %11 = scf.for %arg15 = %c0_21 to %c32_22 step %c1_23 iter_args(%arg16 = %extracted_slice_20) -> (tensor<1x32xf32>) {
# CHECK-NEXT:                  %extracted_slice_25 = tensor.extract_slice %extracted_slice_18[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %extracted_slice_26 = tensor.extract_slice %extracted_slice_19[0, %arg15] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %extracted_slice_27 = tensor.extract_slice %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %12 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_25, %extracted_slice_26 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_27 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:                  %inserted_slice_28 = tensor.insert_slice %12 into %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_28 : tensor<1x32xf32>
# CHECK-NEXT:                } {"./j2"}
# CHECK-NEXT:                %inserted_slice_24 = tensor.insert_slice %11 into %arg14[%arg13, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_24 : tensor<32x32xf32>
# CHECK-NEXT:              } {"./i2"}
# CHECK-NEXT:              %inserted_slice_17 = tensor.insert_slice %10 into %arg12[0, 0] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_17 : tensor<32x32xf32>
# CHECK-NEXT:            } {"./k1"}
# CHECK-NEXT:            %inserted_slice = tensor.insert_slice %9 into %arg10[0, 0] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<32x32xf32>
# CHECK-NEXT:            scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %8 into %arg8[%6, %7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./i1", mapping = [#gpu.thread<x>, #gpu.thread<y>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg5[%3, %4] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<1024x1024xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./i", mapping = [#gpu.block<x>, #gpu.block<y>]}
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./i", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [4, 4, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %arg2) -> (memref<1024x1024xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %1 = scf.for %arg5 = %c0 to %c1024 step %c1 iter_args(%arg6 = %subview) -> (memref<1x1024xf32, strided<[1024, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_6 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:          %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_6, %subview_7 : memref<1x1xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_5 = memref.subview %arg4[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %1, %subview_5 : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1024x1024xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c8_3 = arith.constant 8 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c8, %arg10 = %c8_3, %arg11 = %c1_4) threads(%arg6, %arg7, %arg8) in (%arg12 = %c4, %arg13 = %c4_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %2 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%1, 0] [128, 512] [1, 1] : memref<1024x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg1[0, %2] [512, 128] [1, 1] : memref<512x1024xf32> to memref<512x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %0[%1, %2] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %3 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %4 = affine.apply #map1(%thread_id_y)
# CHECK-NEXT:        %subview_9 = memref.subview %subview[%3, 0] [32, 512] [1, 1] : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<32x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_10 = memref.subview %subview_7[0, %4] [512, 32] [1, 1] : memref<512x128xf32, strided<[1024, 1], offset: ?>> to memref<512x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %subview_11 = memref.subview %subview_8[%3, %4] [32, 32] [1, 1] : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %5 = scf.for %arg15 = %c0 to %c512 step %c64 iter_args(%arg16 = %subview_11) -> (memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_14 = memref.subview %subview_9[0, %arg15] [32, 64] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_15 = memref.subview %subview_10[%arg15, 0] [64, 32] [1, 1] : memref<512x32xf32, strided<[1024, 1], offset: ?>> to memref<64x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %6 = scf.for %arg17 = %c0 to %c64 step %c1 iter_args(%arg18 = %arg16) -> (memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_16 = memref.subview %subview_14[0, %arg17] [32, 1] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_17 = memref.subview %subview_15[%arg17, 0] [1, 32] [1, 1] : memref<64x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %7 = scf.for %arg19 = %c0 to %c32 step %c1 iter_args(%arg20 = %arg18) -> (memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_18 = memref.subview %subview_16[%arg19, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_19 = memref.subview %arg20[%arg19, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %8 = scf.for %arg21 = %c0 to %c32 step %c1 iter_args(%arg22 = %subview_19) -> (memref<1x32xf32, strided<[1024, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_21 = memref.subview %subview_17[0, %arg21] [1, 1] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                %subview_22 = memref.subview %arg22[0, %arg21] [1, 1] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                linalg.matmul {__xtc_id_C_} ins(%subview_18, %subview_21 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[1024, 1], offset: ?>>) outs(%subview_22 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:                %subview_23 = memref.subview %arg22[0, %arg21] [1, 1] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_22, %subview_23 : memref<1x1xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg22 : memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              } {"./j2"}
# CHECK-NEXT:              %subview_20 = memref.subview %arg20[%arg19, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %8, %subview_20 : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg20 : memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            } {"./i2"}
# CHECK-NEXT:            scf.yield %7 : memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          } {"./k1"}
# CHECK-NEXT:          scf.yield %6 : memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        } {"./k"}
# CHECK-NEXT:        %subview_12 = memref.subview %subview_8[%3, %4] [32, 32] [1, 1] : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %5, %subview_12 : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        %subview_13 = memref.subview %0[%1, %2] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %subview_8, %subview_13 : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      memref.copy %0, %arg2 : memref<1024x1024xf32> to memref<1024x1024xf32>
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
