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

impl = Backend(graph, use_tensor_dialect=True)

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
    dump_file="matmul_gpu_shared_tensor_mlir",
    print_source_ir=True,
    print_bufferization_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<256x256xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<256x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<256x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<256x256xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
# CHECK-NEXT:      %2 = linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%1 : tensor<256x256xf32>) -> tensor<256x256xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [16, 16, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {"./k"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.xtc.promote_shared_memory %2, %3 {input_idx = 0 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %5 = transform.xtc.promote_shared_memory %2, %3 {alignment = 4 : i64, input_idx = 1 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
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
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<256x256xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<256x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<256x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<256x256xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %0) -> (tensor<256x256xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 256] [1, 1] : tensor<256x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:        %c0_0 = arith.constant 0 : index
# CHECK-NEXT:        %c256_1 = arith.constant 256 : index
# CHECK-NEXT:        %c1_2 = arith.constant 1 : index
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0_0 to %c256_1 step %c1_2 iter_args(%arg6 = %extracted_slice) -> (tensor<1x256xf32>) {
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %4 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_3 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_4 = tensor.insert_slice %4 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x256xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_4 : tensor<1x256xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<256x256xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<256x256xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %2 = scf.forall (%arg3, %arg4) in (4, 4) shared_outs(%arg5 = %1) -> (tensor<256x256xf32>) {
# CHECK-NEXT:        %3 = affine.apply #map(%arg3)
# CHECK-NEXT:        %4 = affine.apply #map(%arg4)
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [256, 64] [1, 1] : tensor<256x256xf32> to tensor<256x64xf32>
# CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg5[%3, %4] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
# CHECK-NEXT:        %5 = scf.forall (%arg6, %arg7) in (16, 16) shared_outs(%arg8 = %extracted_slice_1) -> (tensor<64x64xf32>) {
# CHECK-NEXT:          %6 = affine.apply #map1(%arg6)
# CHECK-NEXT:          %7 = affine.apply #map1(%arg7)
# CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %extracted_slice[%6, 0] [4, 256] [1, 1] : tensor<64x256xf32> to tensor<4x256xf32>
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, %7] [256, 4] [1, 1] : tensor<256x64xf32> to tensor<256x4xf32>
# CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %arg8[%6, %7] [4, 4] [1, 1] : tensor<64x64xf32> to tensor<4x4xf32>
# CHECK-NEXT:          %c0_5 = arith.constant 0 : index
# CHECK-NEXT:          %c256_6 = arith.constant 256 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %8 = scf.for %arg9 = %c0_5 to %c256_6 step %c32 iter_args(%arg10 = %extracted_slice_4) -> (tensor<4x4xf32>) {
# CHECK-NEXT:            %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[0, %arg9] [4, 32] [1, 1] : tensor<4x256xf32> to tensor<4x32xf32>
# CHECK-NEXT:            %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[%arg9, 0] [32, 4] [1, 1] : tensor<256x4xf32> to tensor<32x4xf32>
# CHECK-NEXT:            %extracted_slice_9 = tensor.extract_slice %arg10[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
# CHECK-NEXT:            %c0_10 = arith.constant 0 : index
# CHECK-NEXT:            %c32_11 = arith.constant 32 : index
# CHECK-NEXT:            %c1_12 = arith.constant 1 : index
# CHECK-NEXT:            %9 = scf.for %arg11 = %c0_10 to %c32_11 step %c1_12 iter_args(%arg12 = %extracted_slice_9) -> (tensor<4x4xf32>) {
# CHECK-NEXT:              %extracted_slice_13 = tensor.extract_slice %extracted_slice_7[0, %arg11] [4, 1] [1, 1] : tensor<4x32xf32> to tensor<4x1xf32>
# CHECK-NEXT:              %extracted_slice_14 = tensor.extract_slice %extracted_slice_8[%arg11, 0] [1, 4] [1, 1] : tensor<32x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:              %extracted_slice_15 = tensor.extract_slice %arg12[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
# CHECK-NEXT:              %c0_16 = arith.constant 0 : index
# CHECK-NEXT:              %c4 = arith.constant 4 : index
# CHECK-NEXT:              %c1_17 = arith.constant 1 : index
# CHECK-NEXT:              %10 = scf.for %arg13 = %c0_16 to %c4 step %c1_17 iter_args(%arg14 = %extracted_slice_15) -> (tensor<4x4xf32>) {
# CHECK-NEXT:                %extracted_slice_19 = tensor.extract_slice %extracted_slice_13[%arg13, 0] [1, 1] [1, 1] : tensor<4x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:                %extracted_slice_21 = tensor.extract_slice %arg14[%arg13, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
# CHECK-NEXT:                %c0_22 = arith.constant 0 : index
# CHECK-NEXT:                %c4_23 = arith.constant 4 : index
# CHECK-NEXT:                %c1_24 = arith.constant 1 : index
# CHECK-NEXT:                %11 = scf.for %arg15 = %c0_22 to %c4_23 step %c1_24 iter_args(%arg16 = %extracted_slice_21) -> (tensor<1x4xf32>) {
# CHECK-NEXT:                  %extracted_slice_26 = tensor.extract_slice %extracted_slice_19[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %extracted_slice_27 = tensor.extract_slice %extracted_slice_20[0, %arg15] [1, 1] [1, 1] : tensor<1x4xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %extracted_slice_28 = tensor.extract_slice %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x4xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %12 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_28 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:                  %inserted_slice_29 = tensor.insert_slice %12 into %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x4xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_29 : tensor<1x4xf32>
# CHECK-NEXT:                } {"./j2"}
# CHECK-NEXT:                %inserted_slice_25 = tensor.insert_slice %11 into %arg14[%arg13, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_25 : tensor<4x4xf32>
# CHECK-NEXT:              } {"./i2"}
# CHECK-NEXT:              %inserted_slice_18 = tensor.insert_slice %10 into %arg12[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x4xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_18 : tensor<4x4xf32>
# CHECK-NEXT:            } {"./k1"}
# CHECK-NEXT:            %inserted_slice = tensor.insert_slice %9 into %arg10[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x4xf32>
# CHECK-NEXT:            scf.yield %inserted_slice : tensor<4x4xf32>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %8 into %arg8[%6, %7] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<64x64xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./i1", mapping = [#gpu.thread<x>, #gpu.thread<y>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg5[%3, %4] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]}
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [16, 16, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {"./k"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.xtc.promote_shared_memory %2, %3 {input_idx = 0 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %5 = transform.xtc.promote_shared_memory %2, %3 {alignment = 4 : i64, input_idx = 1 : i64, target = "nvgpu"} : !transform.any_op, !transform.any_op -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0 + d1 * 16)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map2 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg1: memref<256x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<256x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (memref<256x256xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %1 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %subview) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_17 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_17 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:          %subview_18 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_17, %subview_18 : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_16 = memref.subview %arg4[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %1, %subview_16 : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<256x256xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c16_1 = arith.constant 16 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c4_3 = arith.constant 4 : index
# CHECK-NEXT:      %c4_4 = arith.constant 4 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      %c0_6 = arith.constant 0 : index
# CHECK-NEXT:      %c1_7 = arith.constant 1 : index
# CHECK-NEXT:      %c256_8 = arith.constant 256 : index
# CHECK-NEXT:      %c2048 = arith.constant 2048 : index
# CHECK-NEXT:      %c32_9 = arith.constant 32 : index
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c31 = arith.constant 31 : index
# CHECK-NEXT:      %c0_10 = arith.constant 0 : index
# CHECK-NEXT:      %c1_11 = arith.constant 1 : index
# CHECK-NEXT:      %c256_12 = arith.constant 256 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c16_13 = arith.constant 16 : index
# CHECK-NEXT:      %c4_14 = arith.constant 4 : index
# CHECK-NEXT:      %c15 = arith.constant 15 : index
# CHECK-NEXT:      %c4_15 = arith.constant 4 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c4_3, %arg10 = %c4_4, %arg11 = %c1_5) threads(%arg6, %arg7, %arg8) in (%arg12 = %c16, %arg13 = %c16_1, %arg14 = %c1_2) workgroup(%arg15 : memref<64x32xf32, 3 : i32>, %arg16 : memref<32x64xf32, 3 : i32>) {
# CHECK-NEXT:        %1 = affine.apply #map(%arg6, %arg7)
# CHECK-NEXT:        %2 = affine.apply #map(%arg6, %arg7)
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %3 = affine.apply #map1(%block_id_x)
# CHECK-NEXT:        %4 = affine.apply #map1(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%3, 0] [64, 256] [1, 1] : memref<256x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_18 = memref.subview %arg1[0, %4] [256, 64] [1, 1] : memref<256x256xf32> to memref<256x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_19 = memref.subview %0[%3, %4] [64, 64] [1, 1] : memref<256x256xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %5 = affine.apply #map2(%thread_id_x)
# CHECK-NEXT:        %6 = affine.apply #map2(%thread_id_y)
# CHECK-NEXT:        %subview_20 = memref.subview %subview[%5, 0] [4, 256] [1, 1] : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<4x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_21 = memref.subview %subview_18[0, %6] [256, 4] [1, 1] : memref<256x64xf32, strided<[256, 1], offset: ?>> to memref<256x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_22 = memref.subview %subview_19[%5, %6] [4, 4] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %7 = scf.for %arg17 = %c0 to %c256 step %c32 iter_args(%arg18 = %subview_22) -> (memref<4x4xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_25 = memref.subview %subview_20[0, %arg17] [4, 32] [1, 1] : memref<4x256xf32, strided<[256, 1], offset: ?>> to memref<4x32xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          gpu.barrier
# CHECK-NEXT:          scf.for %arg19 = %2 to %c2048 step %c256_8 {
# CHECK-NEXT:            %10 = arith.shrui %arg19, %c5 : index
# CHECK-NEXT:            %11 = arith.andi %arg19, %c31 : index
# CHECK-NEXT:            %12 = arith.addi %arg17, %11 : index
# CHECK-NEXT:            %13 = nvgpu.device_async_copy %subview[%10, %12], %arg15[%10, %11], 1 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x32xf32, 3 : i32>
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_26 = memref.subview %subview_21[%arg17, 0] [32, 4] [1, 1] : memref<256x4xf32, strided<[256, 1], offset: ?>> to memref<32x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg19 = %1 to %c512 step %c256_12 {
# CHECK-NEXT:            %10 = arith.shrui %arg19, %c4_14 : index
# CHECK-NEXT:            %11 = arith.andi %arg19, %c15 : index
# CHECK-NEXT:            %12 = arith.muli %11, %c4_15 : index
# CHECK-NEXT:            %13 = arith.addi %arg17, %10 : index
# CHECK-NEXT:            %14 = nvgpu.device_async_copy %subview_18[%13, %12], %arg16[%10, %12], 4 : memref<256x64xf32, strided<[256, 1], offset: ?>> to memref<32x64xf32, 3 : i32>
# CHECK-NEXT:          }
# CHECK-NEXT:          %8 = nvgpu.device_async_create_group 
# CHECK-NEXT:          nvgpu.device_async_wait %8 {numGroups = 0 : i32}
# CHECK-NEXT:          gpu.barrier
# CHECK-NEXT:          %9 = scf.for %arg19 = %c0 to %c32 step %c1 iter_args(%arg20 = %arg18) -> (memref<4x4xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_27 = memref.subview %subview_25[0, %arg19] [4, 1] [1, 1] : memref<4x32xf32, strided<[256, 1], offset: ?>> to memref<4x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_28 = memref.subview %subview_26[%arg19, 0] [1, 4] [1, 1] : memref<32x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %10 = scf.for %arg21 = %c0 to %c4 step %c1 iter_args(%arg22 = %arg20) -> (memref<4x4xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:              %11 = arith.addi %c0_6, %5 : index
# CHECK-NEXT:              %12 = arith.addi %c0_6, %arg19 : index
# CHECK-NEXT:              %13 = arith.addi %11, %arg21 : index
# CHECK-NEXT:              %subview_29 = memref.subview %arg15[%13, %12] [1, 1] [1, 1] : memref<64x32xf32, 3 : i32> to memref<1x1xf32, strided<[32, 1], offset: ?>, 3 : i32>
# CHECK-NEXT:              %subview_30 = memref.subview %arg22[%arg21, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %14 = scf.for %arg23 = %c0 to %c4 step %c1 iter_args(%arg24 = %subview_30) -> (memref<1x4xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:                %15 = arith.addi %c0_10, %6 : index
# CHECK-NEXT:                %16 = arith.addi %c0_10, %arg19 : index
# CHECK-NEXT:                %17 = arith.addi %15, %arg23 : index
# CHECK-NEXT:                %subview_32 = memref.subview %arg16[%16, %17] [1, 1] [1, 1] : memref<32x64xf32, 3 : i32> to memref<1x1xf32, strided<[64, 1], offset: ?>, 3 : i32>
# CHECK-NEXT:                %subview_33 = memref.subview %arg24[0, %arg23] [1, 1] [1, 1] : memref<1x4xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                linalg.matmul {__xtc_id_C_} ins(%subview_29, %subview_32 : memref<1x1xf32, strided<[32, 1], offset: ?>, 3 : i32>, memref<1x1xf32, strided<[64, 1], offset: ?>, 3 : i32>) outs(%subview_33 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:                %subview_34 = memref.subview %arg24[0, %arg23] [1, 1] [1, 1] : memref<1x4xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_33, %subview_34 : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg24 : memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              } {"./j2"}
# CHECK-NEXT:              %subview_31 = memref.subview %arg22[%arg21, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %14, %subview_31 : memref<1x4xf32, strided<[256, 1], offset: ?>> to memref<1x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg22 : memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            } {"./i2"}
# CHECK-NEXT:            scf.yield %10 : memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          } {"./k1"}
# CHECK-NEXT:          scf.yield %9 : memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        } {"./k"}
# CHECK-NEXT:        %subview_23 = memref.subview %subview_19[%5, %6] [4, 4] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %7, %subview_23 : memref<4x4xf32, strided<[256, 1], offset: ?>> to memref<4x4xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        %subview_24 = memref.subview %0[%3, %4] [64, 64] [1, 1] : memref<256x256xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %subview_19, %subview_24 : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<64x64xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      memref.copy %0, %arg2 : memref<256x256xf32> to memref<256x256xf32>
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
