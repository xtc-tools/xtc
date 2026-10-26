# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend
from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 128, 128, 128, "bfloat16"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)  # B lives on the accelerator

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)  # C must live on the accelerator

graph = gb.graph
print(graph)

impl = Backend(gb.graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 64, "i2": 2})
sch.tile("j", {"j1": 64, "j2": 2})
sch.tile("k", {"k1": 64})
sch.gpu_block(["i", "j"])
sch.gpu_thread(["i1", "j1"])
sch.interchange(["i", "j", "i1", "j1", "k", "k1", "i2", "j2"])
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    print_source_ir=True,
    print_transformed_ir=True,
    dump_file="matmul_gpu_bf16_mlir",
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<128x128xbf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : bf16
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_C_0_} ins(%cst : bf16) outs(%0 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
# CHECK-NEXT:      %2 = linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<128x128xbf16>, memref<128x128xbf16>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
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
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [2, 2, 0](mapping = [#gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<128x128xbf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : bf16
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %0) -> (tensor<128x128xbf16>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xbf16> to tensor<1x128xbf16>
# CHECK-NEXT:        %c0_0 = arith.constant 0 : index
# CHECK-NEXT:        %c128_1 = arith.constant 128 : index
# CHECK-NEXT:        %c1_2 = arith.constant 1 : index
# CHECK-NEXT:        %3 = scf.for %arg5 = %c0_0 to %c128_1 step %c1_2 iter_args(%arg6 = %extracted_slice) -> (tensor<1x128xbf16>) {
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x128xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:          %4 = linalg.fill {__xtc_id_C_0_} ins(%cst : bf16) outs(%extracted_slice_3 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
# CHECK-NEXT:          %inserted_slice_4 = tensor.insert_slice %4 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xbf16> into tensor<1x128xbf16>
# CHECK-NEXT:          scf.yield %inserted_slice_4 : tensor<1x128xbf16>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<1x128xbf16> into tensor<128x128xbf16>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<128x128xbf16>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %2 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %1) -> (tensor<128x128xbf16>) {
# CHECK-NEXT:        %3 = affine.apply #map(%arg3)
# CHECK-NEXT:        %4 = affine.apply #map(%arg4)
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%3, 0] [64, 128] [1, 1] : tensor<128x128xbf16> to tensor<64x128xbf16>
# CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [128, 64] [1, 1] : tensor<128x128xbf16> to tensor<128x64xbf16>
# CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg5[%3, %4] [64, 64] [1, 1] : tensor<128x128xbf16> to tensor<64x64xbf16>
# CHECK-NEXT:        %5 = scf.forall (%arg6, %arg7) in (32, 32) shared_outs(%arg8 = %extracted_slice_1) -> (tensor<64x64xbf16>) {
# CHECK-NEXT:          %6 = affine.apply #map1(%arg6)
# CHECK-NEXT:          %7 = affine.apply #map1(%arg7)
# CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %extracted_slice[%6, 0] [2, 128] [1, 1] : tensor<64x128xbf16> to tensor<2x128xbf16>
# CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, %7] [128, 2] [1, 1] : tensor<128x64xbf16> to tensor<128x2xbf16>
# CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %arg8[%6, %7] [2, 2] [1, 1] : tensor<64x64xbf16> to tensor<2x2xbf16>
# CHECK-NEXT:          %c0_5 = arith.constant 0 : index
# CHECK-NEXT:          %c128_6 = arith.constant 128 : index
# CHECK-NEXT:          %c64 = arith.constant 64 : index
# CHECK-NEXT:          %8 = scf.for %arg9 = %c0_5 to %c128_6 step %c64 iter_args(%arg10 = %extracted_slice_4) -> (tensor<2x2xbf16>) {
# CHECK-NEXT:            %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[0, %arg9] [2, 64] [1, 1] : tensor<2x128xbf16> to tensor<2x64xbf16>
# CHECK-NEXT:            %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[%arg9, 0] [64, 2] [1, 1] : tensor<128x2xbf16> to tensor<64x2xbf16>
# CHECK-NEXT:            %extracted_slice_9 = tensor.extract_slice %arg10[0, 0] [2, 2] [1, 1] : tensor<2x2xbf16> to tensor<2x2xbf16>
# CHECK-NEXT:            %c0_10 = arith.constant 0 : index
# CHECK-NEXT:            %c64_11 = arith.constant 64 : index
# CHECK-NEXT:            %c1_12 = arith.constant 1 : index
# CHECK-NEXT:            %9 = scf.for %arg11 = %c0_10 to %c64_11 step %c1_12 iter_args(%arg12 = %extracted_slice_9) -> (tensor<2x2xbf16>) {
# CHECK-NEXT:              %extracted_slice_13 = tensor.extract_slice %extracted_slice_7[0, %arg11] [2, 1] [1, 1] : tensor<2x64xbf16> to tensor<2x1xbf16>
# CHECK-NEXT:              %extracted_slice_14 = tensor.extract_slice %extracted_slice_8[%arg11, 0] [1, 2] [1, 1] : tensor<64x2xbf16> to tensor<1x2xbf16>
# CHECK-NEXT:              %extracted_slice_15 = tensor.extract_slice %arg12[0, 0] [2, 2] [1, 1] : tensor<2x2xbf16> to tensor<2x2xbf16>
# CHECK-NEXT:              %c0_16 = arith.constant 0 : index
# CHECK-NEXT:              %c2 = arith.constant 2 : index
# CHECK-NEXT:              %c1_17 = arith.constant 1 : index
# CHECK-NEXT:              %10 = scf.for %arg13 = %c0_16 to %c2 step %c1_17 iter_args(%arg14 = %extracted_slice_15) -> (tensor<2x2xbf16>) {
# CHECK-NEXT:                %extracted_slice_19 = tensor.extract_slice %extracted_slice_13[%arg13, 0] [1, 1] [1, 1] : tensor<2x1xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, 0] [1, 2] [1, 1] : tensor<1x2xbf16> to tensor<1x2xbf16>
# CHECK-NEXT:                %extracted_slice_21 = tensor.extract_slice %arg14[%arg13, 0] [1, 2] [1, 1] : tensor<2x2xbf16> to tensor<1x2xbf16>
# CHECK-NEXT:                %c0_22 = arith.constant 0 : index
# CHECK-NEXT:                %c2_23 = arith.constant 2 : index
# CHECK-NEXT:                %c1_24 = arith.constant 1 : index
# CHECK-NEXT:                %11 = scf.for %arg15 = %c0_22 to %c2_23 step %c1_24 iter_args(%arg16 = %extracted_slice_21) -> (tensor<1x2xbf16>) {
# CHECK-NEXT:                  %extracted_slice_26 = tensor.extract_slice %extracted_slice_19[0, 0] [1, 1] [1, 1] : tensor<1x1xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                  %extracted_slice_27 = tensor.extract_slice %extracted_slice_20[0, %arg15] [1, 1] [1, 1] : tensor<1x2xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                  %extracted_slice_28 = tensor.extract_slice %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x2xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                  %12 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xbf16>, tensor<1x1xbf16>) outs(%extracted_slice_28 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
# CHECK-NEXT:                  %inserted_slice_29 = tensor.insert_slice %12 into %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x1xbf16> into tensor<1x2xbf16>
# CHECK-NEXT:                  scf.yield %inserted_slice_29 : tensor<1x2xbf16>
# CHECK-NEXT:                } {"./j2"}
# CHECK-NEXT:                %inserted_slice_25 = tensor.insert_slice %11 into %arg14[%arg13, 0] [1, 2] [1, 1] : tensor<1x2xbf16> into tensor<2x2xbf16>
# CHECK-NEXT:                scf.yield %inserted_slice_25 : tensor<2x2xbf16>
# CHECK-NEXT:              } {"./i2"}
# CHECK-NEXT:              %inserted_slice_18 = tensor.insert_slice %10 into %arg12[0, 0] [2, 2] [1, 1] : tensor<2x2xbf16> into tensor<2x2xbf16>
# CHECK-NEXT:              scf.yield %inserted_slice_18 : tensor<2x2xbf16>
# CHECK-NEXT:            } {"./k1"}
# CHECK-NEXT:            %inserted_slice = tensor.insert_slice %9 into %arg10[0, 0] [2, 2] [1, 1] : tensor<2x2xbf16> into tensor<2x2xbf16>
# CHECK-NEXT:            scf.yield %inserted_slice : tensor<2x2xbf16>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %8 into %arg8[%6, %7] [2, 2] [1, 1] : tensor<2x2xbf16> into tensor<64x64xbf16>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./i1", mapping = [#gpu.thread<x>, #gpu.thread<y>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg5[%3, %4] [64, 64] [1, 1] : tensor<64x64xbf16> into tensor<128x128xbf16>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]}
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<128x128xbf16>, memref<128x128xbf16>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 128x128xbfloat16
# CHECK-NEXT:    - %1 : 128x128xbfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 128x128xbfloat16
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [128x128xbfloat16, 128x128xbfloat16] -> [128x128xbfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
