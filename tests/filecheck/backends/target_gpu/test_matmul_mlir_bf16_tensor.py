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
    O.matmul(a, b, name="C", device=gpu, acc_dtype="float32")  # C must live on the accelerator

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
    print_bufferization_ir=True,
    print_transformed_ir=True,
    dump_file="matmul_gpu_bf16_tensor_mlir",
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<128x128xbf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<128x128xf32>
# CHECK-NEXT:      %2 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
# CHECK-NEXT:      %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%2 : tensor<128x128xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
# CHECK-NEXT:        %5 = arith.extf %in : bf16 to f32
# CHECK-NEXT:        %6 = arith.extf %in_0 : bf16 to f32
# CHECK-NEXT:        %7 = arith.mulf %5, %6 : f32
# CHECK-NEXT:        %8 = arith.addf %out, %7 : f32
# CHECK-NEXT:        linalg.yield %8 : f32
# CHECK-NEXT:      } -> tensor<128x128xf32>
# CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<128x128xf32>) outs(%0 : tensor<128x128xbf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: bf16):
# CHECK-NEXT:        %5 = arith.truncf %in : f32 to bf16
# CHECK-NEXT:        linalg.yield %5 : bf16
# CHECK-NEXT:      } -> tensor<128x128xbf16>
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<128x128xbf16>, memref<128x128xbf16>) -> ()
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
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_cast_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./j" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: tensor<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<128x128xbf16>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = tensor.empty() : tensor<128x128xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %1) -> (tensor<128x128xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %c128_4 = arith.constant 128 : index
# CHECK-NEXT:        %c1_5 = arith.constant 1 : index
# CHECK-NEXT:        %5 = scf.for %arg5 = %c0_3 to %c128_4 step %c1_5 iter_args(%arg6 = %extracted_slice) -> (tensor<1x128xf32>) {
# CHECK-NEXT:          %extracted_slice_6 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x128xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %6 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_6 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_7 = tensor.insert_slice %6 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x128xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_7 : tensor<1x128xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<128x128xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %3 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %2) -> (tensor<128x128xf32>) {
# CHECK-NEXT:        %5 = affine.apply #map(%arg3)
# CHECK-NEXT:        %6 = affine.apply #map(%arg4)
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%5, 0] [64, 128] [1, 1] : tensor<128x128xbf16> to tensor<64x128xbf16>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg1[0, %6] [128, 64] [1, 1] : tensor<128x128xbf16> to tensor<128x64xbf16>
# CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg5[%5, %6] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
# CHECK-NEXT:        %7 = scf.forall (%arg6, %arg7) in (32, 32) shared_outs(%arg8 = %extracted_slice_4) -> (tensor<64x64xf32>) {
# CHECK-NEXT:          %8 = affine.apply #map1(%arg6)
# CHECK-NEXT:          %9 = affine.apply #map1(%arg7)
# CHECK-NEXT:          %extracted_slice_5 = tensor.extract_slice %extracted_slice[%8, 0] [2, 128] [1, 1] : tensor<64x128xbf16> to tensor<2x128xbf16>
# CHECK-NEXT:          %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %9] [128, 2] [1, 1] : tensor<128x64xbf16> to tensor<128x2xbf16>
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %arg8[%8, %9] [2, 2] [1, 1] : tensor<64x64xf32> to tensor<2x2xf32>
# CHECK-NEXT:          %c0_8 = arith.constant 0 : index
# CHECK-NEXT:          %c128_9 = arith.constant 128 : index
# CHECK-NEXT:          %c64 = arith.constant 64 : index
# CHECK-NEXT:          %10 = scf.for %arg9 = %c0_8 to %c128_9 step %c64 iter_args(%arg10 = %extracted_slice_7) -> (tensor<2x2xf32>) {
# CHECK-NEXT:            %extracted_slice_10 = tensor.extract_slice %extracted_slice_5[0, %arg9] [2, 64] [1, 1] : tensor<2x128xbf16> to tensor<2x64xbf16>
# CHECK-NEXT:            %extracted_slice_11 = tensor.extract_slice %extracted_slice_6[%arg9, 0] [64, 2] [1, 1] : tensor<128x2xbf16> to tensor<64x2xbf16>
# CHECK-NEXT:            %extracted_slice_12 = tensor.extract_slice %arg10[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> to tensor<2x2xf32>
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c64_14 = arith.constant 64 : index
# CHECK-NEXT:            %c1_15 = arith.constant 1 : index
# CHECK-NEXT:            %11 = scf.for %arg11 = %c0_13 to %c64_14 step %c1_15 iter_args(%arg12 = %extracted_slice_12) -> (tensor<2x2xf32>) {
# CHECK-NEXT:              %extracted_slice_16 = tensor.extract_slice %extracted_slice_10[0, %arg11] [2, 1] [1, 1] : tensor<2x64xbf16> to tensor<2x1xbf16>
# CHECK-NEXT:              %extracted_slice_17 = tensor.extract_slice %extracted_slice_11[%arg11, 0] [1, 2] [1, 1] : tensor<64x2xbf16> to tensor<1x2xbf16>
# CHECK-NEXT:              %extracted_slice_18 = tensor.extract_slice %arg12[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> to tensor<2x2xf32>
# CHECK-NEXT:              %c0_19 = arith.constant 0 : index
# CHECK-NEXT:              %c2 = arith.constant 2 : index
# CHECK-NEXT:              %c1_20 = arith.constant 1 : index
# CHECK-NEXT:              %12 = scf.for %arg13 = %c0_19 to %c2 step %c1_20 iter_args(%arg14 = %extracted_slice_18) -> (tensor<2x2xf32>) {
# CHECK-NEXT:                %extracted_slice_22 = tensor.extract_slice %extracted_slice_16[%arg13, 0] [1, 1] [1, 1] : tensor<2x1xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                %extracted_slice_23 = tensor.extract_slice %extracted_slice_17[0, 0] [1, 2] [1, 1] : tensor<1x2xbf16> to tensor<1x2xbf16>
# CHECK-NEXT:                %extracted_slice_24 = tensor.extract_slice %arg14[%arg13, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
# CHECK-NEXT:                %c0_25 = arith.constant 0 : index
# CHECK-NEXT:                %c2_26 = arith.constant 2 : index
# CHECK-NEXT:                %c1_27 = arith.constant 1 : index
# CHECK-NEXT:                %13 = scf.for %arg15 = %c0_25 to %c2_26 step %c1_27 iter_args(%arg16 = %extracted_slice_24) -> (tensor<1x2xf32>) {
# CHECK-NEXT:                  %extracted_slice_29 = tensor.extract_slice %extracted_slice_22[0, 0] [1, 1] [1, 1] : tensor<1x1xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                  %extracted_slice_30 = tensor.extract_slice %extracted_slice_23[0, %arg15] [1, 1] [1, 1] : tensor<1x2xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:                  %extracted_slice_31 = tensor.extract_slice %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x2xf32> to tensor<1x1xf32>
# CHECK-NEXT:                  %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_29, %extracted_slice_30 : tensor<1x1xbf16>, tensor<1x1xbf16>) outs(%extracted_slice_31 : tensor<1x1xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:                  ^bb0(%in: bf16, %in_33: bf16, %out: f32):
# CHECK-NEXT:                    %15 = arith.extf %in : bf16 to f32
# CHECK-NEXT:                    %16 = arith.extf %in_33 : bf16 to f32
# CHECK-NEXT:                    %17 = arith.mulf %15, %16 : f32
# CHECK-NEXT:                    %18 = arith.addf %out, %17 : f32
# CHECK-NEXT:                    linalg.yield %18 : f32
# CHECK-NEXT:                  } -> tensor<1x1xf32>
# CHECK-NEXT:                  %inserted_slice_32 = tensor.insert_slice %14 into %arg16[0, %arg15] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x2xf32>
# CHECK-NEXT:                  scf.yield %inserted_slice_32 : tensor<1x2xf32>
# CHECK-NEXT:                } {"./j2"}
# CHECK-NEXT:                %inserted_slice_28 = tensor.insert_slice %13 into %arg14[%arg13, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_28 : tensor<2x2xf32>
# CHECK-NEXT:              } {"./i2"}
# CHECK-NEXT:              %inserted_slice_21 = tensor.insert_slice %12 into %arg12[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x2xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_21 : tensor<2x2xf32>
# CHECK-NEXT:            } {"./k1"}
# CHECK-NEXT:            %inserted_slice = tensor.insert_slice %11 into %arg10[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x2xf32>
# CHECK-NEXT:            scf.yield %inserted_slice : tensor<2x2xf32>
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %10 into %arg8[%8, %9] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<64x64xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./i1", mapping = [#gpu.thread<x>, #gpu.thread<y>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %7 into %arg5[%5, %6] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./j", mapping = [#gpu.block<x>, #gpu.block<y>]}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c128_1 = arith.constant 128 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %4 = scf.for %arg3 = %c0_0 to %c128_1 step %c1_2 iter_args(%arg4 = %0) -> (tensor<128x128xbf16>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %3[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xbf16> to tensor<1x128xbf16>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c128_5 = arith.constant 128 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        %5 = scf.for %arg5 = %c0_4 to %c128_5 step %c1_6 iter_args(%arg6 = %extracted_slice_3) -> (tensor<1x128xbf16>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, %arg5] [1, 1] [1, 1] : tensor<1x128xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x128xbf16> to tensor<1x1xbf16>
# CHECK-NEXT:          %6 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_7 : tensor<1x1xf32>) outs(%extracted_slice_8 : tensor<1x1xbf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:          ^bb0(%in: f32, %out: bf16):
# CHECK-NEXT:            %7 = arith.truncf %in : f32 to bf16
# CHECK-NEXT:            linalg.yield %7 : bf16
# CHECK-NEXT:          } -> tensor<1x1xbf16>
# CHECK-NEXT:          %inserted_slice_9 = tensor.insert_slice %6 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xbf16> into tensor<1x128xbf16>
# CHECK-NEXT:          scf.yield %inserted_slice_9 : tensor<1x128xbf16>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0] [1, 128] [1, 1] : tensor<1x128xbf16> into tensor<128x128xbf16>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<128x128xbf16>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<128x128xbf16>, memref<128x128xbf16>) -> ()
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
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloc = memref.alloc() {alignment = 256 : i64} : memref<128x128xf32>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %alloc) -> (memref<128x128xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg4[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c128 step %c1 iter_args(%arg6 = %subview) -> (memref<1x128xf32, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_7 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_7 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:          %subview_8 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_7, %subview_8 : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_6 = memref.subview %arg4[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_6 : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<128x128xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c32_1 = arith.constant 32 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c2_3 = arith.constant 2 : index
# CHECK-NEXT:      %c2_4 = arith.constant 2 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c2_3, %arg10 = %c2_4, %arg11 = %c1_5) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c32_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %2 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %3 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%2, 0] [64, 128] [1, 1] : memref<128x128xbf16> to memref<64x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg1[0, %3] [128, 64] [1, 1] : memref<128x128xbf16> to memref<128x64xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_9 = memref.subview %0[%2, %3] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %4 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %5 = affine.apply #map1(%thread_id_y)
# CHECK-NEXT:        %subview_10 = memref.subview %subview[%4, 0] [2, 128] [1, 1] : memref<64x128xbf16, strided<[128, 1], offset: ?>> to memref<2x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_11 = memref.subview %subview_8[0, %5] [128, 2] [1, 1] : memref<128x64xbf16, strided<[128, 1], offset: ?>> to memref<128x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_12 = memref.subview %subview_9[%4, %5] [2, 2] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %6 = scf.for %arg15 = %c0 to %c128 step %c64 iter_args(%arg16 = %subview_12) -> (memref<2x2xf32, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_15 = memref.subview %subview_10[0, %arg15] [2, 64] [1, 1] : memref<2x128xbf16, strided<[128, 1], offset: ?>> to memref<2x64xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_16 = memref.subview %subview_11[%arg15, 0] [64, 2] [1, 1] : memref<128x2xbf16, strided<[128, 1], offset: ?>> to memref<64x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %7 = scf.for %arg17 = %c0 to %c64 step %c1 iter_args(%arg18 = %arg16) -> (memref<2x2xf32, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_17 = memref.subview %subview_15[0, %arg17] [2, 1] [1, 1] : memref<2x64xbf16, strided<[128, 1], offset: ?>> to memref<2x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_18 = memref.subview %subview_16[%arg17, 0] [1, 2] [1, 1] : memref<64x2xbf16, strided<[128, 1], offset: ?>> to memref<1x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %8 = scf.for %arg19 = %c0 to %c2 step %c1 iter_args(%arg20 = %arg18) -> (memref<2x2xf32, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_19 = memref.subview %subview_17[%arg19, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_20 = memref.subview %arg20[%arg19, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<1x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %9 = scf.for %arg21 = %c0 to %c2 step %c1 iter_args(%arg22 = %subview_20) -> (memref<1x2xf32, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_22 = memref.subview %subview_18[0, %arg21] [1, 1] [1, 1] : memref<1x2xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_23 = memref.subview %arg22[0, %arg21] [1, 1] [1, 1] : memref<1x2xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_19, %subview_22 : memref<1x1xbf16, strided<[128, 1], offset: ?>>, memref<1x1xbf16, strided<[128, 1], offset: ?>>) outs(%subview_23 : memref<1x1xf32, strided<[128, 1], offset: ?>>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:                ^bb0(%in: bf16, %in_25: bf16, %out: f32):
# CHECK-NEXT:                  %10 = arith.extf %in : bf16 to f32
# CHECK-NEXT:                  %11 = arith.extf %in_25 : bf16 to f32
# CHECK-NEXT:                  %12 = arith.mulf %10, %11 : f32
# CHECK-NEXT:                  %13 = arith.addf %out, %12 : f32
# CHECK-NEXT:                  linalg.yield %13 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_24 = memref.subview %arg22[0, %arg21] [1, 1] [1, 1] : memref<1x2xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_23, %subview_24 : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg22 : memref<1x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              } {"./j2"}
# CHECK-NEXT:              %subview_21 = memref.subview %arg20[%arg19, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<1x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %9, %subview_21 : memref<1x2xf32, strided<[128, 1], offset: ?>> to memref<1x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg20 : memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            } {"./i2"}
# CHECK-NEXT:            scf.yield %8 : memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          } {"./k1"}
# CHECK-NEXT:          scf.yield %7 : memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        } {"./k"}
# CHECK-NEXT:        %subview_13 = memref.subview %subview_9[%4, %5] [2, 2] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %6, %subview_13 : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        %subview_14 = memref.subview %0[%2, %3] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %subview_9, %subview_14 : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<64x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (memref<128x128xbf16>) {
# CHECK-NEXT:        %subview = memref.subview %0[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_6 = memref.subview %arg4[%arg3, 0] [1, 128] [1, 1] : memref<128x128xbf16> to memref<1x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c128 step %c1 iter_args(%arg6 = %subview_6) -> (memref<1x128xbf16, strided<[128, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_8 = memref.subview %subview[0, %arg5] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_9 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x128xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%subview_8 : memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_9 : memref<1x1xbf16, strided<[128, 1], offset: ?>>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:          ^bb0(%in: f32, %out: bf16):
# CHECK-NEXT:            %3 = arith.truncf %in : f32 to bf16
# CHECK-NEXT:            linalg.yield %3 : bf16
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_10 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x128xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_9, %subview_10 : memref<1x1xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_7 = memref.subview %arg4[%arg3, 0] [1, 128] [1, 1] : memref<128x128xbf16> to memref<1x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_7 : memref<1x128xbf16, strided<[128, 1], offset: ?>> to memref<1x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<128x128xbf16>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      memref.copy %1, %arg2 : memref<128x128xbf16> to memref<128x128xbf16>
# CHECK-NEXT:      return
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
# CHECK-NEXT:    - %2: matmul(%0, %1, acc_dtype=float32) {name = 'C'} : [128x128xbfloat16, 128x128xbfloat16] -> [128x128xbfloat16]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
