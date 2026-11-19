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

impl = Backend(gb.graph)

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
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<128x128xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%alloca : memref<128x128xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<128x128xbf16>, memref<128x128xbf16>) outs(%alloca : memref<128x128xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
# CHECK-NEXT:        %0 = arith.extf %in : bf16 to f32
# CHECK-NEXT:        %1 = arith.extf %in_0 : bf16 to f32
# CHECK-NEXT:        %2 = arith.mulf %0, %1 : f32
# CHECK-NEXT:        %3 = arith.addf %out, %2 : f32
# CHECK-NEXT:        linalg.yield %3 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%alloca : memref<128x128xf32>) outs(%arg2 : memref<128x128xbf16>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: bf16):
# CHECK-NEXT:        %0 = arith.truncf %in : f32 to bf16
# CHECK-NEXT:        linalg.yield %0 : bf16
# CHECK-NEXT:      }
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
# CHECK-NEXT:      %2 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.gpu.map_nested_forall_to_threads %2 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_C_cast_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %4 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xbf16> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xbf16> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<128x128xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c128 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %alloca[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_8 = arith.constant 0 : index
# CHECK-NEXT:        %c128_9 = arith.constant 128 : index
# CHECK-NEXT:        %c1_10 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_8 to %c128_9 step %c1_10 {
# CHECK-NEXT:          %subview_11 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_11 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c32_1 = arith.constant 32 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c2_3 = arith.constant 2 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c2, %arg10 = %c2_3, %arg11 = %c1_4) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c32_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_8 = arith.constant 0 : index
# CHECK-NEXT:        %c0_9 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%0, 0] [64, 128] [1, 1] : memref<128x128xbf16> to memref<64x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_10 = memref.subview %arg1[0, %1] [128, 64] [1, 1] : memref<128x128xbf16> to memref<128x64xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_11 = memref.subview %alloca[%0, %1] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %2 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %3 = affine.apply #map1(%thread_id_y)
# CHECK-NEXT:        %subview_12 = memref.subview %subview[%2, 0] [2, 128] [1, 1] : memref<64x128xbf16, strided<[128, 1], offset: ?>> to memref<2x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_13 = memref.subview %subview_10[0, %3] [128, 2] [1, 1] : memref<128x64xbf16, strided<[128, 1], offset: ?>> to memref<128x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_14 = memref.subview %subview_11[%2, %3] [2, 2] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_15 = arith.constant 0 : index
# CHECK-NEXT:        %c128_16 = arith.constant 128 : index
# CHECK-NEXT:        %c64 = arith.constant 64 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_15 to %c128_16 step %c64 {
# CHECK-NEXT:          %subview_17 = memref.subview %subview_12[0, %arg15] [2, 64] [1, 1] : memref<2x128xbf16, strided<[128, 1], offset: ?>> to memref<2x64xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_18 = memref.subview %subview_13[%arg15, 0] [64, 2] [1, 1] : memref<128x2xbf16, strided<[128, 1], offset: ?>> to memref<64x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_19 = memref.subview %subview_14[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %c0_20 = arith.constant 0 : index
# CHECK-NEXT:          %c64_21 = arith.constant 64 : index
# CHECK-NEXT:          %c1_22 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_20 to %c64_21 step %c1_22 {
# CHECK-NEXT:            %subview_23 = memref.subview %subview_17[0, %arg16] [2, 1] [1, 1] : memref<2x64xbf16, strided<[128, 1], offset: ?>> to memref<2x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_24 = memref.subview %subview_18[%arg16, 0] [1, 2] [1, 1] : memref<64x2xbf16, strided<[128, 1], offset: ?>> to memref<1x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_25 = memref.subview %subview_19[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<2x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %c0_26 = arith.constant 0 : index
# CHECK-NEXT:            %c2_27 = arith.constant 2 : index
# CHECK-NEXT:            %c1_28 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_26 to %c2_27 step %c1_28 {
# CHECK-NEXT:              %subview_29 = memref.subview %subview_23[%arg17, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_30 = memref.subview %subview_24[0, 0] [1, 2] [1, 1] : memref<1x2xbf16, strided<[128, 1], offset: ?>> to memref<1x2xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_31 = memref.subview %subview_25[%arg17, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[128, 1], offset: ?>> to memref<1x2xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %c0_32 = arith.constant 0 : index
# CHECK-NEXT:              %c2_33 = arith.constant 2 : index
# CHECK-NEXT:              %c1_34 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg18 = %c0_32 to %c2_33 step %c1_34 {
# CHECK-NEXT:                %subview_35 = memref.subview %subview_29[0, 0] [1, 1] [1, 1] : memref<1x1xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_36 = memref.subview %subview_30[0, %arg18] [1, 1] [1, 1] : memref<1x2xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_37 = memref.subview %subview_31[0, %arg18] [1, 1] [1, 1] : memref<1x2xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_35, %subview_36 : memref<1x1xbf16, strided<[128, 1], offset: ?>>, memref<1x1xbf16, strided<[128, 1], offset: ?>>) outs(%subview_37 : memref<1x1xf32, strided<[128, 1], offset: ?>>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:                ^bb0(%in: bf16, %in_38: bf16, %out: f32):
# CHECK-NEXT:                  %4 = arith.extf %in : bf16 to f32
# CHECK-NEXT:                  %5 = arith.extf %in_38 : bf16 to f32
# CHECK-NEXT:                  %6 = arith.mulf %4, %5 : f32
# CHECK-NEXT:                  %7 = arith.addf %out, %6 : f32
# CHECK-NEXT:                  linalg.yield %7 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:              } {"./j2"}
# CHECK-NEXT:            } {"./i2"}
# CHECK-NEXT:          } {"./k1"}
# CHECK-NEXT:        } {"./k"}
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      %c0_5 = arith.constant 0 : index
# CHECK-NEXT:      %c128_6 = arith.constant 128 : index
# CHECK-NEXT:      %c1_7 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_5 to %c128_6 step %c1_7 {
# CHECK-NEXT:        %subview = memref.subview %alloca[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg2[%arg3, 0] [1, 128] [1, 1] : memref<128x128xbf16> to memref<1x128xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_9 = arith.constant 0 : index
# CHECK-NEXT:        %c128_10 = arith.constant 128 : index
# CHECK-NEXT:        %c1_11 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_9 to %c128_10 step %c1_11 {
# CHECK-NEXT:          %subview_12 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_13 = memref.subview %subview_8[0, %arg4] [1, 1] [1, 1] : memref<1x128xbf16, strided<[128, 1], offset: ?>> to memref<1x1xbf16, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%subview_12 : memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_13 : memref<1x1xbf16, strided<[128, 1], offset: ?>>) attrs =  {__xtc_id_C_cast_} {
# CHECK-NEXT:          ^bb0(%in: f32, %out: bf16):
# CHECK-NEXT:            %0 = arith.truncf %in : f32 to bf16
# CHECK-NEXT:            linalg.yield %0 : bf16
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
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
