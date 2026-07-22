# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 1024, 1024, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
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
        "I#128": {"gpu_thread": 0},
        "J#128": {"gpu_thread": 1},
        "I#32": {},
        "J#32": {},

    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_descript_mlir_gpu",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
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
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [128, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C/I" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 128, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "C/J" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/K" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [32, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "C/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "C/J0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "C/I1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "C/J1" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<1024x512xf32> {llvm.noalias}, %arg1: memref<512x1024xf32> {llvm.noalias, memref.on_device}, %arg2: memref<1024x1024xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024 = arith.constant 1024 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 1024] [1, 1] : memref<1024x1024xf32> to memref<1x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c1024_3 = arith.constant 1024 : index
# CHECK-NEXT:        %c1_4 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c1024_3 step %c1_4 {
# CHECK-NEXT:          %subview_5 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x1024xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_5 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c1024_1 = arith.constant 1024 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c1024_1 step %c128 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [128, 512] [1, 1] : memref<1024x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[0, 0] [512, 1024] [1, 1] : memref<512x1024xf32> to memref<512x1024xf32, strided<[1024, 1]>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[%arg3, 0] [128, 1024] [1, 1] : memref<1024x1024xf32> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c1024_5 = arith.constant 1024 : index
# CHECK-NEXT:        %c128_6 = arith.constant 128 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c1024_5 step %c128_6 {
# CHECK-NEXT:          %subview_7 = memref.subview %subview[0, 0] [128, 512] [1, 1] : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_2[0, %arg4] [512, 128] [1, 1] : memref<512x1024xf32, strided<[1024, 1]>> to memref<512x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %subview_9 = memref.subview %subview_3[0, %arg4] [128, 128] [1, 1] : memref<128x1024xf32, strided<[1024, 1], offset: ?>> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:          %c0_10 = arith.constant 0 : index
# CHECK-NEXT:          %c512 = arith.constant 512 : index
# CHECK-NEXT:          %c1_11 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_10 to %c512 step %c1_11 {
# CHECK-NEXT:            %subview_12 = memref.subview %subview_7[0, %arg5] [128, 1] [1, 1] : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_8[%arg5, 0] [1, 128] [1, 1] : memref<512x128xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %subview_14 = memref.subview %subview_9[0, 0] [128, 128] [1, 1] : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:            %c0_15 = arith.constant 0 : index
# CHECK-NEXT:            %c128_16 = arith.constant 128 : index
# CHECK-NEXT:            %c32 = arith.constant 32 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_15 to %c128_16 step %c32 {
# CHECK-NEXT:              %subview_17 = memref.subview %subview_12[%arg6, 0] [32, 1] [1, 1] : memref<128x1xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_18 = memref.subview %subview_13[0, 0] [1, 128] [1, 1] : memref<1x128xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %subview_19 = memref.subview %subview_14[%arg6, 0] [32, 128] [1, 1] : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<32x128xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:              %c0_20 = arith.constant 0 : index
# CHECK-NEXT:              %c128_21 = arith.constant 128 : index
# CHECK-NEXT:              %c32_22 = arith.constant 32 : index
# CHECK-NEXT:              scf.for %arg7 = %c0_20 to %c128_21 step %c32_22 {
# CHECK-NEXT:                %subview_23 = memref.subview %subview_17[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<32x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                %subview_24 = memref.subview %subview_18[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                %subview_25 = memref.subview %subview_19[0, %arg7] [32, 32] [1, 1] : memref<32x128xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                %c0_26 = arith.constant 0 : index
# CHECK-NEXT:                %c32_27 = arith.constant 32 : index
# CHECK-NEXT:                %c1_28 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg8 = %c0_26 to %c32_27 step %c1_28 {
# CHECK-NEXT:                  %subview_29 = memref.subview %subview_23[%arg8, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                  %subview_30 = memref.subview %subview_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                  %subview_31 = memref.subview %subview_25[%arg8, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                  %c0_32 = arith.constant 0 : index
# CHECK-NEXT:                  %c32_33 = arith.constant 32 : index
# CHECK-NEXT:                  %c1_34 = arith.constant 1 : index
# CHECK-NEXT:                  scf.for %arg9 = %c0_32 to %c32_33 step %c1_34 {
# CHECK-NEXT:                    %subview_35 = memref.subview %subview_29[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                    %subview_36 = memref.subview %subview_30[0, %arg9] [1, 1] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                    %subview_37 = memref.subview %subview_31[0, %arg9] [1, 1] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x1xf32, strided<[1024, 1], offset: ?>>
# CHECK-NEXT:                    linalg.matmul {__xtc_id_C_} ins(%subview_35, %subview_36 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[1024, 1], offset: ?>>) outs(%subview_37 : memref<1x1xf32, strided<[1024, 1], offset: ?>>)
# CHECK-NEXT:                  } {"C/J1"}
# CHECK-NEXT:                } {"C/I1"}
# CHECK-NEXT:              } {"C/J0"}
# CHECK-NEXT:            } {"C/I0"}
# CHECK-NEXT:          } {"C/K"}
# CHECK-NEXT:        } {"C/J"}
# CHECK-NEXT:      } {"C/I"}
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
