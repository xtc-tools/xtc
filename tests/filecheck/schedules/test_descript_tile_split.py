# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 50, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_dims=["i", "j", "k"],
    spec={
        "k": {},
        "i": {},
        "i#10": {},
        "i[0:5]": {
            "i#5": {},
            "j": {},
        },
        "i[5:]": {
            "i#5": {},
            "j": {},
        }
    }
)

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_tile_slice",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sch.schedule())
evaluator = module.get_evaluator(
    validate=True,
)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<50x64xf32> {llvm.noalias}, %arg1: memref<64x64xf32> {llvm.noalias}, %arg2: memref<50x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<50x64xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<50x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<50x64xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "C/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [10, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "C/i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [5, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C/i0" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.split %tiled_linalg_op_2 after 5  {dimension = 0 : i64} : !transform.any_op
# CHECK-NEXT:      %2:2 = transform.split_handle %1 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %2#0 tile_sizes [5, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "C/i[0]/i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/i[0]/i0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "C/i[0]/j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %2#1 tile_sizes [5, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "C/i[1]/i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "C/i[1]/i0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "C/i[1]/j" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<50x64xf32> {llvm.noalias}, %arg1: memref<64x64xf32> {llvm.noalias}, %arg2: memref<50x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<50x64xf32>)
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c64 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [50, 1] [1, 1] : memref<50x64xf32> to memref<50x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[0, 0] [50, 64] [1, 1] : memref<50x64xf32> to memref<50x64xf32, strided<[64, 1]>>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c50 = arith.constant 50 : index
# CHECK-NEXT:        %c10 = arith.constant 10 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c50 step %c10 {
# CHECK-NEXT:          %subview_3 = memref.subview %subview[%arg4, 0] [10, 1] [1, 1] : memref<50x1xf32, strided<[64, 1], offset: ?>> to memref<10x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview_1[%arg4, 0] [10, 64] [1, 1] : memref<50x64xf32, strided<[64, 1]>> to memref<10x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %c0_6 = arith.constant 0 : index
# CHECK-NEXT:          %c10_7 = arith.constant 10 : index
# CHECK-NEXT:          %c5 = arith.constant 5 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_6 to %c10_7 step %c5 {
# CHECK-NEXT:            %subview_8 = memref.subview %subview_3[%arg5, 0] [5, 1] [1, 1] : memref<10x1xf32, strided<[64, 1], offset: ?>> to memref<5x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_4[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_5[%arg5, 0] [5, 64] [1, 1] : memref<10x64xf32, strided<[64, 1], offset: ?>> to memref<5x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %c0_11 = arith.constant 0 : index
# CHECK-NEXT:            %c5_12 = arith.constant 5 : index
# CHECK-NEXT:            %c5_13 = arith.constant 5 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_11 to %c5_12 step %c5_13 {
# CHECK-NEXT:              %subview_14 = memref.subview %subview_8[%arg6, 0] [5, 1] [1, 1] : memref<5x1xf32, strided<[64, 1], offset: ?>> to memref<5x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              %subview_15 = memref.subview %subview_9[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              %subview_16 = memref.subview %subview_10[%arg6, 0] [5, 64] [1, 1] : memref<5x64xf32, strided<[64, 1], offset: ?>> to memref<5x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              %c0_17 = arith.constant 0 : index
# CHECK-NEXT:              %c5_18 = arith.constant 5 : index
# CHECK-NEXT:              %c1_19 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg7 = %c0_17 to %c5_18 step %c1_19 {
# CHECK-NEXT:                %subview_20 = memref.subview %subview_14[%arg7, 0] [1, 1] [1, 1] : memref<5x1xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %subview_21 = memref.subview %subview_15[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %subview_22 = memref.subview %subview_16[%arg7, 0] [1, 64] [1, 1] : memref<5x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %c0_23 = arith.constant 0 : index
# CHECK-NEXT:                %c64_24 = arith.constant 64 : index
# CHECK-NEXT:                %c1_25 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg8 = %c0_23 to %c64_24 step %c1_25 {
# CHECK-NEXT:                  %subview_26 = memref.subview %subview_20[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_27 = memref.subview %subview_21[0, %arg8] [1, 1] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                  %subview_28 = memref.subview %subview_22[0, %arg8] [1, 1] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                  linalg.matmul {__xtc_id_C_} ins(%subview_26, %subview_27 : memref<1x1xf32, strided<[64, 1], offset: ?>>, memref<1x1xf32, strided<[64, 1], offset: ?>>) outs(%subview_28 : memref<1x1xf32, strided<[64, 1], offset: ?>>)
# CHECK-NEXT:                } {"C/i[0]/j"}
# CHECK-NEXT:              } {"C/i[0]/i0"}
# CHECK-NEXT:            } {"C/i[0]/i"}
# CHECK-NEXT:          } {"C/i0"}
# CHECK-NEXT:        } {"C/i"}
# CHECK-NEXT:      } {"C/k"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 50x64xfloat32
# CHECK-NEXT:    - %1 : 64x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 50x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [50x64xfloat32, 64x64xfloat32] -> [50x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
