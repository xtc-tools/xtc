# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler = sch,
    node_name = "C_reduce",
    abstract_axis = ["i","j","k"],
    spec = {
        "k": {},
        "i": {},
        "j": {},
        "i#2": {"unroll": None},
        "j#16": {"vectorize": None},
    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_fill_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_reduce_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_fill_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_reduce_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C_reduce/k0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "C_reduce/i0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C_reduce/j0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "C_reduce/i1" : !transform.any_op
# CHECK-NEXT:      transform.structured.vectorize %tiled_linalg_op_8 : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, 0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (0, d1, d0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_3 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c32 step %c1_3 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_fill_} ins(%cst : f32) outs(%subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:        } {j}
# CHECK-NEXT:      } {i}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c512 step %c1_1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c4_5 = arith.constant 4 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c4_5 step %c2 {
# CHECK-NEXT:          %subview_6 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_7 = memref.subview %subview_2[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_3[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %c0_9 = arith.constant 0 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_9 to %c32 step %c16 {
# CHECK-NEXT:            %subview_10 = memref.subview %subview_6[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_7[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_8[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_13 = arith.constant 0 : index
# CHECK-NEXT:            %c2_14 = arith.constant 2 : index
# CHECK-NEXT:            %c1_15 = arith.constant 1 : index
# CHECK-NEXT:            %c2_16 = arith.constant 2 : index
# CHECK-NEXT:            %subview_17 = memref.subview %subview_10[%c0_13, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_18 = memref.subview %subview_11[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %subview_12[%c0_13, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_20 = arith.constant 1 : index
# CHECK-NEXT:            %c16_21 = arith.constant 16 : index
# CHECK-NEXT:            %c1_22 = arith.constant 1 : index
# CHECK-NEXT:            %c0_23 = arith.constant 0 : index
# CHECK-NEXT:            %cst_24 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %0 = vector.transfer_read %subview_17[%c0_23, %c0_23], %cst_24 {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_25 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %1 = vector.transfer_read %subview_18[%c0_23, %c0_23], %cst_25 {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_26 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %2 = vector.transfer_read %subview_19[%c0_23, %c0_23], %cst_26 : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %3 = arith.mulf %0, %1 : vector<1x16x1xf32>
# CHECK-NEXT:            %4 = vector.multi_reduction <add>, %3, %2 [2] : vector<1x16x1xf32> to vector<1x16xf32>
# CHECK-NEXT:            %c0_27 = arith.constant 0 : index
# CHECK-NEXT:            vector.transfer_write %4, %subview_19[%c0_27, %c0_27] : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_28 = arith.constant 1 : index
# CHECK-NEXT:            %5 = arith.muli %c1_15, %c1_28 : index
# CHECK-NEXT:            %6 = arith.addi %c0_13, %5 : index
# CHECK-NEXT:            %subview_29 = memref.subview %subview_10[%6, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_11[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_31 = memref.subview %subview_12[%6, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_32 = arith.constant 1 : index
# CHECK-NEXT:            %c16_33 = arith.constant 16 : index
# CHECK-NEXT:            %c1_34 = arith.constant 1 : index
# CHECK-NEXT:            %c0_35 = arith.constant 0 : index
# CHECK-NEXT:            %cst_36 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %7 = vector.transfer_read %subview_29[%c0_35, %c0_35], %cst_36 {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_37 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %8 = vector.transfer_read %subview_30[%c0_35, %c0_35], %cst_37 {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_38 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %9 = vector.transfer_read %subview_31[%c0_35, %c0_35], %cst_38 : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %10 = arith.mulf %7, %8 : vector<1x16x1xf32>
# CHECK-NEXT:            %11 = vector.multi_reduction <add>, %10, %9 [2] : vector<1x16x1xf32> to vector<1x16xf32>
# CHECK-NEXT:            %c0_39 = arith.constant 0 : index
# CHECK-NEXT:            vector.transfer_write %11, %subview_31[%c0_39, %c0_39] : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {"C_reduce/j0"}
# CHECK-NEXT:        } {"C_reduce/i0"}
# CHECK-NEXT:      } {"C_reduce/k0"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'}
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
