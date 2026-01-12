# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad2d(a, padding=(0, 2), name="A_pad")
    p2 = O.pad2d(b, padding=(0, 2), name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad2d(m_pad, padding=(0, 2), name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_matmul_unpad_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: memref<14x14xf32> {llvm.noalias}, %arg1: memref<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%alloca : memref<16x16xf32>)
# CHECK-NEXT:      %subview = memref.subview %alloca[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      linalg.copy {__xtc_id_A_pad_} ins(%arg0 : memref<14x14xf32>) outs(%subview : memref<14x14xf32, strided<[16, 1]>>)
# CHECK-NEXT:      %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_B_pad_0_} ins(%cst_1 : f32) outs(%alloca_0 : memref<16x16xf32>)
# CHECK-NEXT:      %subview_2 = memref.subview %alloca_0[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      linalg.copy {__xtc_id_B_pad_} ins(%arg1 : memref<14x14xf32>) outs(%subview_2 : memref<14x14xf32, strided<[16, 1]>>)
# CHECK-NEXT:      %alloca_3 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_4 : f32) outs(%alloca_3 : memref<16x16xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_matmul_padded_} ins(%alloca, %alloca_0 : memref<16x16xf32>, memref<16x16xf32>) outs(%alloca_3 : memref<16x16xf32>)
# CHECK-NEXT:      %cst_5 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst_5 : f32) outs(%arg2 : memref<14x14xf32>)
# CHECK-NEXT:      %subview_6 = memref.subview %alloca_3[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      linalg.copy {__xtc_id_C_} ins(%subview_6 : memref<14x14xf32, strided<[16, 1]>>) outs(%arg2 : memref<14x14xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_A_pad_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./d1" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_A_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./d1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_B_pad_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./d1" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_B_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %3 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./d1" : !transform.any_op
# CHECK-NEXT:      %4 = transform.structured.match attributes {__xtc_id_matmul_padded_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %4 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./j" : !transform.any_op
# CHECK-NEXT:      %5 = transform.structured.match attributes {__xtc_id_matmul_padded_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %5 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_22, %loops_23 = transform.structured.tile_using_for %tiled_linalg_op_20 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_23 "./k" : !transform.any_op
# CHECK-NEXT:      %6 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_24, %loops_25 = transform.structured.tile_using_for %6 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_25 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_26, %loops_27 = transform.structured.tile_using_for %tiled_linalg_op_24 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_27 "./d1" : !transform.any_op
# CHECK-NEXT:      %7 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_28, %loops_29 = transform.structured.tile_using_for %7 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_29 "./d0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_30, %loops_31 = transform.structured.tile_using_for %tiled_linalg_op_28 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_31 "./d1" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: memref<14x14xf32> {llvm.noalias}, %arg1: memref<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c16 step %c1 {
# CHECK-NEXT:        %subview_27 = memref.subview %alloca[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_28 = arith.constant 0 : index
# CHECK-NEXT:        %c16_29 = arith.constant 16 : index
# CHECK-NEXT:        %c1_30 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_28 to %c16_29 step %c1_30 {
# CHECK-NEXT:          %subview_31 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%subview_31 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %subview = memref.subview %alloca[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c14 = arith.constant 14 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c14 step %c1_1 {
# CHECK-NEXT:        %subview_27 = memref.subview %arg0[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %subview_28 = memref.subview %subview[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_29 = arith.constant 0 : index
# CHECK-NEXT:        %c14_30 = arith.constant 14 : index
# CHECK-NEXT:        %c1_31 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_29 to %c14_30 step %c1_31 {
# CHECK-NEXT:          %subview_32 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          %subview_33 = memref.subview %subview_28[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_A_pad_} ins(%subview_32 : memref<1x1xf32, strided<[14, 1], offset: ?>>) outs(%subview_33 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %alloca_2 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0_4 = arith.constant 0 : index
# CHECK-NEXT:      %c16_5 = arith.constant 16 : index
# CHECK-NEXT:      %c1_6 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_4 to %c16_5 step %c1_6 {
# CHECK-NEXT:        %subview_27 = memref.subview %alloca_2[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_28 = arith.constant 0 : index
# CHECK-NEXT:        %c16_29 = arith.constant 16 : index
# CHECK-NEXT:        %c1_30 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_28 to %c16_29 step %c1_30 {
# CHECK-NEXT:          %subview_31 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_B_pad_0_} ins(%cst_3 : f32) outs(%subview_31 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %subview_7 = memref.subview %alloca_2[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0_8 = arith.constant 0 : index
# CHECK-NEXT:      %c14_9 = arith.constant 14 : index
# CHECK-NEXT:      %c1_10 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_8 to %c14_9 step %c1_10 {
# CHECK-NEXT:        %subview_27 = memref.subview %arg1[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %subview_28 = memref.subview %subview_7[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_29 = arith.constant 0 : index
# CHECK-NEXT:        %c14_30 = arith.constant 14 : index
# CHECK-NEXT:        %c1_31 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_29 to %c14_30 step %c1_31 {
# CHECK-NEXT:          %subview_32 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          %subview_33 = memref.subview %subview_28[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_B_pad_} ins(%subview_32 : memref<1x1xf32, strided<[14, 1], offset: ?>>) outs(%subview_33 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %alloca_11 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_12 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0_13 = arith.constant 0 : index
# CHECK-NEXT:      %c16_14 = arith.constant 16 : index
# CHECK-NEXT:      %c1_15 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_13 to %c16_14 step %c1_15 {
# CHECK-NEXT:        %subview_27 = memref.subview %alloca_11[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_28 = arith.constant 0 : index
# CHECK-NEXT:        %c16_29 = arith.constant 16 : index
# CHECK-NEXT:        %c1_30 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_28 to %c16_29 step %c1_30 {
# CHECK-NEXT:          %subview_31 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_12 : f32) outs(%subview_31 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_16 = arith.constant 0 : index
# CHECK-NEXT:      %c16_17 = arith.constant 16 : index
# CHECK-NEXT:      %c1_18 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_16 to %c16_17 step %c1_18 {
# CHECK-NEXT:        %subview_27 = memref.subview %alloca[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_28 = memref.subview %alloca_2[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1]>>
# CHECK-NEXT:        %subview_29 = memref.subview %alloca_11[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_30 = arith.constant 0 : index
# CHECK-NEXT:        %c16_31 = arith.constant 16 : index
# CHECK-NEXT:        %c1_32 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_30 to %c16_31 step %c1_32 {
# CHECK-NEXT:          %subview_33 = memref.subview %subview_27[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_34 = memref.subview %subview_28[0, %arg4] [16, 1] [1, 1] : memref<16x16xf32, strided<[16, 1]>> to memref<16x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_35 = memref.subview %subview_29[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %c0_36 = arith.constant 0 : index
# CHECK-NEXT:          %c16_37 = arith.constant 16 : index
# CHECK-NEXT:          %c1_38 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_36 to %c16_37 step %c1_38 {
# CHECK-NEXT:            %subview_39 = memref.subview %subview_33[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_40 = memref.subview %subview_34[%arg5, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_41 = memref.subview %subview_35[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            linalg.matmul {__xtc_id_matmul_padded_} ins(%subview_39, %subview_40 : memref<1x1xf32, strided<[16, 1], offset: ?>>, memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%subview_41 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %cst_19 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0_20 = arith.constant 0 : index
# CHECK-NEXT:      %c14_21 = arith.constant 14 : index
# CHECK-NEXT:      %c1_22 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_20 to %c14_21 step %c1_22 {
# CHECK-NEXT:        %subview_27 = memref.subview %arg2[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %c0_28 = arith.constant 0 : index
# CHECK-NEXT:        %c14_29 = arith.constant 14 : index
# CHECK-NEXT:        %c1_30 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_28 to %c14_29 step %c1_30 {
# CHECK-NEXT:          %subview_31 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst_19 : f32) outs(%subview_31 : memref<1x1xf32, strided<[14, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      %subview_23 = memref.subview %alloca_11[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0_24 = arith.constant 0 : index
# CHECK-NEXT:      %c14_25 = arith.constant 14 : index
# CHECK-NEXT:      %c1_26 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_24 to %c14_25 step %c1_26 {
# CHECK-NEXT:        %subview_27 = memref.subview %subview_23[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_28 = memref.subview %arg2[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %c0_29 = arith.constant 0 : index
# CHECK-NEXT:        %c14_30 = arith.constant 14 : index
# CHECK-NEXT:        %c1_31 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_29 to %c14_30 step %c1_31 {
# CHECK-NEXT:          %subview_32 = memref.subview %subview_27[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_33 = memref.subview %subview_28[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_C_} ins(%subview_32 : memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%subview_33 : memref<1x1xf32, strided<[14, 1], offset: ?>>)
# CHECK-NEXT:        } {"./d1"}
# CHECK-NEXT:      } {"./d0"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: pad_matmul_unpad
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 14x14xfloat32
# CHECK-NEXT:    - %1 : 14x14xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %5 : 14x14xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding=(0, 2, 0, 2), axis=(-2, -1), constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %3: pad2d(%1, padding=(0, 2, 0, 2), axis=(-2, -1), constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %4: matmul(%2, %3) {name = 'matmul_padded'} : [16x16xfloat32, 16x16xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %5: unpad2d(%4, padding=(0, 2, 0, 2), axis=(-2, -1)) {name = 'C'} : [16x16xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
