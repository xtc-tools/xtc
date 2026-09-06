# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad2d(a, padding=(0, 2, 0, 2), axes=(-2, -1), name="A_pad")
    p2 = O.pad2d(b, padding=((0, 2), (0, 2)), axes=(-2, -1), name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad(m_pad, padding={-2: (0, 2), -1: (0, 2)}, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_tuple_matmul_unpad_mlir",
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
# CHECK-NEXT:      %subview_5 = memref.subview %alloca_3[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      linalg.copy {__xtc_id_C_} ins(%subview_5 : memref<14x14xf32, strided<[16, 1]>>) outs(%arg2 : memref<14x14xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_A_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./h" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_B_pad_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./b" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./h" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_matmul_padded_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %3 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./j" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_matmul_unpad(%arg0: memref<14x14xf32> {llvm.noalias}, %arg1: memref<14x14xf32> {llvm.noalias}, %arg2: memref<14x14xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_A_pad_0_} ins(%cst : f32) outs(%alloca : memref<16x16xf32>)
# CHECK-NEXT:      %subview = memref.subview %alloca[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c14 = arith.constant 14 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c14 step %c1 {
# CHECK-NEXT:        %subview_14 = memref.subview %arg0[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %subview_15 = memref.subview %subview[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c14_17 = arith.constant 14 : index
# CHECK-NEXT:        %c1_18 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_16 to %c14_17 step %c1_18 {
# CHECK-NEXT:          %subview_19 = memref.subview %subview_14[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          %subview_20 = memref.subview %subview_15[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_A_pad_} ins(%subview_19 : memref<1x1xf32, strided<[14, 1], offset: ?>>) outs(%subview_20 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %alloca_0 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_B_pad_0_} ins(%cst_1 : f32) outs(%alloca_0 : memref<16x16xf32>)
# CHECK-NEXT:      %subview_2 = memref.subview %alloca_0[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0_3 = arith.constant 0 : index
# CHECK-NEXT:      %c14_4 = arith.constant 14 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_3 to %c14_4 step %c1_5 {
# CHECK-NEXT:        %subview_14 = memref.subview %arg1[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %subview_15 = memref.subview %subview_2[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c14_17 = arith.constant 14 : index
# CHECK-NEXT:        %c1_18 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_16 to %c14_17 step %c1_18 {
# CHECK-NEXT:          %subview_19 = memref.subview %subview_14[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          %subview_20 = memref.subview %subview_15[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_B_pad_} ins(%subview_19 : memref<1x1xf32, strided<[14, 1], offset: ?>>) outs(%subview_20 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./h"}
# CHECK-NEXT:      } {"./b"}
# CHECK-NEXT:      %alloca_6 = memref.alloca() {alignment = 256 : i64} : memref<16x16xf32>
# CHECK-NEXT:      %cst_7 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_matmul_padded_0_} ins(%cst_7 : f32) outs(%alloca_6 : memref<16x16xf32>)
# CHECK-NEXT:      %c0_8 = arith.constant 0 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1_9 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_8 to %c16 step %c1_9 {
# CHECK-NEXT:        %subview_14 = memref.subview %alloca[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_15 = memref.subview %alloca_0[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1]>>
# CHECK-NEXT:        %subview_16 = memref.subview %alloca_6[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %c16_18 = arith.constant 16 : index
# CHECK-NEXT:        %c1_19 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_17 to %c16_18 step %c1_19 {
# CHECK-NEXT:          %subview_20 = memref.subview %subview_14[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_21 = memref.subview %subview_15[0, %arg4] [16, 1] [1, 1] : memref<16x16xf32, strided<[16, 1]>> to memref<16x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_22 = memref.subview %subview_16[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %c0_23 = arith.constant 0 : index
# CHECK-NEXT:          %c16_24 = arith.constant 16 : index
# CHECK-NEXT:          %c1_25 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_23 to %c16_24 step %c1_25 {
# CHECK-NEXT:            %subview_26 = memref.subview %subview_20[0, %arg5] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_27 = memref.subview %subview_21[%arg5, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_28 = memref.subview %subview_22[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            linalg.matmul {__xtc_id_matmul_padded_} ins(%subview_26, %subview_27 : memref<1x1xf32, strided<[16, 1], offset: ?>>, memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%subview_28 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %subview_10 = memref.subview %alloca_6[0, 0] [14, 14] [1, 1] : memref<16x16xf32> to memref<14x14xf32, strided<[16, 1]>>
# CHECK-NEXT:      %c0_11 = arith.constant 0 : index
# CHECK-NEXT:      %c14_12 = arith.constant 14 : index
# CHECK-NEXT:      %c1_13 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_11 to %c14_12 step %c1_13 {
# CHECK-NEXT:        %subview_14 = memref.subview %subview_10[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32, strided<[16, 1]>> to memref<1x14xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_15 = memref.subview %arg2[%arg3, 0] [1, 14] [1, 1] : memref<14x14xf32> to memref<1x14xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c14_17 = arith.constant 14 : index
# CHECK-NEXT:        %c1_18 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_16 to %c14_17 step %c1_18 {
# CHECK-NEXT:          %subview_19 = memref.subview %subview_14[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_20 = memref.subview %subview_15[0, %arg4] [1, 1] [1, 1] : memref<1x14xf32, strided<[14, 1], offset: ?>> to memref<1x1xf32, strided<[14, 1], offset: ?>>
# CHECK-NEXT:          linalg.copy {__xtc_id_C_} ins(%subview_19 : memref<1x1xf32, strided<[16, 1], offset: ?>>) outs(%subview_20 : memref<1x1xf32, strided<[14, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
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
# CHECK-NEXT:    - %2: pad2d(%0, padding={-2: (0, 2), -1: (0, 2)}, constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %3: pad2d(%1, padding={-2: (0, 2), -1: (0, 2)}, constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %4: matmul(%2, %3) {name = 'matmul_padded'} : [16x16xfloat32, 16x16xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:    - %5: unpad(%4, padding={-2: (0, 2), -1: (0, 2)}) {name = 'C'} : [16x16xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
