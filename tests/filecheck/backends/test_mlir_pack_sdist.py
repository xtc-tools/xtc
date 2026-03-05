# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_sdist

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.unroll({"i1": 2})
sch.pack_at("i", 1)
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="mlir_pack",
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
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./k" : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %tiled_linalg_op {
# CHECK-NEXT:        transform.apply_patterns.memref.fold_memref_alias_ops
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %1 = transform.sdist.local_buffer_at %tiled_linalg_op tensor 1 : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./j1" : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_5 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        %alloc = memref.alloc() : memref<1x32xf32, 2>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        sdist.read %arg1[%arg3, %c0_2] to %alloc : memref<512x32xf32>, memref<1x32xf32, 2>
# CHECK-NEXT:        %c0_3 = arith.constant 0 : index
# CHECK-NEXT:        %c4 = arith.constant 4 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_3 to %c4 step %c2 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %alloc[0, 0] [1, 32] [1, 1] : memref<1x32xf32, 2> to memref<1x32xf32, strided<[32, 1]>, 2>
# CHECK-NEXT:          %subview_6 = memref.subview %subview_1[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %c0_7 = arith.constant 0 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_7 to %c32 step %c16 {
# CHECK-NEXT:            %subview_8 = memref.subview %subview_4[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_5[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1]>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_6[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_11 = arith.constant 0 : index
# CHECK-NEXT:            %c2_12 = arith.constant 2 : index
# CHECK-NEXT:            %c1_13 = arith.constant 1 : index
# CHECK-NEXT:            %c2_14 = arith.constant 2 : index
# CHECK-NEXT:            %subview_15 = memref.subview %subview_8[%c0_11, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_16 = memref.subview %subview_9[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_17 = memref.subview %subview_10[%c0_11, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_18 = arith.constant 0 : index
# CHECK-NEXT:            %c16_19 = arith.constant 16 : index
# CHECK-NEXT:            %c1_20 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_18 to %c16_19 step %c1_20 {
# CHECK-NEXT:              %subview_28 = memref.subview %subview_15[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_29 = memref.subview %subview_16[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x1xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:              %subview_30 = memref.subview %subview_17[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_28, %subview_29 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>, 2>) outs(%subview_30 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %c1_21 = arith.constant 1 : index
# CHECK-NEXT:            %0 = arith.muli %c1_13, %c1_21 : index
# CHECK-NEXT:            %1 = arith.addi %c0_11, %0 : index
# CHECK-NEXT:            %subview_22 = memref.subview %subview_8[%1, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_23 = memref.subview %subview_9[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_24 = memref.subview %subview_10[%1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_25 = arith.constant 0 : index
# CHECK-NEXT:            %c16_26 = arith.constant 16 : index
# CHECK-NEXT:            %c1_27 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_25 to %c16_26 step %c1_27 {
# CHECK-NEXT:              %subview_28 = memref.subview %subview_22[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_29 = memref.subview %subview_23[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x1xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:              %subview_30 = memref.subview %subview_24[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_28, %subview_29 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>, 2>) outs(%subview_30 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:          } {"./j"}
# CHECK-NEXT:        } {"./i"}
# CHECK-NEXT:      } {"./k"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
