# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

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
sch.parallelize(["i", "j"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir_parallel",
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %tiled_linalg_op tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_op_0, %forall_op_1 = transform.structured.tile_using_forall %tiled_op tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_1 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_op_0 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j1" : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_3 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 16)>
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
# CHECK-NEXT:        scf.forall (%arg4) in (2) {
# CHECK-NEXT:          %0 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_2 = memref.subview %subview[%0, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_0[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %subview_1[%0, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.forall (%arg5) in (2) {
# CHECK-NEXT:            %1 = affine.apply #map1(%arg5)
# CHECK-NEXT:            %subview_5 = memref.subview %subview_2[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %subview_3[0, %1] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_4[0, %1] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_8 = arith.constant 0 : index
# CHECK-NEXT:            %c2 = arith.constant 2 : index
# CHECK-NEXT:            %c1_9 = arith.constant 1 : index
# CHECK-NEXT:            %c2_10 = arith.constant 2 : index
# CHECK-NEXT:            %subview_11 = memref.subview %subview_5[%c0_8, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_6[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_7[%c0_8, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %c16 = arith.constant 16 : index
# CHECK-NEXT:            %c1_15 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_14 to %c16 step %c1_15 {
# CHECK-NEXT:              %subview_23 = memref.subview %subview_11[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_24 = memref.subview %subview_12[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %subview_25 = memref.subview %subview_13[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_23, %subview_24 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%subview_25 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %c1_16 = arith.constant 1 : index
# CHECK-NEXT:            %2 = arith.muli %c1_9, %c1_16 : index
# CHECK-NEXT:            %3 = arith.addi %c0_8, %2 : index
# CHECK-NEXT:            %subview_17 = memref.subview %subview_5[%3, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_18 = memref.subview %subview_6[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %subview_7[%3, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_20 = arith.constant 0 : index
# CHECK-NEXT:            %c16_21 = arith.constant 16 : index
# CHECK-NEXT:            %c1_22 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_20 to %c16_21 step %c1_22 {
# CHECK-NEXT:              %subview_23 = memref.subview %subview_17[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_24 = memref.subview %subview_18[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %subview_25 = memref.subview %subview_19[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_23, %subview_24 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%subview_25 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
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
