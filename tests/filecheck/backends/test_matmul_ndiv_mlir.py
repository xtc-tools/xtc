# RUN: python %s 2>&1 | filecheck %s

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
sch.tile("i", {"i1": 3}) # non-divisible tile
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})    # non-full/non-divisible unrolled
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_ndiv_mlir",
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [3, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i1" : !transform.any_op
# CHECK-NEXT:      %3 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_8) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %4 = transform.structured.match attributes {"./i1"} in %3 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (-d0 + 4, 3)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<1x16xf32>
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c32 step %c1 {
# CHECK-NEXT:          %subview_1 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst_0 : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c4 step %c3 {
# CHECK-NEXT:          %1 = affine.min #map(%arg4)
# CHECK-NEXT:          %subview_3 = memref.subview %subview[%arg4, 0] [%1, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<?x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %subview_2[%arg4, 0] [%1, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<?x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c16 {
# CHECK-NEXT:            %subview_5 = memref.subview %subview_1[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %subview_4[0, %arg5] [%1, 16] [1, 1] : memref<?x32xf32, strided<[32, 1], offset: ?>> to memref<?x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %2 = arith.subi %1, %c0 : index
# CHECK-NEXT:            %c1_7 = arith.constant 1 : index
# CHECK-NEXT:            %3 = arith.subi %c1, %c1_7 : index
# CHECK-NEXT:            %4 = arith.addi %2, %3 : index
# CHECK-NEXT:            %5 = arith.divui %4, %c1 : index
# CHECK-NEXT:            %c2 = arith.constant 2 : index
# CHECK-NEXT:            %6 = arith.remsi %5, %c2 : index
# CHECK-NEXT:            %7 = arith.subi %5, %6 : index
# CHECK-NEXT:            %8 = arith.muli %7, %c1 : index
# CHECK-NEXT:            %9 = arith.addi %c0, %8 : index
# CHECK-NEXT:            %10 = arith.muli %c1, %c2 : index
# CHECK-NEXT:            scf.for %arg6 = %c0 to %9 step %10 {
# CHECK-NEXT:              %subview_8 = memref.subview %subview_3[%arg6, 0] [1, 1] [1, 1] : memref<?x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_9 = memref.subview %subview_6[%arg6, 0] [1, 16] [1, 1] : memref<?x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %11 = vector.transfer_read %subview_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:              %12 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %13 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %14 = vector.extract %12[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %15 = vector.extract %11[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:              %16 = vector.broadcast %15 : f32 to vector<16xf32>
# CHECK-NEXT:              %17 = vector.extract %13[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %18 = vector.fma %16, %14, %17 : vector<16xf32>
# CHECK-NEXT:              %19 = vector.insert %18, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:              vector.transfer_write %19, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %c1_10 = arith.constant 1 : index
# CHECK-NEXT:              %20 = arith.muli %c1, %c1_10 : index
# CHECK-NEXT:              %21 = arith.addi %arg6, %20 : index
# CHECK-NEXT:              %subview_11 = memref.subview %subview_3[%21, 0] [1, 1] [1, 1] : memref<?x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_12 = memref.subview %subview_6[%21, 0] [1, 16] [1, 1] : memref<?x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %22 = vector.transfer_read %subview_11[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:              %23 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %24 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %25 = vector.extract %23[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %26 = vector.extract %22[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:              %27 = vector.broadcast %26 : f32 to vector<16xf32>
# CHECK-NEXT:              %28 = vector.extract %24[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %29 = vector.fma %27, %25, %28 : vector<16xf32>
# CHECK-NEXT:              %30 = vector.insert %29, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:              vector.transfer_write %30, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            } {"./i1"}
# CHECK-NEXT:            scf.for %arg6 = %9 to %1 step %c1 {
# CHECK-NEXT:              %subview_8 = memref.subview %subview_3[%arg6, 0] [1, 1] [1, 1] : memref<?x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_9 = memref.subview %subview_6[%arg6, 0] [1, 16] [1, 1] : memref<?x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              %11 = vector.transfer_read %subview_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:              %12 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %13 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:              %14 = vector.extract %12[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %15 = vector.extract %11[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:              %16 = vector.broadcast %15 : f32 to vector<16xf32>
# CHECK-NEXT:              %17 = vector.extract %13[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:              %18 = vector.fma %16, %14, %17 : vector<16xf32>
# CHECK-NEXT:              %19 = vector.insert %18, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:              vector.transfer_write %19, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            } {"./i1"}
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
