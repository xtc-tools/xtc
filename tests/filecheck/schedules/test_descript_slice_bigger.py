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
    abstract_axis=["i", "j", "k"],
    spec={
        'k': {},
        'j': {},
        'i[:32]': {
            'i#32': {},
            'k#32': {},
            'j#16': {'vectorize': True},
        },
        'i[32:]': {
            'i#18': {},
            'k#32': {},
            'j#16': {'vectorize': True},
        }
    }
)

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_slice_first_bigger",
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
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "C/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "C/j" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.split %tiled_linalg_op_4 after 32  {dimension = 0 : i64} : !transform.any_op
# CHECK-NEXT:      %3:2 = transform.split_handle %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %3#0 tile_sizes [32, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/i[0]/i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "C/i[0]/i0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "C/i[0]/k0" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_10) : (!transform.any_op) -> ()
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %3#1 tile_sizes [18, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "C/i[1]/i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "C/i[1]/i0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "C/i[1]/k0" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op_16) : (!transform.any_op) -> ()
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<50x64xf32> {llvm.noalias}, %arg1: memref<64x64xf32> {llvm.noalias}, %arg2: memref<50x64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<1x16xf32>
# CHECK-NEXT:      %c18 = arith.constant 18 : index
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c50 = arith.constant 50 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c50 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<50x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c64 step %c1 {
# CHECK-NEXT:          %subview_1 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst_0 : f32) outs(%subview_1 : memref<1x1xf32, strided<[64, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c64 step %c32 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [50, 32] [1, 1] : memref<50x64xf32> to memref<50x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg1[%arg3, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg2[0, 0] [50, 64] [1, 1] : memref<50x64xf32> to memref<50x64xf32, strided<[64, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c64 step %c16 {
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4] [32, 16] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %subview_2[0, %arg4] [50, 16] [1, 1] : memref<50x64xf32, strided<[64, 1]>> to memref<50x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview[0, 0] [32, 32] [1, 1] : memref<50x32xf32, strided<[64, 1], offset: ?>> to memref<32x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %subview_4[0, 0] [32, 16] [1, 1] : memref<50x16xf32, strided<[64, 1], offset: ?>> to memref<32x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c32 {
# CHECK-NEXT:            %subview_9 = memref.subview %subview_5[%arg5, 0] [32, 32] [1, 1] : memref<32x32xf32, strided<[64, 1], offset: ?>> to memref<32x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_6[%arg5, 0] [32, 16] [1, 1] : memref<32x16xf32, strided<[64, 1], offset: ?>> to memref<32x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c32 step %c1 {
# CHECK-NEXT:              %subview_11 = memref.subview %subview_9[%arg6, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[64, 1], offset: ?>> to memref<1x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[%arg6, 0] [1, 16] [1, 1] : memref<32x16xf32, strided<[64, 1], offset: ?>> to memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              scf.for %arg7 = %c0 to %c32 step %c1 {
# CHECK-NEXT:                %subview_13 = memref.subview %subview_11[0, %arg7] [1, 1] [1, 1] : memref<1x32xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %subview_14 = memref.subview %subview_3[%arg7, 0] [1, 16] [1, 1] : memref<32x16xf32, strided<[64, 1], offset: ?>> to memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %1 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[64, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:                %2 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[64, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %3 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[64, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %4 = vector.extract %2[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:                %6 = vector.broadcast %5 : f32 to vector<16xf32>
# CHECK-NEXT:                %7 = vector.extract %3[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %8 = vector.fma %6, %4, %7 : vector<16xf32>
# CHECK-NEXT:                %9 = vector.insert %8, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:                vector.transfer_write %9, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              } {"C/i[0]/k0"}
# CHECK-NEXT:            } {"C/i[0]/i0"}
# CHECK-NEXT:          } {"C/i[0]/i"}
# CHECK-NEXT:          %subview_7 = memref.subview %subview[32, 0] [18, 32] [1, 1] : memref<50x32xf32, strided<[64, 1], offset: ?>> to memref<18x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_4[32, 0] [18, 16] [1, 1] : memref<50x16xf32, strided<[64, 1], offset: ?>> to memref<18x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c18 step %c18 {
# CHECK-NEXT:            %subview_9 = memref.subview %subview_7[%arg5, 0] [18, 32] [1, 1] : memref<18x32xf32, strided<[64, 1], offset: ?>> to memref<18x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_8[%arg5, 0] [18, 16] [1, 1] : memref<18x16xf32, strided<[64, 1], offset: ?>> to memref<18x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            scf.for %arg6 = %c0 to %c18 step %c1 {
# CHECK-NEXT:              %subview_11 = memref.subview %subview_9[%arg6, 0] [1, 32] [1, 1] : memref<18x32xf32, strided<[64, 1], offset: ?>> to memref<1x32xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[%arg6, 0] [1, 16] [1, 1] : memref<18x16xf32, strided<[64, 1], offset: ?>> to memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              scf.for %arg7 = %c0 to %c32 step %c1 {
# CHECK-NEXT:                %subview_13 = memref.subview %subview_11[0, %arg7] [1, 1] [1, 1] : memref<1x32xf32, strided<[64, 1], offset: ?>> to memref<1x1xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %subview_14 = memref.subview %subview_3[%arg7, 0] [1, 16] [1, 1] : memref<32x16xf32, strided<[64, 1], offset: ?>> to memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:                %1 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[64, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:                %2 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[64, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %3 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[64, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:                %4 = vector.extract %2[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:                %6 = vector.broadcast %5 : f32 to vector<16xf32>
# CHECK-NEXT:                %7 = vector.extract %3[0] : vector<16xf32> from vector<1x16xf32>
# CHECK-NEXT:                %8 = vector.fma %6, %4, %7 : vector<16xf32>
# CHECK-NEXT:                %9 = vector.insert %8, %cst [0] : vector<16xf32> into vector<1x16xf32>
# CHECK-NEXT:                vector.transfer_write %9, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:              } {"C/i[1]/k0"}
# CHECK-NEXT:            } {"C/i[1]/i0"}
# CHECK-NEXT:          } {"C/i[1]/i"}
# CHECK-NEXT:        } {"C/j"}
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
