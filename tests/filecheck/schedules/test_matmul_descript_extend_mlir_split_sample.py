# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend
from xtc.schedules.descript_extend import descript_extend_scheduler

I, J, K, dtype = 16, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()
descript_extend_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_axis=["i", "j", "k"],
    spec={
        "DDR": {
            "j": {},
            "k": {},
            "i[:i_split]": {
                "Rr": {
                    "i#2": {"unroll": None},
                    "j#16": {"vectorize": None},
                },
            },
            "i[i_split:]": {
                "Rl": {
                    "i#2": {"unroll": None},
                    "j#16": {"vectorize": None},
                },
            },
        },
    },
    sample={"i_split": 8},
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_mlir_split_sample",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

#CHECK: // -----// IR Dump Before transform //----- //
#CHECK-NEXT: module attributes {transform.with_named_sequence} {
#CHECK-NEXT:   func.func @matmul(%arg0: memref<16x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<16x32xf32> {llvm.noalias}) {
#CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
#CHECK-NEXT:     linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<16x32xf32>)
#CHECK-NEXT:     linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<16x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<16x32xf32>)
#CHECK-NEXT:     return
#CHECK-NEXT:   }
#CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
#CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
#CHECK-NEXT:     transform.yield 
#CHECK-NEXT:   }
#CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
#CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
#CHECK-NEXT:     %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     %2 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_3 "C/j" : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_5 "C/k" : !transform.any_op
#CHECK-NEXT:     %first, %second = transform.structured.split %tiled_linalg_op_4 after 8  {dimension = 0 : i64} : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %first tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_7 "C/i[0]/i0" : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_9 "C/i[0]/j0" : !transform.any_op
#CHECK-NEXT:     %3 = transform.get_parent_op %loops_7 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %3 {
#CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
#CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %3 {
#CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
#CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     %4 = transform.structured.match attributes {"C/i[0]/i0"} in %3 : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.loop.unroll %loops_7 {factor = 2 : i64} : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %second tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_11 "C/i[1]/i0" : !transform.any_op
#CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_10) : (!transform.any_op) -> ()
#CHECK-NEXT:     %5 = transform.get_parent_op %loops_11 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %5 {
#CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
#CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %5 {
#CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
#CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     %6 = transform.structured.match attributes {"C/i[1]/i0"} in %5 : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.loop.unroll %loops_11 {factor = 2 : i64} : !transform.any_op
#CHECK-NEXT:     %7 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %7 {
#CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
#CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %7 {
#CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
#CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.yield 
#CHECK-NEXT:   }
#CHECK-NEXT: }
#CHECK-NEXT:  
#CHECK-NEXT: // -----// IR Dump After transform //----- //
#CHECK-NEXT: module attributes {transform.with_named_sequence} {
#CHECK-NEXT:   func.func @matmul(%arg0: memref<16x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<16x32xf32> {llvm.noalias}) {
#CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<1x16xf32>
#CHECK-NEXT:     %c4 = arith.constant 4 : index
#CHECK-NEXT:     %c2 = arith.constant 2 : index
#CHECK-NEXT:     %c8 = arith.constant 8 : index
#CHECK-NEXT:     %c512 = arith.constant 512 : index
#CHECK-NEXT:     %c32 = arith.constant 32 : index
#CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
#CHECK-NEXT:     %c0 = arith.constant 0 : index
#CHECK-NEXT:     %c16 = arith.constant 16 : index
#CHECK-NEXT:     %c1 = arith.constant 1 : index
#CHECK-NEXT:     scf.for %arg3 = %c0 to %c16 step %c1 {
#CHECK-NEXT:       %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<16x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:       scf.for %arg4 = %c0 to %c32 step %c1 {
#CHECK-NEXT:         %subview_1 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:         linalg.fill {__xtc_id_C_0_} ins(%cst_0 : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
#CHECK-NEXT:       } {"./j"}
#CHECK-NEXT:     } {"./i"}
#CHECK-NEXT:     scf.for %arg3 = %c0 to %c32 step %c16 {
#CHECK-NEXT:       %subview = memref.subview %arg0[0, 0] [16, 512] [1, 1] : memref<16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
#CHECK-NEXT:       %subview_1 = memref.subview %arg1[0, %arg3] [512, 16] [1, 1] : memref<512x32xf32> to memref<512x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:       %subview_2 = memref.subview %arg2[0, %arg3] [16, 16] [1, 1] : memref<16x32xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:       scf.for %arg4 = %c0 to %c512 step %c1 {
#CHECK-NEXT:         %subview_3 = memref.subview %subview[0, %arg4] [16, 1] [1, 1] : memref<16x512xf32, strided<[512, 1]>> to memref<16x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:         %subview_4 = memref.subview %subview_1[%arg4, 0] [1, 16] [1, 1] : memref<512x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:         %subview_5 = memref.subview %subview_3[0, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[512, 1], offset: ?>> to memref<8x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:         %subview_6 = memref.subview %subview_2[0, 0] [8, 16] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<8x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:         scf.for %arg5 = %c0 to %c8 step %c4 {
#CHECK-NEXT:           %subview_9 = memref.subview %subview_5[%arg5, 0] [2, 1] [1, 1] : memref<8x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_10 = memref.subview %subview_6[%arg5, 0] [2, 16] [1, 1] : memref<8x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           scf.for %arg6 = %c0 to %c16 step %c16 {
#CHECK-NEXT:             linalg.matmul {__xtc_id_C_} ins(%subview_9, %subview_4 : memref<2x1xf32, strided<[512, 1], offset: ?>>, memref<1x16xf32, strided<[32, 1], offset: ?>>) outs(%subview_10 : memref<2x16xf32, strided<[32, 1], offset: ?>>)
#CHECK-NEXT:           } {"C/i[0]/j0"}
#CHECK-NEXT:           %0 = arith.addi %arg5, %c2 : index
#CHECK-NEXT:           %subview_11 = memref.subview %subview_5[%0, 0] [2, 1] [1, 1] : memref<8x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_12 = memref.subview %subview_6[%0, 0] [2, 16] [1, 1] : memref<8x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           scf.for %arg6 = %c0 to %c16 step %c16 {
#CHECK-NEXT:             linalg.matmul {__xtc_id_C_} ins(%subview_11, %subview_4 : memref<2x1xf32, strided<[512, 1], offset: ?>>, memref<1x16xf32, strided<[32, 1], offset: ?>>) outs(%subview_12 : memref<2x16xf32, strided<[32, 1], offset: ?>>)
#CHECK-NEXT:           } {"C/i[0]/j0"}
#CHECK-NEXT:         } {"C/i[0]/i0"}
#CHECK-NEXT:         %subview_7 = memref.subview %subview_3[8, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[512, 1], offset: ?>> to memref<8x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:         %subview_8 = memref.subview %subview_2[8, 0] [8, 16] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<8x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:         scf.for %arg5 = %c0 to %c8 step %c2 {
#CHECK-NEXT:           %subview_9 = memref.subview %subview_7[%arg5, 0] [1, 1] [1, 1] : memref<8x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_10 = memref.subview %subview_8[%arg5, 0] [1, 16] [1, 1] : memref<8x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           %0 = vector.transfer_read %subview_9[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:           %1 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
#CHECK-NEXT:           %2 = vector.transfer_read %subview_10[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
#CHECK-NEXT:           %3 = vector.extract %1[0] : vector<16xf32> from vector<1x16xf32>
#CHECK-NEXT:           %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:           %5 = vector.broadcast %4 : f32 to vector<16xf32>
#CHECK-NEXT:           %6 = vector.extract %2[0] : vector<16xf32> from vector<1x16xf32>
#CHECK-NEXT:           %7 = vector.fma %5, %3, %6 : vector<16xf32>
#CHECK-NEXT:           %8 = vector.insert %7, %cst [0] : vector<16xf32> into vector<1x16xf32>
#CHECK-NEXT:           vector.transfer_write %8, %subview_10[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           %9 = arith.addi %arg5, %c1 : index
#CHECK-NEXT:           %subview_11 = memref.subview %subview_7[%9, 0] [1, 1] [1, 1] : memref<8x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_12 = memref.subview %subview_8[%9, 0] [1, 16] [1, 1] : memref<8x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           %10 = vector.transfer_read %subview_11[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:           %11 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
#CHECK-NEXT:           %12 = vector.transfer_read %subview_12[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
#CHECK-NEXT:           %13 = vector.extract %11[0] : vector<16xf32> from vector<1x16xf32>
#CHECK-NEXT:           %14 = vector.extract %10[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:           %15 = vector.broadcast %14 : f32 to vector<16xf32>
#CHECK-NEXT:           %16 = vector.extract %12[0] : vector<16xf32> from vector<1x16xf32>
#CHECK-NEXT:           %17 = vector.fma %15, %13, %16 : vector<16xf32>
#CHECK-NEXT:           %18 = vector.insert %17, %cst [0] : vector<16xf32> into vector<1x16xf32>
#CHECK-NEXT:           vector.transfer_write %18, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:         } {"C/i[1]/i0"}
#CHECK-NEXT:       } {"C/k"}
#CHECK-NEXT:     } {"C/j"}
#CHECK-NEXT:     return
#CHECK-NEXT:   }
#CHECK-NEXT: }
#CHECK-NEXT:  
#CHECK-NEXT: graph:
#CHECK-NEXT:   name: matmul
#CHECK-NEXT:   inputs:
#CHECK-NEXT:   - %0 : 16x512xfloat32
#CHECK-NEXT:   - %1 : 512x32xfloat32
#CHECK-NEXT:   outputs:
#CHECK-NEXT:   - %2 : 16x32xfloat32
#CHECK-NEXT:   nodes:
#CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'C'} : [16x512xfloat32, 512x32xfloat32] -> [16x32xfloat32]
#CHECK-NEXT:  
#CHECK-NEXT: CODE: 0
