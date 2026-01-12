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
axes_sizes = {"i": I, "j": J, "k": K}
descript_extend_scheduler(
    scheduler=sch,
    node_name="C",
    abstract_axis=["i", "j", "k"],
    abstract_axis_sizes=axes_sizes,
    spec={
        "DDR": {
            "j": {},
            "k": {},
        },
        "L2": {
            "j#jDDR": {},
            "i[:4]": {
                "R": {
                    "i#iR1": {"unroll": None},
                    "j#jR": {"vectorize": None},
                },
            },
            "i[4:]": {
                "R": {
                    "i#iR2": {"unroll": None},
                    "j#jR": {"vectorize": None},
                },
            },
        },
    },
    sample={"jDDR": 16, "jR": 4, "iR1": 2, "iR2": 4},
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_mlir_split",
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
#CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_3 "C/j" : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_5 "C/k" : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_7 "C/j0" : !transform.any_op
#CHECK-NEXT:     %2 = transform.structured.split %tiled_linalg_op_6 after 4  {dimension = 0 : i64} : !transform.any_op
#CHECK-NEXT:     %3:2 = transform.split_handle %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %3#0 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_9 "C/i[0]/i0" : !transform.any_op
#CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_8) : (!transform.any_op) -> ()
#CHECK-NEXT:     transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
#CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %3#1 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
#CHECK-NEXT:     transform.annotate %loops_11 "C/i[1]/i0" : !transform.any_op
#CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_10) : (!transform.any_op) -> ()
#CHECK-NEXT:     transform.loop.unroll %loops_11 {factor = 4 : i64} : !transform.any_op
#CHECK-NEXT:     %4 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %4 {
#CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
#CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.apply_patterns to %4 {
#CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
#CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
#CHECK-NEXT:     } : !transform.any_op
#CHECK-NEXT:     transform.yield 
#CHECK-NEXT:   }
#CHECK-NEXT: }
#CHECK-EMPTY:
#CHECK-NEXT: // -----// IR Dump After transform //----- //
#CHECK-NEXT: module attributes {transform.with_named_sequence} {
#CHECK-NEXT:   func.func @matmul(%arg0: memref<16x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<16x32xf32> {llvm.noalias}) {
#CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<1x4xf32>
#CHECK-NEXT:     %c3 = arith.constant 3 : index
#CHECK-NEXT:     %c12 = arith.constant 12 : index
#CHECK-NEXT:     %0 = ub.poison : f32
#CHECK-NEXT:     %c2 = arith.constant 2 : index
#CHECK-NEXT:     %c4 = arith.constant 4 : index
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
#CHECK-NEXT:         scf.for %arg5 = %c0 to %c16 step %c4 {
#CHECK-NEXT:           %subview_5 = memref.subview %subview_4[0, %arg5] [1, 4] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           %subview_6 = memref.subview %subview_2[0, %arg5] [16, 4] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<16x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           %subview_7 = memref.subview %subview_3[0, 0] [4, 1] [1, 1] : memref<16x1xf32, strided<[512, 1], offset: ?>> to memref<4x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_8 = memref.subview %subview_6[0, 0] [4, 4] [1, 1] : memref<16x4xf32, strided<[32, 1], offset: ?>> to memref<4x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           scf.for %arg6 = %c0 to %c4 step %c2 {
#CHECK-NEXT:             %subview_11 = memref.subview %subview_7[%arg6, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_12 = memref.subview %subview_8[%arg6, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %1 = vector.transfer_read %subview_11[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %2 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %3 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %4 = vector.extract %2[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %6 = vector.broadcast %5 : f32 to vector<4xf32>
#CHECK-NEXT:             %7 = vector.extract %3[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %8 = vector.fma %6, %4, %7 : vector<4xf32>
#CHECK-NEXT:             %9 = vector.insert %8, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %9, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %10 = arith.addi %arg6, %c1 : index
#CHECK-NEXT:             %subview_13 = memref.subview %subview_7[%10, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_14 = memref.subview %subview_8[%10, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %11 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %12 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %13 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %14 = vector.extract %12[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %15 = vector.extract %11[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %16 = vector.broadcast %15 : f32 to vector<4xf32>
#CHECK-NEXT:             %17 = vector.extract %13[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %18 = vector.fma %16, %14, %17 : vector<4xf32>
#CHECK-NEXT:             %19 = vector.insert %18, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %19, %subview_14[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           } {"C/i[0]/i0"}
#CHECK-NEXT:           %subview_9 = memref.subview %subview_3[4, 0] [12, 1] [1, 1] : memref<16x1xf32, strided<[512, 1], offset: ?>> to memref<12x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:           %subview_10 = memref.subview %subview_6[4, 0] [12, 4] [1, 1] : memref<16x4xf32, strided<[32, 1], offset: ?>> to memref<12x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           scf.for %arg6 = %c0 to %c12 step %c4 {
#CHECK-NEXT:             %subview_11 = memref.subview %subview_9[%arg6, 0] [1, 1] [1, 1] : memref<12x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_12 = memref.subview %subview_10[%arg6, 0] [1, 4] [1, 1] : memref<12x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %1 = vector.transfer_read %subview_11[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %2 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %3 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %4 = vector.extract %2[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %6 = vector.broadcast %5 : f32 to vector<4xf32>
#CHECK-NEXT:             %7 = vector.extract %3[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %8 = vector.fma %6, %4, %7 : vector<4xf32>
#CHECK-NEXT:             %9 = vector.insert %8, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %9, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %10 = arith.addi %arg6, %c1 : index
#CHECK-NEXT:             %subview_13 = memref.subview %subview_9[%10, 0] [1, 1] [1, 1] : memref<12x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_14 = memref.subview %subview_10[%10, 0] [1, 4] [1, 1] : memref<12x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %11 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %12 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %13 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %14 = vector.extract %12[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %15 = vector.extract %11[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %16 = vector.broadcast %15 : f32 to vector<4xf32>
#CHECK-NEXT:             %17 = vector.extract %13[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %18 = vector.fma %16, %14, %17 : vector<4xf32>
#CHECK-NEXT:             %19 = vector.insert %18, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %19, %subview_14[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %20 = arith.addi %arg6, %c2 : index
#CHECK-NEXT:             %subview_15 = memref.subview %subview_9[%20, 0] [1, 1] [1, 1] : memref<12x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_16 = memref.subview %subview_10[%20, 0] [1, 4] [1, 1] : memref<12x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %21 = vector.transfer_read %subview_15[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %22 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %23 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %24 = vector.extract %22[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %25 = vector.extract %21[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %26 = vector.broadcast %25 : f32 to vector<4xf32>
#CHECK-NEXT:             %27 = vector.extract %23[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %28 = vector.fma %26, %24, %27 : vector<4xf32>
#CHECK-NEXT:             %29 = vector.insert %28, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %29, %subview_16[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %30 = arith.addi %arg6, %c3 : index
#CHECK-NEXT:             %subview_17 = memref.subview %subview_9[%30, 0] [1, 1] [1, 1] : memref<12x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
#CHECK-NEXT:             %subview_18 = memref.subview %subview_10[%30, 0] [1, 4] [1, 1] : memref<12x4xf32, strided<[32, 1], offset: ?>> to memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:             %31 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
#CHECK-NEXT:             %32 = vector.transfer_read %subview_5[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %33 = vector.transfer_read %subview_18[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x4xf32, strided<[32, 1], offset: ?>>, vector<1x4xf32>
#CHECK-NEXT:             %34 = vector.extract %32[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %35 = vector.extract %31[0, 0] : f32 from vector<1x1xf32>
#CHECK-NEXT:             %36 = vector.broadcast %35 : f32 to vector<4xf32>
#CHECK-NEXT:             %37 = vector.extract %33[0] : vector<4xf32> from vector<1x4xf32>
#CHECK-NEXT:             %38 = vector.fma %36, %34, %37 : vector<4xf32>
#CHECK-NEXT:             %39 = vector.insert %38, %cst [0] : vector<4xf32> into vector<1x4xf32>
#CHECK-NEXT:             vector.transfer_write %39, %subview_18[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, strided<[32, 1], offset: ?>>
#CHECK-NEXT:           } {"C/i[1]/i0"}
#CHECK-NEXT:         } {"C/j0"}
#CHECK-NEXT:       } {"C/k"}
#CHECK-NEXT:     } {"C/j"}
#CHECK-NEXT:     return
#CHECK-NEXT:   }
#CHECK-NEXT: }
#CHECK-EMPTY:
#CHECK-NEXT: graph:
#CHECK-NEXT:   name: matmul
#CHECK-NEXT:   inputs:
#CHECK-NEXT:   - %0 : 16x512xfloat32
#CHECK-NEXT:   - %1 : 512x32xfloat32
#CHECK-NEXT:   outputs:
#CHECK-NEXT:   - %2 : 16x32xfloat32
#CHECK-NEXT:   nodes:
#CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'C'} : [16x512xfloat32, 512x32xfloat32] -> [16x32xfloat32]
#CHECK-EMPTY:
#CHECK-NEXT: CODE: 0

