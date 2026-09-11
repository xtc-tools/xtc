# RUN: python -O %s 2>&1 | filecheck %s
# REQUIRES: module_tvm
# REQUIRES: module_xvs

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.search.strategies import Strategy_Descript as Strategy

I, J, K, dtype = 8, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler()

spec = """
T: parallelize
W:
U:
O:
P: unroll vectorize
"""

strategy = Strategy(graph, spec)

sample = {
    "prt_j_0": 32,
    "prt_j_1": 16,
    "prt_j_2": 2,
    "prt_i_0": 8,
    "prt_i_1": 4,
    "prt_i_2": 2,
    "prt_k_0": 8,
    "prt_k_1": 2,
    "prt_interchange_u_0": 2,
}

strategy.generate(sch, sample)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_extend_twuop",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: memref<8x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<8x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<8x32xf32>)
# CHECK-NEXT:     linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<8x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<8x32xf32>)
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:     %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [8, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %forall_op "./i" : !transform.any_op
# CHECK-NEXT:     %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [0, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %forall_op_3 "./j" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op_2 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_5 "./k" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [4, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_7 "./i0" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 0, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_9 "./k0" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_11 "./j0" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_13 "./i1" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_15 "./k1" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 2, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_17 "./j1" : !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops_19 "./i2" : !transform.any_op
# CHECK-NEXT:     transform.include @_vecto failures(suppress) (%tiled_linalg_op_18) : (!transform.any_op) -> ()
# CHECK-NEXT:     transform.loop.unroll %loops_19 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:     %2 = transform.get_parent_op %forall_op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %2 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-EMPTY:
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0 * 8)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 * 32)>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @matmul(%arg0: memref<8x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<8x32xf32> {llvm.noalias}) {
# CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<1x2xf32>
# CHECK-NEXT:     %0 = ub.poison : f32
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c2 = arith.constant 2 : index
# CHECK-NEXT:     %c4 = arith.constant 4 : index
# CHECK-NEXT:     %c512 = arith.constant 512 : index
# CHECK-NEXT:     %c32 = arith.constant 32 : index
# CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %c8 = arith.constant 8 : index
# CHECK-NEXT:     %c1 = arith.constant 1 : index
# CHECK-NEXT:     scf.for %arg3 = %c0 to %c8 step %c1 {
# CHECK-NEXT:       %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<8x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.for %arg4 = %c0 to %c32 step %c1 {
# CHECK-NEXT:         %subview_1 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         linalg.fill {__xtc_id_C_0_} ins(%cst_0 : f32) outs(%subview_1 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     scf.forall (%arg3) in (1) {
# CHECK-NEXT:       %1 = affine.apply #map(%arg3)
# CHECK-NEXT:       %subview = memref.subview %arg0[%1, 0] [8, 512] [1, 1] : memref<8x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:       %subview_1 = memref.subview %arg1[0, 0] [512, 32] [1, 1] : memref<512x32xf32> to memref<512x32xf32, strided<[32, 1]>>
# CHECK-NEXT:       %subview_2 = memref.subview %arg2[%1, 0] [8, 32] [1, 1] : memref<8x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:       scf.forall (%arg4) in (1) {
# CHECK-NEXT:         %2 = affine.apply #map1(%arg4)
# CHECK-NEXT:         %subview_3 = memref.subview %subview_1[0, %2] [512, 32] [1, 1] : memref<512x32xf32, strided<[32, 1]>> to memref<512x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         %subview_4 = memref.subview %subview_2[0, %2] [8, 32] [1, 1] : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:         scf.for %arg5 = %c0 to %c512 step %c8 {
# CHECK-NEXT:           %subview_5 = memref.subview %subview[0, %arg5] [8, 8] [1, 1] : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x8xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:           %subview_6 = memref.subview %subview_3[%arg5, 0] [8, 32] [1, 1] : memref<512x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:           scf.for %arg6 = %c0 to %c8 step %c4 {
# CHECK-NEXT:             %subview_7 = memref.subview %subview_5[%arg6, 0] [4, 8] [1, 1] : memref<8x8xf32, strided<[512, 1], offset: ?>> to memref<4x8xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:             %subview_8 = memref.subview %subview_4[%arg6, 0] [4, 32] [1, 1] : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<4x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:             scf.for %arg7 = %c0 to %c8 step %c2 {
# CHECK-NEXT:               %subview_9 = memref.subview %subview_7[0, %arg7] [4, 2] [1, 1] : memref<4x8xf32, strided<[512, 1], offset: ?>> to memref<4x2xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:               %subview_10 = memref.subview %subview_6[%arg7, 0] [2, 32] [1, 1] : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:               scf.for %arg8 = %c0 to %c32 step %c16 {
# CHECK-NEXT:                 %subview_11 = memref.subview %subview_10[0, %arg8] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                 %subview_12 = memref.subview %subview_8[0, %arg8] [4, 16] [1, 1] : memref<4x32xf32, strided<[32, 1], offset: ?>> to memref<4x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                 scf.for %arg9 = %c0 to %c4 step %c2 {
# CHECK-NEXT:                   %subview_13 = memref.subview %subview_9[%arg9, 0] [2, 2] [1, 1] : memref<4x2xf32, strided<[512, 1], offset: ?>> to memref<2x2xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                   %subview_14 = memref.subview %subview_12[%arg9, 0] [2, 16] [1, 1] : memref<4x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                   scf.for %arg10 = %c0 to %c2 step %c1 {
# CHECK-NEXT:                     %subview_15 = memref.subview %subview_13[0, %arg10] [2, 1] [1, 1] : memref<2x2xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                     %subview_16 = memref.subview %subview_11[%arg10, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                     scf.for %arg11 = %c0 to %c16 step %c2 {
# CHECK-NEXT:                       %subview_17 = memref.subview %subview_16[0, %arg11] [1, 2] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                       %subview_18 = memref.subview %subview_14[0, %arg11] [2, 2] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<2x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                       %subview_19 = memref.subview %subview_15[%c0, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                       %subview_20 = memref.subview %subview_18[%c0, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[32, 1], offset: ?>> to memref<1x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                       %3 = vector.transfer_read %subview_19[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:                       %4 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x2xf32, strided<[32, 1], offset: ?>>, vector<1x2xf32>
# CHECK-NEXT:                       %5 = vector.transfer_read %subview_20[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x2xf32, strided<[32, 1], offset: ?>>, vector<1x2xf32>
# CHECK-NEXT:                       %6 = vector.extract %4[0] : vector<2xf32> from vector<1x2xf32>
# CHECK-NEXT:                       %7 = vector.extract %3[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:                       %8 = vector.broadcast %7 : f32 to vector<2xf32>
# CHECK-NEXT:                       %9 = vector.extract %5[0] : vector<2xf32> from vector<1x2xf32>
# CHECK-NEXT:                       %10 = vector.fma %8, %6, %9 : vector<2xf32>
# CHECK-NEXT:                       %11 = vector.insert %10, %cst [0] : vector<2xf32> into vector<1x2xf32>
# CHECK-NEXT:                       vector.transfer_write %11, %subview_20[%c0, %c0] {in_bounds = [true, true]} : vector<1x2xf32>, memref<1x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                       %subview_21 = memref.subview %subview_15[%c1, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:                       %subview_22 = memref.subview %subview_18[%c1, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[32, 1], offset: ?>> to memref<1x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                       %12 = vector.transfer_read %subview_21[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
# CHECK-NEXT:                       %13 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x2xf32, strided<[32, 1], offset: ?>>, vector<1x2xf32>
# CHECK-NEXT:                       %14 = vector.transfer_read %subview_22[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x2xf32, strided<[32, 1], offset: ?>>, vector<1x2xf32>
# CHECK-NEXT:                       %15 = vector.extract %13[0] : vector<2xf32> from vector<1x2xf32>
# CHECK-NEXT:                       %16 = vector.extract %12[0, 0] : f32 from vector<1x1xf32>
# CHECK-NEXT:                       %17 = vector.broadcast %16 : f32 to vector<2xf32>
# CHECK-NEXT:                       %18 = vector.extract %14[0] : vector<2xf32> from vector<1x2xf32>
# CHECK-NEXT:                       %19 = vector.fma %17, %15, %18 : vector<2xf32>
# CHECK-NEXT:                       %20 = vector.insert %19, %cst [0] : vector<2xf32> into vector<1x2xf32>
# CHECK-NEXT:                       vector.transfer_write %20, %subview_22[%c0, %c0] {in_bounds = [true, true]} : vector<1x2xf32>, memref<1x2xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:                     } {"./j1"}
# CHECK-NEXT:                   } {"./k1"}
# CHECK-NEXT:                 } {"./i1"}
# CHECK-NEXT:               } {"./j0"}
# CHECK-NEXT:             } {"./k0"}
# CHECK-NEXT:           } {"./i0"}
# CHECK-NEXT:         } {"./k"}
# CHECK-NEXT:       } {"./j"}
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-EMPTY:
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: matmul
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 8x512xfloat32
# CHECK-NEXT:   - %1 : 512x32xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %2 : 8x32xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %2: matmul(%0, %1) {name = 'C'} : [8x512xfloat32, 512x32xfloat32] -> [8x32xfloat32]
# CHECK-EMPTY:
# CHECK-NEXT: CODE: 0
