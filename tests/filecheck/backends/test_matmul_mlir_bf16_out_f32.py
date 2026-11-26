# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=c

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 256, "bfloat16"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", out_dtype="float32")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.strip_mine("i", {"i1": 2})
sch.strip_mine("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    print_source_ir=True,
    print_transformed_ir=True,
    dump_file="matmul_bf16_out_f32_mlir",
)
module = comp.compile(sched)

executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")


# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x256xbf16> {llvm.noalias}, %arg1: memref<256x32xbf16> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x256xbf16>, memref<256x32xbf16>) outs(%arg2 : memref<4x32xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
# CHECK-NEXT:        %0 = arith.extf %in : bf16 to f32
# CHECK-NEXT:        %1 = arith.extf %in_0 : bf16 to f32
# CHECK-NEXT:        %2 = arith.mulf %0, %1 : f32
# CHECK-NEXT:        %3 = arith.addf %out, %2 : f32
# CHECK-NEXT:        linalg.yield %3 : f32
# CHECK-NEXT:      }
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
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %tiled_linalg_op_8 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.xtc.fold_unit_extent_dims_via_slices_for_vectorization
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match interface{LinalgOp} in %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%3) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.fold_arith_extension
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
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x256xbf16> {llvm.noalias}, %arg1: memref<256x32xbf16> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %1 = ub.poison : bf16
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c32 step %c1 {
# CHECK-NEXT:          %subview_0 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_0 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x256xbf16> to memref<4x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<256x32xbf16> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c4 step %c2 {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xbf16, strided<[256, 1], offset: ?>> to memref<2x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c16 {
# CHECK-NEXT:            %subview_4 = memref.subview %subview_0[0, %arg5] [1, 16] [1, 1] : memref<1x32xbf16, strided<[32, 1], offset: ?>> to memref<1x16xbf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_5 = memref.subview %subview_3[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %subview_2[%c0, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[256, 1], offset: ?>> to memref<1x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_5[%c0, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_8 = memref.subview %subview_6[0, 0] [1, 1] [1, 1] : memref<1x1xbf16, strided<[256, 1], offset: ?>> to memref<1xbf16, strided<[256], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_7[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %2 = vector.transfer_read %subview_8[%c0], %1 {in_bounds = [true]} : memref<1xbf16, strided<[256], offset: ?>>, vector<1xbf16>
# CHECK-NEXT:            %3 = vector.transfer_read %subview_4[%c0, %c0], %1 {in_bounds = [true, true]} : memref<1x16xbf16, strided<[32, 1], offset: ?>>, vector<1x16xbf16>
# CHECK-NEXT:            %4 = vector.transfer_read %subview_9[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:            %5 = vector.extract %3[0] : vector<16xbf16> from vector<1x16xbf16>
# CHECK-NEXT:            %6 = vector.extract %2[0] : bf16 from vector<1xbf16>
# CHECK-NEXT:            %7 = arith.extf %5 : vector<16xbf16> to vector<16xf32>
# CHECK-NEXT:            %8 = arith.extf %6 : bf16 to f32
# CHECK-NEXT:            %9 = vector.broadcast %8 : f32 to vector<16xf32>
# CHECK-NEXT:            %10 = vector.fma %7, %9, %4 : vector<16xf32>
# CHECK-NEXT:            vector.transfer_write %10, %subview_9[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_2[%c1, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[256, 1], offset: ?>> to memref<1x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_5[%c1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xbf16, strided<[256, 1], offset: ?>> to memref<1xbf16, strided<[256], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_11[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %11 = vector.transfer_read %subview_12[%c0], %1 {in_bounds = [true]} : memref<1xbf16, strided<[256], offset: ?>>, vector<1xbf16>
# CHECK-NEXT:            %12 = vector.transfer_read %subview_4[%c0, %c0], %1 {in_bounds = [true, true]} : memref<1x16xbf16, strided<[32, 1], offset: ?>>, vector<1x16xbf16>
# CHECK-NEXT:            %13 = vector.transfer_read %subview_13[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:            %14 = vector.extract %12[0] : vector<16xbf16> from vector<1x16xbf16>
# CHECK-NEXT:            %15 = vector.extract %11[0] : bf16 from vector<1xbf16>
# CHECK-NEXT:            %16 = arith.extf %14 : vector<16xbf16> to vector<16xf32>
# CHECK-NEXT:            %17 = arith.extf %15 : bf16 to f32
# CHECK-NEXT:            %18 = vector.broadcast %17 : f32 to vector<16xf32>
# CHECK-NEXT:            %19 = vector.fma %16, %18, %13 : vector<16xf32>
# CHECK-NEXT:            vector.transfer_write %19, %subview_13[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
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
# CHECK-NEXT:    - %0 : 4x256xbfloat16
# CHECK-NEXT:    - %1 : 256x32xbfloat16
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1, out_dtype=float32) {name = 'C'} : [4x256xbfloat16, 256x32xbfloat16] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
