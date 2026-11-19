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
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %1 = ub.poison : bf16
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
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
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x256xbf16> to memref<4x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<256x32xbf16> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c4 step %c2 {
# CHECK-NEXT:          %subview_3 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xbf16, strided<[256, 1], offset: ?>> to memref<2x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %subview_2[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c16 {
# CHECK-NEXT:            %subview_5 = memref.subview %subview_1[0, %arg5] [1, 16] [1, 1] : memref<1x32xbf16, strided<[32, 1], offset: ?>> to memref<1x16xbf16, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %subview_4[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_3[%c0, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[256, 1], offset: ?>> to memref<1x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_8 = memref.subview %subview_6[%c0, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xbf16, strided<[256, 1], offset: ?>> to memref<1xbf16, strided<[256], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_8[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %2 = vector.transfer_read %subview_9[%c0], %1 {in_bounds = [true]} : memref<1xbf16, strided<[256], offset: ?>>, vector<1xbf16>
# CHECK-NEXT:            %3 = vector.broadcast %2 : vector<1xbf16> to vector<16x1xbf16>
# CHECK-NEXT:            %4 = vector.transfer_read %subview_5[%c0, %c0], %1 {in_bounds = [true, true]} : memref<1x16xbf16, strided<[32, 1], offset: ?>>, vector<1x16xbf16>
# CHECK-NEXT:            %5 = vector.transpose %4, [1, 0] : vector<1x16xbf16> to vector<16x1xbf16>
# CHECK-NEXT:            %6 = vector.transfer_read %subview_10[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:            %7 = arith.extf %3 : vector<16x1xbf16> to vector<16x1xf32>
# CHECK-NEXT:            %8 = arith.extf %5 : vector<16x1xbf16> to vector<16x1xf32>
# CHECK-NEXT:            %9 = vector.extract %7[0] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %10 = vector.extract %8[0] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %11 = vector.extract %6[0] : f32 from vector<16xf32>
# CHECK-NEXT:            %12 = arith.mulf %9, %10 : vector<1xf32>
# CHECK-NEXT:            %13 = vector.reduction <add>, %12, %11 : vector<1xf32> into f32
# CHECK-NEXT:            %14 = vector.insert %13, %cst [0] : f32 into vector<16xf32>
# CHECK-NEXT:            %15 = vector.extract %7[1] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %16 = vector.extract %8[1] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %17 = vector.extract %6[1] : f32 from vector<16xf32>
# CHECK-NEXT:            %18 = arith.mulf %15, %16 : vector<1xf32>
# CHECK-NEXT:            %19 = vector.reduction <add>, %18, %17 : vector<1xf32> into f32
# CHECK-NEXT:            %20 = vector.insert %19, %14 [1] : f32 into vector<16xf32>
# CHECK-NEXT:            %21 = vector.extract %7[2] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %22 = vector.extract %8[2] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %23 = vector.extract %6[2] : f32 from vector<16xf32>
# CHECK-NEXT:            %24 = arith.mulf %21, %22 : vector<1xf32>
# CHECK-NEXT:            %25 = vector.reduction <add>, %24, %23 : vector<1xf32> into f32
# CHECK-NEXT:            %26 = vector.insert %25, %20 [2] : f32 into vector<16xf32>
# CHECK-NEXT:            %27 = vector.extract %7[3] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %28 = vector.extract %8[3] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %29 = vector.extract %6[3] : f32 from vector<16xf32>
# CHECK-NEXT:            %30 = arith.mulf %27, %28 : vector<1xf32>
# CHECK-NEXT:            %31 = vector.reduction <add>, %30, %29 : vector<1xf32> into f32
# CHECK-NEXT:            %32 = vector.insert %31, %26 [3] : f32 into vector<16xf32>
# CHECK-NEXT:            %33 = vector.extract %7[4] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %34 = vector.extract %8[4] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %35 = vector.extract %6[4] : f32 from vector<16xf32>
# CHECK-NEXT:            %36 = arith.mulf %33, %34 : vector<1xf32>
# CHECK-NEXT:            %37 = vector.reduction <add>, %36, %35 : vector<1xf32> into f32
# CHECK-NEXT:            %38 = vector.insert %37, %32 [4] : f32 into vector<16xf32>
# CHECK-NEXT:            %39 = vector.extract %7[5] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %40 = vector.extract %8[5] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %41 = vector.extract %6[5] : f32 from vector<16xf32>
# CHECK-NEXT:            %42 = arith.mulf %39, %40 : vector<1xf32>
# CHECK-NEXT:            %43 = vector.reduction <add>, %42, %41 : vector<1xf32> into f32
# CHECK-NEXT:            %44 = vector.insert %43, %38 [5] : f32 into vector<16xf32>
# CHECK-NEXT:            %45 = vector.extract %7[6] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %46 = vector.extract %8[6] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %47 = vector.extract %6[6] : f32 from vector<16xf32>
# CHECK-NEXT:            %48 = arith.mulf %45, %46 : vector<1xf32>
# CHECK-NEXT:            %49 = vector.reduction <add>, %48, %47 : vector<1xf32> into f32
# CHECK-NEXT:            %50 = vector.insert %49, %44 [6] : f32 into vector<16xf32>
# CHECK-NEXT:            %51 = vector.extract %7[7] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %52 = vector.extract %8[7] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %53 = vector.extract %6[7] : f32 from vector<16xf32>
# CHECK-NEXT:            %54 = arith.mulf %51, %52 : vector<1xf32>
# CHECK-NEXT:            %55 = vector.reduction <add>, %54, %53 : vector<1xf32> into f32
# CHECK-NEXT:            %56 = vector.insert %55, %50 [7] : f32 into vector<16xf32>
# CHECK-NEXT:            %57 = vector.extract %7[8] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %58 = vector.extract %8[8] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %59 = vector.extract %6[8] : f32 from vector<16xf32>
# CHECK-NEXT:            %60 = arith.mulf %57, %58 : vector<1xf32>
# CHECK-NEXT:            %61 = vector.reduction <add>, %60, %59 : vector<1xf32> into f32
# CHECK-NEXT:            %62 = vector.insert %61, %56 [8] : f32 into vector<16xf32>
# CHECK-NEXT:            %63 = vector.extract %7[9] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %64 = vector.extract %8[9] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %65 = vector.extract %6[9] : f32 from vector<16xf32>
# CHECK-NEXT:            %66 = arith.mulf %63, %64 : vector<1xf32>
# CHECK-NEXT:            %67 = vector.reduction <add>, %66, %65 : vector<1xf32> into f32
# CHECK-NEXT:            %68 = vector.insert %67, %62 [9] : f32 into vector<16xf32>
# CHECK-NEXT:            %69 = vector.extract %7[10] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %70 = vector.extract %8[10] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %71 = vector.extract %6[10] : f32 from vector<16xf32>
# CHECK-NEXT:            %72 = arith.mulf %69, %70 : vector<1xf32>
# CHECK-NEXT:            %73 = vector.reduction <add>, %72, %71 : vector<1xf32> into f32
# CHECK-NEXT:            %74 = vector.insert %73, %68 [10] : f32 into vector<16xf32>
# CHECK-NEXT:            %75 = vector.extract %7[11] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %76 = vector.extract %8[11] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %77 = vector.extract %6[11] : f32 from vector<16xf32>
# CHECK-NEXT:            %78 = arith.mulf %75, %76 : vector<1xf32>
# CHECK-NEXT:            %79 = vector.reduction <add>, %78, %77 : vector<1xf32> into f32
# CHECK-NEXT:            %80 = vector.insert %79, %74 [11] : f32 into vector<16xf32>
# CHECK-NEXT:            %81 = vector.extract %7[12] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %82 = vector.extract %8[12] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %83 = vector.extract %6[12] : f32 from vector<16xf32>
# CHECK-NEXT:            %84 = arith.mulf %81, %82 : vector<1xf32>
# CHECK-NEXT:            %85 = vector.reduction <add>, %84, %83 : vector<1xf32> into f32
# CHECK-NEXT:            %86 = vector.insert %85, %80 [12] : f32 into vector<16xf32>
# CHECK-NEXT:            %87 = vector.extract %7[13] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %88 = vector.extract %8[13] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %89 = vector.extract %6[13] : f32 from vector<16xf32>
# CHECK-NEXT:            %90 = arith.mulf %87, %88 : vector<1xf32>
# CHECK-NEXT:            %91 = vector.reduction <add>, %90, %89 : vector<1xf32> into f32
# CHECK-NEXT:            %92 = vector.insert %91, %86 [13] : f32 into vector<16xf32>
# CHECK-NEXT:            %93 = vector.extract %7[14] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %94 = vector.extract %8[14] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %95 = vector.extract %6[14] : f32 from vector<16xf32>
# CHECK-NEXT:            %96 = arith.mulf %93, %94 : vector<1xf32>
# CHECK-NEXT:            %97 = vector.reduction <add>, %96, %95 : vector<1xf32> into f32
# CHECK-NEXT:            %98 = vector.insert %97, %92 [14] : f32 into vector<16xf32>
# CHECK-NEXT:            %99 = vector.extract %7[15] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %100 = vector.extract %8[15] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %101 = vector.extract %6[15] : f32 from vector<16xf32>
# CHECK-NEXT:            %102 = arith.mulf %99, %100 : vector<1xf32>
# CHECK-NEXT:            %103 = vector.reduction <add>, %102, %101 : vector<1xf32> into f32
# CHECK-NEXT:            %104 = vector.insert %103, %98 [15] : f32 into vector<16xf32>
# CHECK-NEXT:            vector.transfer_write %104, %subview_10[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_3[%c1, 0] [1, 1] [1, 1] : memref<2x1xbf16, strided<[256, 1], offset: ?>> to memref<1x1xbf16, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_6[%c1, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_11[0, 0] [1, 1] [1, 1] : memref<1x1xbf16, strided<[256, 1], offset: ?>> to memref<1xbf16, strided<[256], offset: ?>>
# CHECK-NEXT:            %subview_14 = memref.subview %subview_12[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:            %105 = vector.transfer_read %subview_13[%c0], %1 {in_bounds = [true]} : memref<1xbf16, strided<[256], offset: ?>>, vector<1xbf16>
# CHECK-NEXT:            %106 = vector.broadcast %105 : vector<1xbf16> to vector<16x1xbf16>
# CHECK-NEXT:            %107 = vector.transfer_read %subview_5[%c0, %c0], %1 {in_bounds = [true, true]} : memref<1x16xbf16, strided<[32, 1], offset: ?>>, vector<1x16xbf16>
# CHECK-NEXT:            %108 = vector.transpose %107, [1, 0] : vector<1x16xbf16> to vector<16x1xbf16>
# CHECK-NEXT:            %109 = vector.transfer_read %subview_14[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:            %110 = arith.extf %106 : vector<16x1xbf16> to vector<16x1xf32>
# CHECK-NEXT:            %111 = arith.extf %108 : vector<16x1xbf16> to vector<16x1xf32>
# CHECK-NEXT:            %112 = vector.extract %110[0] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %113 = vector.extract %111[0] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %114 = vector.extract %109[0] : f32 from vector<16xf32>
# CHECK-NEXT:            %115 = arith.mulf %112, %113 : vector<1xf32>
# CHECK-NEXT:            %116 = vector.reduction <add>, %115, %114 : vector<1xf32> into f32
# CHECK-NEXT:            %117 = vector.insert %116, %cst [0] : f32 into vector<16xf32>
# CHECK-NEXT:            %118 = vector.extract %110[1] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %119 = vector.extract %111[1] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %120 = vector.extract %109[1] : f32 from vector<16xf32>
# CHECK-NEXT:            %121 = arith.mulf %118, %119 : vector<1xf32>
# CHECK-NEXT:            %122 = vector.reduction <add>, %121, %120 : vector<1xf32> into f32
# CHECK-NEXT:            %123 = vector.insert %122, %117 [1] : f32 into vector<16xf32>
# CHECK-NEXT:            %124 = vector.extract %110[2] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %125 = vector.extract %111[2] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %126 = vector.extract %109[2] : f32 from vector<16xf32>
# CHECK-NEXT:            %127 = arith.mulf %124, %125 : vector<1xf32>
# CHECK-NEXT:            %128 = vector.reduction <add>, %127, %126 : vector<1xf32> into f32
# CHECK-NEXT:            %129 = vector.insert %128, %123 [2] : f32 into vector<16xf32>
# CHECK-NEXT:            %130 = vector.extract %110[3] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %131 = vector.extract %111[3] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %132 = vector.extract %109[3] : f32 from vector<16xf32>
# CHECK-NEXT:            %133 = arith.mulf %130, %131 : vector<1xf32>
# CHECK-NEXT:            %134 = vector.reduction <add>, %133, %132 : vector<1xf32> into f32
# CHECK-NEXT:            %135 = vector.insert %134, %129 [3] : f32 into vector<16xf32>
# CHECK-NEXT:            %136 = vector.extract %110[4] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %137 = vector.extract %111[4] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %138 = vector.extract %109[4] : f32 from vector<16xf32>
# CHECK-NEXT:            %139 = arith.mulf %136, %137 : vector<1xf32>
# CHECK-NEXT:            %140 = vector.reduction <add>, %139, %138 : vector<1xf32> into f32
# CHECK-NEXT:            %141 = vector.insert %140, %135 [4] : f32 into vector<16xf32>
# CHECK-NEXT:            %142 = vector.extract %110[5] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %143 = vector.extract %111[5] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %144 = vector.extract %109[5] : f32 from vector<16xf32>
# CHECK-NEXT:            %145 = arith.mulf %142, %143 : vector<1xf32>
# CHECK-NEXT:            %146 = vector.reduction <add>, %145, %144 : vector<1xf32> into f32
# CHECK-NEXT:            %147 = vector.insert %146, %141 [5] : f32 into vector<16xf32>
# CHECK-NEXT:            %148 = vector.extract %110[6] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %149 = vector.extract %111[6] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %150 = vector.extract %109[6] : f32 from vector<16xf32>
# CHECK-NEXT:            %151 = arith.mulf %148, %149 : vector<1xf32>
# CHECK-NEXT:            %152 = vector.reduction <add>, %151, %150 : vector<1xf32> into f32
# CHECK-NEXT:            %153 = vector.insert %152, %147 [6] : f32 into vector<16xf32>
# CHECK-NEXT:            %154 = vector.extract %110[7] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %155 = vector.extract %111[7] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %156 = vector.extract %109[7] : f32 from vector<16xf32>
# CHECK-NEXT:            %157 = arith.mulf %154, %155 : vector<1xf32>
# CHECK-NEXT:            %158 = vector.reduction <add>, %157, %156 : vector<1xf32> into f32
# CHECK-NEXT:            %159 = vector.insert %158, %153 [7] : f32 into vector<16xf32>
# CHECK-NEXT:            %160 = vector.extract %110[8] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %161 = vector.extract %111[8] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %162 = vector.extract %109[8] : f32 from vector<16xf32>
# CHECK-NEXT:            %163 = arith.mulf %160, %161 : vector<1xf32>
# CHECK-NEXT:            %164 = vector.reduction <add>, %163, %162 : vector<1xf32> into f32
# CHECK-NEXT:            %165 = vector.insert %164, %159 [8] : f32 into vector<16xf32>
# CHECK-NEXT:            %166 = vector.extract %110[9] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %167 = vector.extract %111[9] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %168 = vector.extract %109[9] : f32 from vector<16xf32>
# CHECK-NEXT:            %169 = arith.mulf %166, %167 : vector<1xf32>
# CHECK-NEXT:            %170 = vector.reduction <add>, %169, %168 : vector<1xf32> into f32
# CHECK-NEXT:            %171 = vector.insert %170, %165 [9] : f32 into vector<16xf32>
# CHECK-NEXT:            %172 = vector.extract %110[10] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %173 = vector.extract %111[10] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %174 = vector.extract %109[10] : f32 from vector<16xf32>
# CHECK-NEXT:            %175 = arith.mulf %172, %173 : vector<1xf32>
# CHECK-NEXT:            %176 = vector.reduction <add>, %175, %174 : vector<1xf32> into f32
# CHECK-NEXT:            %177 = vector.insert %176, %171 [10] : f32 into vector<16xf32>
# CHECK-NEXT:            %178 = vector.extract %110[11] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %179 = vector.extract %111[11] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %180 = vector.extract %109[11] : f32 from vector<16xf32>
# CHECK-NEXT:            %181 = arith.mulf %178, %179 : vector<1xf32>
# CHECK-NEXT:            %182 = vector.reduction <add>, %181, %180 : vector<1xf32> into f32
# CHECK-NEXT:            %183 = vector.insert %182, %177 [11] : f32 into vector<16xf32>
# CHECK-NEXT:            %184 = vector.extract %110[12] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %185 = vector.extract %111[12] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %186 = vector.extract %109[12] : f32 from vector<16xf32>
# CHECK-NEXT:            %187 = arith.mulf %184, %185 : vector<1xf32>
# CHECK-NEXT:            %188 = vector.reduction <add>, %187, %186 : vector<1xf32> into f32
# CHECK-NEXT:            %189 = vector.insert %188, %183 [12] : f32 into vector<16xf32>
# CHECK-NEXT:            %190 = vector.extract %110[13] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %191 = vector.extract %111[13] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %192 = vector.extract %109[13] : f32 from vector<16xf32>
# CHECK-NEXT:            %193 = arith.mulf %190, %191 : vector<1xf32>
# CHECK-NEXT:            %194 = vector.reduction <add>, %193, %192 : vector<1xf32> into f32
# CHECK-NEXT:            %195 = vector.insert %194, %189 [13] : f32 into vector<16xf32>
# CHECK-NEXT:            %196 = vector.extract %110[14] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %197 = vector.extract %111[14] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %198 = vector.extract %109[14] : f32 from vector<16xf32>
# CHECK-NEXT:            %199 = arith.mulf %196, %197 : vector<1xf32>
# CHECK-NEXT:            %200 = vector.reduction <add>, %199, %198 : vector<1xf32> into f32
# CHECK-NEXT:            %201 = vector.insert %200, %195 [14] : f32 into vector<16xf32>
# CHECK-NEXT:            %202 = vector.extract %110[15] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %203 = vector.extract %111[15] : vector<1xf32> from vector<16x1xf32>
# CHECK-NEXT:            %204 = vector.extract %109[15] : f32 from vector<16xf32>
# CHECK-NEXT:            %205 = arith.mulf %202, %203 : vector<1xf32>
# CHECK-NEXT:            %206 = vector.reduction <add>, %205, %204 : vector<1xf32> into f32
# CHECK-NEXT:            %207 = vector.insert %206, %201 [15] : f32 into vector<16xf32>
# CHECK-NEXT:            vector.transfer_write %207, %subview_14[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
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
