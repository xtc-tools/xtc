// RUN: mlir-loop --old-syntax %s --no-alias --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<250x480xf32>,
  %B: memref<480x640xf32>,
  %C: memref<250x640xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<250x640xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {
        "i" = {"i1" = 5,  "i2" = 1},
        "j" = {"j1" = 64, "j2" = 32},
        "k" = {"k1" = 96, "k2" = 8}},
      loop.interchange = ["i","j","k","i1","k1","j1","i2","k2","j2"],
      loop.vectorize = ["j2"],
      loop.unroll = {"i2" = 1, "k2" = 8}
    }
    ins(%A, %B : memref<250x480xf32>, memref<480x640xf32>)
    outs(%C : memref<250x640xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<250x480xf32> {llvm.noalias}, %arg1: memref<480x640xf32> {llvm.noalias}, %arg2: memref<250x640xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<1x32xf32>
// CHECK-NEXT:      %c7 = arith.constant 7 : index
// CHECK-NEXT:      %c6 = arith.constant 6 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c3 = arith.constant 3 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %c32 = arith.constant 32 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c96 = arith.constant 96 : index
// CHECK-NEXT:      %c480 = arith.constant 480 : index
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c640 = arith.constant 640 : index
// CHECK-NEXT:      %c5 = arith.constant 5 : index
// CHECK-NEXT:      %c250 = arith.constant 250 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst_1 = arith.constant dense<0.000000e+00> : vector<250x640xf32>
// CHECK-NEXT:      vector.transfer_write %cst_1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<250x640xf32>, memref<250x640xf32>
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c250 step %c5 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [5, 480] [1, 1] : memref<250x480xf32> to memref<5x480xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:        %subview_2 = memref.subview %arg1[0, 0] [480, 640] [1, 1] : memref<480x640xf32> to memref<480x640xf32, strided<[640, 1]>>
// CHECK-NEXT:        %subview_3 = memref.subview %arg2[%arg3, 0] [5, 640] [1, 1] : memref<250x640xf32> to memref<5x640xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c640 step %c64 {
// CHECK-NEXT:          %subview_4 = memref.subview %subview_2[0, %arg4] [480, 64] [1, 1] : memref<480x640xf32, strided<[640, 1]>> to memref<480x64xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:          %subview_5 = memref.subview %subview_3[0, %arg4] [5, 64] [1, 1] : memref<5x640xf32, strided<[640, 1], offset: ?>> to memref<5x64xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c480 step %c96 {
// CHECK-NEXT:            %subview_6 = memref.subview %subview[0, %arg5] [5, 96] [1, 1] : memref<5x480xf32, strided<[480, 1], offset: ?>> to memref<5x96xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:            %subview_7 = memref.subview %subview_4[%arg5, 0] [96, 64] [1, 1] : memref<480x64xf32, strided<[640, 1], offset: ?>> to memref<96x64xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c5 step %c1 {
// CHECK-NEXT:              %subview_8 = memref.subview %subview_6[%arg6, 0] [1, 96] [1, 1] : memref<5x96xf32, strided<[480, 1], offset: ?>> to memref<1x96xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:              %subview_9 = memref.subview %subview_5[%arg6, 0] [1, 64] [1, 1] : memref<5x64xf32, strided<[640, 1], offset: ?>> to memref<1x64xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:              scf.for %arg7 = %c0 to %c96 step %c8 {
// CHECK-NEXT:                %subview_10 = memref.subview %subview_8[0, %arg7] [1, 8] [1, 1] : memref<1x96xf32, strided<[480, 1], offset: ?>> to memref<1x8xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                %subview_11 = memref.subview %subview_7[%arg7, 0] [8, 64] [1, 1] : memref<96x64xf32, strided<[640, 1], offset: ?>> to memref<8x64xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                scf.for %arg8 = %c0 to %c64 step %c32 {
// CHECK-NEXT:                  %subview_12 = memref.subview %subview_11[0, %arg8] [8, 32] [1, 1] : memref<8x64xf32, strided<[640, 1], offset: ?>> to memref<8x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_13 = memref.subview %subview_9[0, %arg8] [1, 32] [1, 1] : memref<1x64xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_14 = memref.subview %subview_10[0, %c0] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_15 = memref.subview %subview_12[%c0, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %0 = vector.transfer_read %subview_14[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %1 = vector.transfer_read %subview_15[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %2 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %3 = vector.extract %1[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %5 = vector.broadcast %4 : f32 to vector<32xf32>
// CHECK-NEXT:                  %6 = vector.extract %2[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %7 = vector.fma %5, %3, %6 : vector<32xf32>
// CHECK-NEXT:                  %8 = vector.insert %7, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %8, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_16 = memref.subview %subview_10[0, %c1] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_17 = memref.subview %subview_12[%c1, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %9 = vector.transfer_read %subview_16[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %10 = vector.transfer_read %subview_17[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %11 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %12 = vector.extract %10[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %13 = vector.extract %9[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %14 = vector.broadcast %13 : f32 to vector<32xf32>
// CHECK-NEXT:                  %15 = vector.extract %11[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %16 = vector.fma %14, %12, %15 : vector<32xf32>
// CHECK-NEXT:                  %17 = vector.insert %16, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %17, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_18 = memref.subview %subview_10[0, %c2] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_19 = memref.subview %subview_12[%c2, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %18 = vector.transfer_read %subview_18[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %19 = vector.transfer_read %subview_19[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %20 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %21 = vector.extract %19[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %22 = vector.extract %18[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %23 = vector.broadcast %22 : f32 to vector<32xf32>
// CHECK-NEXT:                  %24 = vector.extract %20[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %25 = vector.fma %23, %21, %24 : vector<32xf32>
// CHECK-NEXT:                  %26 = vector.insert %25, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %26, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_20 = memref.subview %subview_10[0, %c3] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_21 = memref.subview %subview_12[%c3, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %27 = vector.transfer_read %subview_20[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %28 = vector.transfer_read %subview_21[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %29 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %30 = vector.extract %28[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %31 = vector.extract %27[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %32 = vector.broadcast %31 : f32 to vector<32xf32>
// CHECK-NEXT:                  %33 = vector.extract %29[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %34 = vector.fma %32, %30, %33 : vector<32xf32>
// CHECK-NEXT:                  %35 = vector.insert %34, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %35, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_22 = memref.subview %subview_10[0, %c4] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_23 = memref.subview %subview_12[%c4, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %36 = vector.transfer_read %subview_22[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %37 = vector.transfer_read %subview_23[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %38 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %39 = vector.extract %37[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %40 = vector.extract %36[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %41 = vector.broadcast %40 : f32 to vector<32xf32>
// CHECK-NEXT:                  %42 = vector.extract %38[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %43 = vector.fma %41, %39, %42 : vector<32xf32>
// CHECK-NEXT:                  %44 = vector.insert %43, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %44, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_24 = memref.subview %subview_10[0, %c5] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_25 = memref.subview %subview_12[%c5, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %45 = vector.transfer_read %subview_24[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %46 = vector.transfer_read %subview_25[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %47 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %48 = vector.extract %46[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %49 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %50 = vector.broadcast %49 : f32 to vector<32xf32>
// CHECK-NEXT:                  %51 = vector.extract %47[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %52 = vector.fma %50, %48, %51 : vector<32xf32>
// CHECK-NEXT:                  %53 = vector.insert %52, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %53, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_26 = memref.subview %subview_10[0, %c6] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_27 = memref.subview %subview_12[%c6, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %54 = vector.transfer_read %subview_26[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %55 = vector.transfer_read %subview_27[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %56 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %57 = vector.extract %55[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %58 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %59 = vector.broadcast %58 : f32 to vector<32xf32>
// CHECK-NEXT:                  %60 = vector.extract %56[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %61 = vector.fma %59, %57, %60 : vector<32xf32>
// CHECK-NEXT:                  %62 = vector.insert %61, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %62, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %subview_28 = memref.subview %subview_10[0, %c7] [1, 1] [1, 1] : memref<1x8xf32, strided<[480, 1], offset: ?>> to memref<1x1xf32, strided<[480, 1], offset: ?>>
// CHECK-NEXT:                  %subview_29 = memref.subview %subview_12[%c7, 0] [1, 32] [1, 1] : memref<8x32xf32, strided<[640, 1], offset: ?>> to memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                  %63 = vector.transfer_read %subview_28[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[480, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:                  %64 = vector.transfer_read %subview_29[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %65 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x32xf32, strided<[640, 1], offset: ?>>, vector<1x32xf32>
// CHECK-NEXT:                  %66 = vector.extract %64[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %67 = vector.extract %63[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:                  %68 = vector.broadcast %67 : f32 to vector<32xf32>
// CHECK-NEXT:                  %69 = vector.extract %65[0] : vector<32xf32> from vector<1x32xf32>
// CHECK-NEXT:                  %70 = vector.fma %68, %66, %69 : vector<32xf32>
// CHECK-NEXT:                  %71 = vector.insert %70, %cst [0] : vector<32xf32> into vector<1x32xf32>
// CHECK-NEXT:                  vector.transfer_write %71, %subview_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[640, 1], offset: ?>>
// CHECK-NEXT:                } {__id0__j1}
// CHECK-NEXT:              } {__id0__k1}
// CHECK-NEXT:            } {__id0__i1}
// CHECK-NEXT:          } {__id0__k}
// CHECK-NEXT:        } {__id0__j}
// CHECK-NEXT:      } {__id0__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
