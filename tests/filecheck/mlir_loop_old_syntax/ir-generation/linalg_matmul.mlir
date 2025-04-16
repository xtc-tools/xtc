// RUN: mlir-loop --old-syntax %s --no-alias --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = ["i","j"],
        loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 64}},
        loop.interchange = ["i","j","i1","j1"],
        loop.vectorize = ["j1"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 64}, "k" = {"k1" = 8}},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {"i1" = 1, "k1" = 8}
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %c7 = arith.constant 7 : index
// CHECK-NEXT:      %c6 = arith.constant 6 : index
// CHECK-NEXT:      %c5 = arith.constant 5 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c3 = arith.constant 3 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c512 = arith.constant 512 : index
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c256 = arith.constant 256 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c256 step %c64 {
// CHECK-NEXT:          %subview_1 = memref.subview %subview[0, %arg4] [1, 64] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c1 step %c1 {
// CHECK-NEXT:            vector.transfer_write %cst_0, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          } {__id0__i1}
// CHECK-NEXT:        } {__id0__j}
// CHECK-NEXT:      } {__id0__i}
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [1, 512] [1, 1] : memref<256x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_1 = memref.subview %arg1[0, 0] [512, 256] [1, 1] : memref<512x256xf32> to memref<512x256xf32, strided<[256, 1]>>
// CHECK-NEXT:        %subview_2 = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c256 step %c64 {
// CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4] [512, 64] [1, 1] : memref<512x256xf32, strided<[256, 1]>> to memref<512x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %subview_4 = memref.subview %subview_2[0, %arg4] [1, 64] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c512 step %c8 {
// CHECK-NEXT:            %subview_5 = memref.subview %subview[0, %arg5] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_6 = memref.subview %subview_3[%arg5, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_7 = memref.subview %subview_5[0, %c0] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_8 = memref.subview %subview_6[%c0, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %0 = vector.transfer_read %subview_7[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %1 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %2 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %3 = vector.extract %1[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %5 = vector.broadcast %4 : f32 to vector<64xf32>
// CHECK-NEXT:            %6 = vector.extract %2[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %7 = vector.fma %5, %3, %6 : vector<64xf32>
// CHECK-NEXT:            %8 = vector.insert %7, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %8, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_9 = memref.subview %subview_5[0, %c1] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_10 = memref.subview %subview_6[%c1, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %9 = vector.transfer_read %subview_9[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %10 = vector.transfer_read %subview_10[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %11 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %12 = vector.extract %10[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %13 = vector.extract %9[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %14 = vector.broadcast %13 : f32 to vector<64xf32>
// CHECK-NEXT:            %15 = vector.extract %11[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %16 = vector.fma %14, %12, %15 : vector<64xf32>
// CHECK-NEXT:            %17 = vector.insert %16, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %17, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_11 = memref.subview %subview_5[0, %c2] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_12 = memref.subview %subview_6[%c2, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %18 = vector.transfer_read %subview_11[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %19 = vector.transfer_read %subview_12[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %20 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %21 = vector.extract %19[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %22 = vector.extract %18[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %23 = vector.broadcast %22 : f32 to vector<64xf32>
// CHECK-NEXT:            %24 = vector.extract %20[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %25 = vector.fma %23, %21, %24 : vector<64xf32>
// CHECK-NEXT:            %26 = vector.insert %25, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %26, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_13 = memref.subview %subview_5[0, %c3] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_14 = memref.subview %subview_6[%c3, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %27 = vector.transfer_read %subview_13[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %28 = vector.transfer_read %subview_14[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %29 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %30 = vector.extract %28[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %31 = vector.extract %27[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %32 = vector.broadcast %31 : f32 to vector<64xf32>
// CHECK-NEXT:            %33 = vector.extract %29[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %34 = vector.fma %32, %30, %33 : vector<64xf32>
// CHECK-NEXT:            %35 = vector.insert %34, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %35, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_15 = memref.subview %subview_5[0, %c4] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_16 = memref.subview %subview_6[%c4, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %36 = vector.transfer_read %subview_15[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %37 = vector.transfer_read %subview_16[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %38 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %39 = vector.extract %37[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %40 = vector.extract %36[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %41 = vector.broadcast %40 : f32 to vector<64xf32>
// CHECK-NEXT:            %42 = vector.extract %38[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %43 = vector.fma %41, %39, %42 : vector<64xf32>
// CHECK-NEXT:            %44 = vector.insert %43, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %44, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_17 = memref.subview %subview_5[0, %c5] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_18 = memref.subview %subview_6[%c5, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %45 = vector.transfer_read %subview_17[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %46 = vector.transfer_read %subview_18[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %47 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %48 = vector.extract %46[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %49 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %50 = vector.broadcast %49 : f32 to vector<64xf32>
// CHECK-NEXT:            %51 = vector.extract %47[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %52 = vector.fma %50, %48, %51 : vector<64xf32>
// CHECK-NEXT:            %53 = vector.insert %52, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %53, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_19 = memref.subview %subview_5[0, %c6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_20 = memref.subview %subview_6[%c6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %54 = vector.transfer_read %subview_19[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %55 = vector.transfer_read %subview_20[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %56 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %57 = vector.extract %55[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %58 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %59 = vector.broadcast %58 : f32 to vector<64xf32>
// CHECK-NEXT:            %60 = vector.extract %56[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %61 = vector.fma %59, %57, %60 : vector<64xf32>
// CHECK-NEXT:            %62 = vector.insert %61, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %62, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_21 = memref.subview %subview_5[0, %c7] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_22 = memref.subview %subview_6[%c7, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %63 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %64 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %65 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %66 = vector.extract %64[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %67 = vector.extract %63[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %68 = vector.broadcast %67 : f32 to vector<64xf32>
// CHECK-NEXT:            %69 = vector.extract %65[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %70 = vector.fma %68, %66, %69 : vector<64xf32>
// CHECK-NEXT:            %71 = vector.insert %70, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %71, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          } {__id1__k}
// CHECK-NEXT:        } {__id1__j}
// CHECK-NEXT:      } {__id1__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
