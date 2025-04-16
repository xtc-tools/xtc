// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-transformed-ir %s 2>&1 | filecheck %s

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
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
    attrs = {
      loop.dims = ["i","j","k"],
      loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 64}, "k" = {"k1" = 8}},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {"i1" = 1, "k" = 8}
    }
  {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %c56 = arith.constant 56 : index
// CHECK-NEXT:      %c48 = arith.constant 48 : index
// CHECK-NEXT:      %c40 = arith.constant 40 : index
// CHECK-NEXT:      %c32 = arith.constant 32 : index
// CHECK-NEXT:      %c24 = arith.constant 24 : index
// CHECK-NEXT:      %c16 = arith.constant 16 : index
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
// CHECK-NEXT:          scf.for %arg5 = %c0 to %c512 step %c64 {
// CHECK-NEXT:            %subview_5 = memref.subview %subview[0, %arg5] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_6 = memref.subview %subview_3[%arg5, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_5[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_6[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %0 = arith.addi %arg5, %c8 : index
// CHECK-NEXT:            %subview_7 = memref.subview %subview[0, %0] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_8 = memref.subview %subview_3[%0, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_7[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_8[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %1 = arith.addi %arg5, %c16 : index
// CHECK-NEXT:            %subview_9 = memref.subview %subview[0, %1] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_10 = memref.subview %subview_3[%1, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_9[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_10[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %2 = arith.addi %arg5, %c24 : index
// CHECK-NEXT:            %subview_11 = memref.subview %subview[0, %2] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_12 = memref.subview %subview_3[%2, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_11[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_12[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %3 = arith.addi %arg5, %c32 : index
// CHECK-NEXT:            %subview_13 = memref.subview %subview[0, %3] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_14 = memref.subview %subview_3[%3, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_13[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_14[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %4 = arith.addi %arg5, %c40 : index
// CHECK-NEXT:            %subview_15 = memref.subview %subview[0, %4] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_16 = memref.subview %subview_3[%4, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_15[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_16[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %5 = arith.addi %arg5, %c48 : index
// CHECK-NEXT:            %subview_17 = memref.subview %subview[0, %5] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_18 = memref.subview %subview_3[%5, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_17[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_18[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:            %6 = arith.addi %arg5, %c56 : index
// CHECK-NEXT:            %subview_19 = memref.subview %subview[0, %6] [1, 8] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x8xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_20 = memref.subview %subview_3[%6, 0] [8, 64] [1, 1] : memref<512x64xf32, strided<[256, 1], offset: ?>> to memref<8x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            scf.for %arg6 = %c0 to %c8 step %c1 {
// CHECK-NEXT:              %subview_21 = memref.subview %subview_19[0, %arg6] [1, 1] [1, 1] : memref<1x8xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:              %subview_22 = memref.subview %subview_20[%arg6, 0] [1, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>> to memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:              %7 = vector.transfer_read %subview_21[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:              %8 = vector.transfer_read %subview_22[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %9 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:              %10 = vector.extract %8[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %11 = vector.extract %7[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:              %12 = vector.broadcast %11 : f32 to vector<64xf32>
// CHECK-NEXT:              %13 = vector.extract %9[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:              %14 = vector.fma %12, %10, %13 : vector<64xf32>
// CHECK-NEXT:              %15 = vector.insert %14, %cst_0 [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:              vector.transfer_write %15, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            } {__id1__k1}
// CHECK-NEXT:          } {__id1__k}
// CHECK-NEXT:        } {__id1__j}
// CHECK-NEXT:      } {__id1__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
