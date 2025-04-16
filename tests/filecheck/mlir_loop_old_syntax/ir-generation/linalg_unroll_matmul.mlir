// RUN: mlir-loop --old-syntax %s --print-transformed-ir --no-alias 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    {loop.dims = ["i","j"]}
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {loop.dims = ["i","j","k"], loop.unroll = {"k" = 8}}
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c256 = arith.constant 256 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        %c0_3 = arith.constant 0 : index
// CHECK-NEXT:        %c256_4 = arith.constant 256 : index
// CHECK-NEXT:        %c1_5 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_3 to %c256_4 step %c1_5 {
// CHECK-NEXT:          %subview_6 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          linalg.fill {__id0__} ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:        } {__id0__j}
// CHECK-NEXT:      } {__id0__i}
// CHECK-NEXT:      %c0_0 = arith.constant 0 : index
// CHECK-NEXT:      %c256_1 = arith.constant 256 : index
// CHECK-NEXT:      %c1_2 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c256_1 step %c1_2 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [1, 512] [1, 1] : memref<256x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_3 = memref.subview %arg1[0, 0] [512, 256] [1, 1] : memref<512x256xf32> to memref<512x256xf32, strided<[256, 1]>>
// CHECK-NEXT:        %subview_4 = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        %c0_5 = arith.constant 0 : index
// CHECK-NEXT:        %c256_6 = arith.constant 256 : index
// CHECK-NEXT:        %c1_7 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_5 to %c256_6 step %c1_7 {
// CHECK-NEXT:          %subview_8 = memref.subview %subview[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:          %subview_9 = memref.subview %subview_3[0, %arg4] [512, 1] [1, 1] : memref<512x256xf32, strided<[256, 1]>> to memref<512x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %subview_10 = memref.subview %subview_4[0, %arg4] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %c0_11 = arith.constant 0 : index
// CHECK-NEXT:          %c512 = arith.constant 512 : index
// CHECK-NEXT:          %c1_12 = arith.constant 1 : index
// CHECK-NEXT:          %c8 = arith.constant 8 : index
// CHECK-NEXT:          scf.for %arg5 = %c0_11 to %c512 step %c8 {
// CHECK-NEXT:            %subview_13 = memref.subview %subview_8[0, %arg5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_14 = memref.subview %subview_9[%arg5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_15 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_13, %subview_14 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_15 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c1_16 = arith.constant 1 : index
// CHECK-NEXT:            %0 = arith.muli %c1_12, %c1_16 : index
// CHECK-NEXT:            %1 = arith.addi %arg5, %0 : index
// CHECK-NEXT:            %subview_17 = memref.subview %subview_8[0, %1] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_18 = memref.subview %subview_9[%1, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_19 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_17, %subview_18 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_19 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c2 = arith.constant 2 : index
// CHECK-NEXT:            %2 = arith.muli %c1_12, %c2 : index
// CHECK-NEXT:            %3 = arith.addi %arg5, %2 : index
// CHECK-NEXT:            %subview_20 = memref.subview %subview_8[0, %3] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_21 = memref.subview %subview_9[%3, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_22 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_20, %subview_21 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_22 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c3 = arith.constant 3 : index
// CHECK-NEXT:            %4 = arith.muli %c1_12, %c3 : index
// CHECK-NEXT:            %5 = arith.addi %arg5, %4 : index
// CHECK-NEXT:            %subview_23 = memref.subview %subview_8[0, %5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_24 = memref.subview %subview_9[%5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_25 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_23, %subview_24 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_25 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c4 = arith.constant 4 : index
// CHECK-NEXT:            %6 = arith.muli %c1_12, %c4 : index
// CHECK-NEXT:            %7 = arith.addi %arg5, %6 : index
// CHECK-NEXT:            %subview_26 = memref.subview %subview_8[0, %7] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_27 = memref.subview %subview_9[%7, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_28 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_26, %subview_27 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_28 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c5 = arith.constant 5 : index
// CHECK-NEXT:            %8 = arith.muli %c1_12, %c5 : index
// CHECK-NEXT:            %9 = arith.addi %arg5, %8 : index
// CHECK-NEXT:            %subview_29 = memref.subview %subview_8[0, %9] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_30 = memref.subview %subview_9[%9, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_31 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_29, %subview_30 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_31 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c6 = arith.constant 6 : index
// CHECK-NEXT:            %10 = arith.muli %c1_12, %c6 : index
// CHECK-NEXT:            %11 = arith.addi %arg5, %10 : index
// CHECK-NEXT:            %subview_32 = memref.subview %subview_8[0, %11] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_33 = memref.subview %subview_9[%11, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_34 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_32, %subview_33 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_34 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c7 = arith.constant 7 : index
// CHECK-NEXT:            %12 = arith.muli %c1_12, %c7 : index
// CHECK-NEXT:            %13 = arith.addi %arg5, %12 : index
// CHECK-NEXT:            %subview_35 = memref.subview %subview_8[0, %13] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_36 = memref.subview %subview_9[%13, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_37 = memref.subview %subview_10[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id1__} ins(%subview_35, %subview_36 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_37 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:          } {__id1__k}
// CHECK-NEXT:        } {__id1__j}
// CHECK-NEXT:      } {__id1__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
