// RUN: mlir-loop %s --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<4x4xf32>,
  %B: memref<4x4xf32>,
  %C: memref<4x4xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.add_attributes = ["JoeDassin"]
    }
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0] [4, 4] [1, 1] : memref<4x4xf32> to memref<4x4xf32, strided<[4, 1]>>
// CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:        %c0_2 = arith.constant 0 : index
// CHECK-NEXT:        %c4_3 = arith.constant 4 : index
// CHECK-NEXT:        %c1_4 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c4_3 step %c1_4 {
// CHECK-NEXT:          %subview_5 = memref.subview %subview[0, 0] [1, 4] [1, 1] : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:          %subview_6 = memref.subview %subview_0[0, %arg4] [4, 1] [1, 1] : memref<4x4xf32, strided<[4, 1]>> to memref<4x1xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:          %subview_7 = memref.subview %subview_1[0, %arg4] [1, 1] [1, 1] : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x1xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:          %c0_8 = arith.constant 0 : index
// CHECK-NEXT:          %c4_9 = arith.constant 4 : index
// CHECK-NEXT:          %c1_10 = arith.constant 1 : index
// CHECK-NEXT:          scf.for %arg5 = %c0_8 to %c4_9 step %c1_10 {
// CHECK-NEXT:            %subview_11 = memref.subview %subview_5[0, %arg5] [1, 1] [1, 1] : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x1xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:            %subview_12 = memref.subview %subview_6[%arg5, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[4, 1], offset: ?>> to memref<1x1xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:            %subview_13 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>> to memref<1x1xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__id0__, loop.add_attributes = ["JoeDassin"]} ins(%subview_11, %subview_12 : memref<1x1xf32, strided<[4, 1], offset: ?>>, memref<1x1xf32, strided<[4, 1], offset: ?>>) outs(%subview_13 : memref<1x1xf32, strided<[4, 1], offset: ?>>)
// CHECK-NEXT:          } {__id0__k}
// CHECK-NEXT:        } {__id0__j}
// CHECK-NEXT:      } {JoeDassin, __id0__i}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
