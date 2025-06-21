// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "J",
            "K" = {"unroll" = 4}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c256 = arith.constant 256 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c256 step %c1 {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [1, 512] [1, 1] : memref<256x512xf32> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0] [512, 256] [1, 1] : memref<512x256xf32> to memref<512x256xf32, strided<[256, 1]>>
// CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        %c0_2 = arith.constant 0 : index
// CHECK-NEXT:        %c256_3 = arith.constant 256 : index
// CHECK-NEXT:        %c1_4 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c256_3 step %c1_4 {
// CHECK-NEXT:          %subview_5 = memref.subview %subview[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:          %subview_6 = memref.subview %subview_0[0, %arg4] [512, 1] [1, 1] : memref<512x256xf32, strided<[256, 1]>> to memref<512x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %subview_7 = memref.subview %subview_1[0, %arg4] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %c0_8 = arith.constant 0 : index
// CHECK-NEXT:          %c512 = arith.constant 512 : index
// CHECK-NEXT:          %c1_9 = arith.constant 1 : index
// CHECK-NEXT:          %c4 = arith.constant 4 : index
// CHECK-NEXT:          scf.for %arg5 = %c0_8 to %c512 step %c4 {
// CHECK-NEXT:            %subview_10 = memref.subview %subview_5[0, %arg5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_11 = memref.subview %subview_6[%arg5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_12 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_10, %subview_11 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_12 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c1_13 = arith.constant 1 : index
// CHECK-NEXT:            %0 = arith.muli %c1_9, %c1_13 : index
// CHECK-NEXT:            %1 = arith.addi %arg5, %0 : index
// CHECK-NEXT:            %subview_14 = memref.subview %subview_5[0, %1] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_15 = memref.subview %subview_6[%1, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_16 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_14, %subview_15 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_16 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c2 = arith.constant 2 : index
// CHECK-NEXT:            %2 = arith.muli %c1_9, %c2 : index
// CHECK-NEXT:            %3 = arith.addi %arg5, %2 : index
// CHECK-NEXT:            %subview_17 = memref.subview %subview_5[0, %3] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_18 = memref.subview %subview_6[%3, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_19 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_17, %subview_18 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_19 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:            %c3 = arith.constant 3 : index
// CHECK-NEXT:            %4 = arith.muli %c1_9, %c3 : index
// CHECK-NEXT:            %5 = arith.addi %arg5, %4 : index
// CHECK-NEXT:            %subview_20 = memref.subview %subview_5[0, %5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_21 = memref.subview %subview_6[%5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_22 = memref.subview %subview_7[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_20, %subview_21 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_22 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:          } {"__node0__/K0"}
// CHECK-NEXT:        } {"__node0__/J0"}
// CHECK-NEXT:      } {"__node0__/I0"}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
