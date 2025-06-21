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
          "J[:128]" = {
            "J",
              "K"
          },
          "J[128:]" = {
            "K",
              "J"
          }
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
// CHECK-NEXT:        %subview_2 = memref.subview %subview[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_3 = memref.subview %subview_0[0, 0] [512, 128] [1, 1] : memref<512x256xf32, strided<[256, 1]>> to memref<512x128xf32, strided<[256, 1]>>
// CHECK-NEXT:        %subview_4 = memref.subview %subview_1[0, 0] [1, 128] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x128xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        %c0_5 = arith.constant 0 : index
// CHECK-NEXT:        %c128 = arith.constant 128 : index
// CHECK-NEXT:        %c1_6 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_5 to %c128 step %c1_6 {
// CHECK-NEXT:          %subview_12 = memref.subview %subview_2[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:          %subview_13 = memref.subview %subview_3[0, %arg4] [512, 1] [1, 1] : memref<512x128xf32, strided<[256, 1]>> to memref<512x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %subview_14 = memref.subview %subview_4[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %c0_15 = arith.constant 0 : index
// CHECK-NEXT:          %c512_16 = arith.constant 512 : index
// CHECK-NEXT:          %c1_17 = arith.constant 1 : index
// CHECK-NEXT:          scf.for %arg5 = %c0_15 to %c512_16 step %c1_17 {
// CHECK-NEXT:            %subview_18 = memref.subview %subview_12[0, %arg5] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_19 = memref.subview %subview_13[%arg5, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_20 = memref.subview %subview_14[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_18, %subview_19 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_20 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:          } {"__node0__/J[0]/K0"}
// CHECK-NEXT:        } {"__node0__/J[0]/J0"}
// CHECK-NEXT:        %subview_7 = memref.subview %subview[0, 0] [1, 512] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_8 = memref.subview %subview_0[0, 128] [512, 128] [1, 1] : memref<512x256xf32, strided<[256, 1]>> to memref<512x128xf32, strided<[256, 1], offset: 128>>
// CHECK-NEXT:        %subview_9 = memref.subview %subview_1[0, 128] [1, 128] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x128xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:        %c0_10 = arith.constant 0 : index
// CHECK-NEXT:        %c512 = arith.constant 512 : index
// CHECK-NEXT:        %c1_11 = arith.constant 1 : index
// CHECK-NEXT:        scf.for %arg4 = %c0_10 to %c512 step %c1_11 {
// CHECK-NEXT:          %subview_12 = memref.subview %subview_7[0, %arg4] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:          %subview_13 = memref.subview %subview_8[%arg4, 0] [1, 128] [1, 1] : memref<512x128xf32, strided<[256, 1], offset: 128>> to memref<1x128xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %subview_14 = memref.subview %subview_9[0, 0] [1, 128] [1, 1] : memref<1x128xf32, strided<[256, 1], offset: ?>> to memref<1x128xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:          %c0_15 = arith.constant 0 : index
// CHECK-NEXT:          %c128_16 = arith.constant 128 : index
// CHECK-NEXT:          %c1_17 = arith.constant 1 : index
// CHECK-NEXT:          scf.for %arg5 = %c0_15 to %c128_16 step %c1_17 {
// CHECK-NEXT:            %subview_18 = memref.subview %subview_12[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_19 = memref.subview %subview_13[0, %arg5] [1, 1] [1, 1] : memref<1x128xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %subview_20 = memref.subview %subview_14[0, %arg5] [1, 1] [1, 1] : memref<1x128xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            linalg.matmul {__node0__} ins(%subview_18, %subview_19 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_20 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
// CHECK-NEXT:          } {"__node0__/J[1]/J0"}
// CHECK-NEXT:        } {"__node0__/J[1]/K0"}
// CHECK-NEXT:      } {"__node0__/I0"}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
