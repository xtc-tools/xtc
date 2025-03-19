// RUN: mlir-backend %s --print-source-ir 2>&1 | filecheck %s

module {
func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32>, %arg1: memref<512x256xf32>, %arg2: memref<256x256xf32>) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      linalg.matmul ins(%arg0, %arg1 : memref<256x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  