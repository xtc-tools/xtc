// RUN: mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s

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
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<256x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<256x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      linalg.matmul {__node0__} ins(%arg0, %arg1 : memref<256x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<256x256xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__node0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__node0__/I0" : !transform.any_op
// CHECK-NEXT:      %first, %second = transform.structured.split %tiled_linalg_op after 128  {dimension = 1 : i64} : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %second tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/J[1]/K0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__node0__/J[1]/J0" : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
