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
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64"
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
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/J0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__node0__/K0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__node0__/I1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__node0__/K1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "__node0__/J1" : !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_7 {factor = 8 : i64} : !transform.any_op
// CHECK-NEXT:      transform.loop.unroll %loops_5 {factor = 1 : i64} : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
