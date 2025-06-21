// RUN: mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<258x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<258x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I[:2]" = {},
        "I[2:]" = {
          "I",
            "J",
              "K",
                "I#2",
                  "K#8",
                    "J#64"
        }
      }
    }
    ins(%A, %B : memref<258x512xf32>, memref<512x256xf32>)
    outs(%C : memref<258x256xf32>)
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<258x512xf32> {llvm.noalias}, %arg1: memref<512x256xf32> {llvm.noalias}, %arg2: memref<258x256xf32> {llvm.noalias}) {
// CHECK-NEXT:      linalg.matmul {__node0__} ins(%arg0, %arg1 : memref<258x512xf32>, memref<512x256xf32>) outs(%arg2 : memref<258x256xf32>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__node0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %first, %second = transform.structured.split %0 after 2  {dimension = 0 : i64} : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %second tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__node0__/I[1]/I0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/I[1]/J0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__node0__/I[1]/K0" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_5 "__node0__/I[1]/I1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_7 "__node0__/I[1]/K1" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_9 "__node0__/I[1]/J1" : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
