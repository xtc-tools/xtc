// RUN: mlir-loop --extend --no-alias --print-source-ir %s 2>&1 | filecheck %s

func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
	linalg.matmul {
		loop.dims = ["i", "j"],
		loop.schedule = {
      "One" = {
  		  "i[:5]" = { "Two" = {"j"} },
			  "i[5:]" = { "Two" = {"j"} },
        "fusion"
		}
    }
	}
	ins(%A, %B : memref<256x512xf64>, memref<512x256xf64>)
	outs(%C: memref<256x256xf64>)
	return
}// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @matmul(%arg0: memref<256x512xf64> {llvm.noalias}, %arg1: memref<512x256xf64> {llvm.noalias}, %arg2: memref<256x256xf64> {llvm.noalias}) {
// CHECK-NEXT:      linalg.matmul {__node0__} ins(%arg0, %arg1 : memref<256x512xf64>, memref<512x256xf64>) outs(%arg2 : memref<256x256xf64>)
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
// CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__node0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %first, %second = transform.structured.split %0 after 5  {dimension = 0 : i64} : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %first tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__node0__/i[0]/j" : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %second tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__node0__/i[1]/j" : !transform.any_op
// CHECK-NEXT:      %2 = transform.get_parent_op %loops_1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %3 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
