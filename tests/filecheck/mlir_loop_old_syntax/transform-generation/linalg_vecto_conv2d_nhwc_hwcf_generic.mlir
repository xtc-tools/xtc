// RUN: mlir-loop --old-syntax %s --vectors-size 8 --no-alias --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %I: memref<1x30x30x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x28x128xf32>
) {
  linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
      ],
      iterator_types = [
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "reduction",
        "reduction",
        "reduction"
      ]
  }
  ins (%I, %K : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>)
  outs(%O : memref<1x28x28x128xf32>)
  attrs = {
    loop.dims = ["n","h","w","f","r","s","c"],
    loop.vectorize = ["n"]
  }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}

// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>) outs(%arg2 : memref<1x28x28x128xf32>) attrs =  {__id0__} {
// CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK-NEXT:        %0 = arith.mulf %in, %in_0 : f32
// CHECK-NEXT:        %1 = arith.addf %out, %0 : f32
// CHECK-NEXT:        linalg.yield %1 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      %3 = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %2 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %4 = transform.apply_registered_pass "affine-super-vectorize" to %3 {options = "virtual-vector-size=8"} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
