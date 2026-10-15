// RUN: mlir-loop --no-alias --print-transformed-ir --print-bufferization-ir %s 2>&1 | filecheck %s
// UNSUPPORTED: mlir-target=nvgpu

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
  func.func @matmul_relu(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
    %0 = tensor.empty() : tensor<4x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = linalg.matmul
        {
          loop.dims = ["I","J","K"],
          loop.schedule = {
            "I",
              "J",
                  "I#2",
                      "J#16" = {"fuse_consumer"},
                      "K"
          }
        }
    	ins(%arg0, %arg1 : tensor<4x512xf32>, tensor<512x32xf32>) 
    	outs(%1 : tensor<4x32xf32>) -> tensor<4x32xf32>

    %3 = tensor.empty() : tensor<4x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.generic {
		indexing_maps = [#map, #map1, #map], 
		iterator_types = ["parallel", "parallel"]} 
		ins(%2, %cst_0 : tensor<4x32xf32>, f32) 
		outs(%3 : tensor<4x32xf32>) 
		attrs =  {__xtc_id_relu_} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.maximumf %in, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<4x32xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
    return
  }


// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> ()>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @matmul_relu(%arg0: tensor<4x512xf32> {llvm.noalias}, %arg1: tensor<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
// CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf32>
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %1 = linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %2 = tensor.empty() : tensor<4x32xf32>
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %3:2 = scf.for %arg3 = %c0 to %c4 step %c2 iter_args(%arg4 = %1, %arg5 = %2) -> (tensor<4x32xf32>, tensor<4x32xf32>) {
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [2, 512] [1, 1] : tensor<4x512xf32> to tensor<2x512xf32>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x32xf32> to tensor<512x32xf32>
// CHECK-NEXT:        %extracted_slice_2 = tensor.extract_slice %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
// CHECK-NEXT:        %c0_3 = arith.constant 0 : index
// CHECK-NEXT:        %c32 = arith.constant 32 : index
// CHECK-NEXT:        %c16 = arith.constant 16 : index
// CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<4x32xf32> to tensor<2x32xf32>
// CHECK-NEXT:        %5:2 = scf.for %arg6 = %c0_3 to %c32 step %c16 iter_args(%arg7 = %extracted_slice_2, %arg8 = %extracted_slice_4) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
// CHECK-NEXT:          %extracted_slice_6 = tensor.extract_slice %extracted_slice[0, 0] [2, 512] [1, 1] : tensor<2x512xf32> to tensor<2x512xf32>
// CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice_1[0, %arg6] [512, 16] [1, 1] : tensor<512x32xf32> to tensor<512x16xf32>
// CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %arg7[0, %arg6] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
// CHECK-NEXT:          %c0_9 = arith.constant 0 : index
// CHECK-NEXT:          %c2_10 = arith.constant 2 : index
// CHECK-NEXT:          %c1 = arith.constant 1 : index
// CHECK-NEXT:          %extracted_slice_11 = tensor.extract_slice %arg7[0, %arg6] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
// CHECK-NEXT:          %7:2 = scf.for %arg9 = %c0_9 to %c2_10 step %c1 iter_args(%arg10 = %extracted_slice_8, %arg11 = %extracted_slice_11) -> (tensor<2x16xf32>, tensor<2x16xf32>) {
// CHECK-NEXT:            %extracted_slice_14 = tensor.extract_slice %extracted_slice_6[%arg9, 0] [1, 512] [1, 1] : tensor<2x512xf32> to tensor<1x512xf32>
// CHECK-NEXT:            %extracted_slice_15 = tensor.extract_slice %extracted_slice_7[0, 0] [512, 16] [1, 1] : tensor<512x16xf32> to tensor<512x16xf32>
// CHECK-NEXT:            %extracted_slice_16 = tensor.extract_slice %arg10[%arg9, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
// CHECK-NEXT:            %c0_17 = arith.constant 0 : index
// CHECK-NEXT:            %c16_18 = arith.constant 16 : index
// CHECK-NEXT:            %c1_19 = arith.constant 1 : index
// CHECK-NEXT:            %extracted_slice_20 = tensor.extract_slice %arg10[%arg9, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
// CHECK-NEXT:            %9:2 = scf.for %arg12 = %c0_17 to %c16_18 step %c1_19 iter_args(%arg13 = %extracted_slice_16, %arg14 = %extracted_slice_20) -> (tensor<1x16xf32>, tensor<1x16xf32>) {
// CHECK-NEXT:              %extracted_slice_23 = tensor.extract_slice %extracted_slice_14[0, 0] [1, 512] [1, 1] : tensor<1x512xf32> to tensor<1x512xf32>
// CHECK-NEXT:              %extracted_slice_24 = tensor.extract_slice %extracted_slice_15[0, %arg12] [512, 1] [1, 1] : tensor<512x16xf32> to tensor<512x1xf32>
// CHECK-NEXT:              %extracted_slice_25 = tensor.extract_slice %arg13[0, %arg12] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
// CHECK-NEXT:              %c0_26 = arith.constant 0 : index
// CHECK-NEXT:              %c512 = arith.constant 512 : index
// CHECK-NEXT:              %c1_27 = arith.constant 1 : index
// CHECK-NEXT:              %11 = scf.for %arg15 = %c0_26 to %c512 step %c1_27 iter_args(%arg16 = %extracted_slice_25) -> (tensor<1x1xf32>) {
// CHECK-NEXT:                %extracted_slice_31 = tensor.extract_slice %extracted_slice_23[0, %arg15] [1, 1] [1, 1] : tensor<1x512xf32> to tensor<1x1xf32>
// CHECK-NEXT:                %extracted_slice_32 = tensor.extract_slice %extracted_slice_24[%arg15, 0] [1, 1] [1, 1] : tensor<512x1xf32> to tensor<1x1xf32>
// CHECK-NEXT:                %extracted_slice_33 = tensor.extract_slice %arg16[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
// CHECK-NEXT:                %13 = linalg.matmul {__node0__} ins(%extracted_slice_31, %extracted_slice_32 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_33 : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:                %inserted_slice_34 = tensor.insert_slice %13 into %arg16[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
// CHECK-NEXT:                scf.yield %inserted_slice_34 : tensor<1x1xf32>
// CHECK-NEXT:              } {"__node0__/K"}
// CHECK-NEXT:              %extracted_slice_28 = tensor.extract_slice %arg13[0, %arg12] [1, 1] [1, 1] : tensor<1x16xf32> to tensor<1x1xf32>
// CHECK-NEXT:              %12 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_0 : tensor<1x1xf32>, f32) outs(%11 : tensor<1x1xf32>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:              ^bb0(%in: f32, %in_31: f32, %out: f32):
// CHECK-NEXT:                %13 = arith.maximumf %in, %in_31 : f32
// CHECK-NEXT:                linalg.yield %13 : f32
// CHECK-NEXT:              } -> tensor<1x1xf32>
// CHECK-NEXT:              %inserted_slice_29 = tensor.insert_slice %12 into %arg13[0, %arg12] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
// CHECK-NEXT:              %inserted_slice_30 = tensor.insert_slice %12 into %arg13[0, %arg12] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x16xf32>
// CHECK-NEXT:              scf.yield %inserted_slice_29, %inserted_slice_30 : tensor<1x16xf32>, tensor<1x16xf32>
// CHECK-NEXT:            } {"__node0__/J0"}
// CHECK-NEXT:            %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9#0, %cst_0 : tensor<1x16xf32>, f32) outs(%extracted_slice_20 : tensor<1x16xf32>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:            ^bb0(%in: f32, %in_23: f32, %out: f32):
// CHECK-NEXT:              %11 = arith.maximumf %in, %in_23 : f32
// CHECK-NEXT:              linalg.yield %11 : f32
// CHECK-NEXT:            } -> tensor<1x16xf32>
// CHECK-NEXT:            %inserted_slice_21 = tensor.insert_slice %9#0 into %arg10[%arg9, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
// CHECK-NEXT:            %inserted_slice_22 = tensor.insert_slice %9#0 into %arg10[%arg9, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
// CHECK-NEXT:            scf.yield %inserted_slice_21, %inserted_slice_22 : tensor<2x16xf32>, tensor<2x16xf32>
// CHECK-NEXT:          } {"__node0__/I0"}
// CHECK-NEXT:          %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7#0, %cst_0 : tensor<2x16xf32>, f32) outs(%extracted_slice_11 : tensor<2x16xf32>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:          ^bb0(%in: f32, %in_14: f32, %out: f32):
// CHECK-NEXT:            %9 = arith.maximumf %in, %in_14 : f32
// CHECK-NEXT:            linalg.yield %9 : f32
// CHECK-NEXT:          } -> tensor<2x16xf32>
// CHECK-NEXT:          %inserted_slice_12 = tensor.insert_slice %7#0 into %arg7[0, %arg6] [2, 16] [1, 1] : tensor<2x16xf32> into tensor<2x32xf32>
// CHECK-NEXT:          %inserted_slice_13 = tensor.insert_slice %7#0 into %arg7[0, %arg6] [2, 16] [1, 1] : tensor<2x16xf32> into tensor<2x32xf32>
// CHECK-NEXT:          scf.yield %inserted_slice_12, %inserted_slice_13 : tensor<2x32xf32>, tensor<2x32xf32>
// CHECK-NEXT:        } {"__node0__/J"}
// CHECK-NEXT:        %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5#0, %cst_0 : tensor<2x32xf32>, f32) outs(%extracted_slice_4 : tensor<2x32xf32>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:        ^bb0(%in: f32, %in_6: f32, %out: f32):
// CHECK-NEXT:          %7 = arith.maximumf %in, %in_6 : f32
// CHECK-NEXT:          linalg.yield %7 : f32
// CHECK-NEXT:        } -> tensor<2x32xf32>
// CHECK-NEXT:        %inserted_slice = tensor.insert_slice %5#0 into %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<4x32xf32>
// CHECK-NEXT:        %inserted_slice_5 = tensor.insert_slice %5#0 into %arg4[%arg3, 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<4x32xf32>
// CHECK-NEXT:        scf.yield %inserted_slice, %inserted_slice_5 : tensor<4x32xf32>, tensor<4x32xf32>
// CHECK-NEXT:      } {"__node0__/I"}
// CHECK-NEXT:      %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3#0, %cst_0 : tensor<4x32xf32>, f32) outs(%2 : tensor<4x32xf32>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:        %5 = arith.maximumf %in, %in_1 : f32
// CHECK-NEXT:        linalg.yield %5 : f32
// CHECK-NEXT:      } -> tensor<4x32xf32>
// CHECK-NEXT:      bufferization.materialize_in_destination %3#0 in restrict writable %arg2 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
// CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  
// CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> ()>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
// CHECK-NEXT:      %c512 = arith.constant 512 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c16 = arith.constant 16 : index
// CHECK-NEXT:      %c32 = arith.constant 32 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
// CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c4 step %c2 iter_args(%arg4 = %arg2) -> (memref<4x32xf32>) {
// CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [2, 512] [1, 1] : memref<4x512xf32> to memref<2x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:        %subview_0 = memref.subview %arg4[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:        %1 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %subview_0) -> (memref<2x32xf32, strided<[32, 1], offset: ?>>) {
// CHECK-NEXT:          %subview_2 = memref.subview %arg1[0, %arg5] [512, 16] [1, 1] : memref<512x32xf32> to memref<512x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:          %subview_3 = memref.subview %arg6[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:          %2 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %subview_3) -> (memref<2x16xf32, strided<[32, 1], offset: ?>>) {
// CHECK-NEXT:            %subview_5 = memref.subview %subview[%arg7, 0] [1, 512] [1, 1] : memref<2x512xf32, strided<[512, 1], offset: ?>> to memref<1x512xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:            %subview_6 = memref.subview %arg8[%arg7, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:            %3 = scf.for %arg9 = %c0 to %c16 step %c1 iter_args(%arg10 = %subview_6) -> (memref<1x16xf32, strided<[32, 1], offset: ?>>) {
// CHECK-NEXT:              %subview_8 = memref.subview %subview_2[0, %arg9] [512, 1] [1, 1] : memref<512x16xf32, strided<[32, 1], offset: ?>> to memref<512x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:              %subview_9 = memref.subview %arg10[0, %arg9] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:              %4 = scf.for %arg11 = %c0 to %c512 step %c1 iter_args(%arg12 = %subview_9) -> (memref<1x1xf32, strided<[32, 1], offset: ?>>) {
// CHECK-NEXT:                %subview_11 = memref.subview %subview_5[0, %arg11] [1, 1] [1, 1] : memref<1x512xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
// CHECK-NEXT:                %subview_12 = memref.subview %subview_8[%arg11, 0] [1, 1] [1, 1] : memref<512x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:                linalg.matmul {__node0__} ins(%subview_11, %subview_12 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>>) outs(%arg12 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
// CHECK-NEXT:                scf.yield %arg12 : memref<1x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:              } {"__node0__/K"}
// CHECK-NEXT:              linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : memref<1x1xf32, strided<[32, 1], offset: ?>>, f32) outs(%4 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_relu_} {
// CHECK-NEXT:              ^bb0(%in: f32, %in_11: f32, %out: f32):
// CHECK-NEXT:                %5 = arith.maximumf %in, %in_11 : f32
// CHECK-NEXT:                linalg.yield %5 : f32
// CHECK-NEXT:              }
// CHECK-NEXT:              %subview_10 = memref.subview %arg10[0, %arg9] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:              memref.copy %4, %subview_10 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:              scf.yield %arg10 : memref<1x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:            } {"__node0__/J0"}
// CHECK-NEXT:            %subview_7 = memref.subview %arg8[%arg7, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:            memref.copy %3, %subview_7 : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:            scf.yield %arg8 : memref<2x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:          } {"__node0__/I0"}
// CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:          memref.copy %2, %subview_4 : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:          scf.yield %arg6 : memref<2x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:        } {"__node0__/J"}
// CHECK-NEXT:        %subview_1 = memref.subview %arg4[%arg3, 0] [2, 32] [1, 1] : memref<4x32xf32> to memref<2x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:        memref.copy %1, %subview_1 : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NEXT:        scf.yield %arg4 : memref<4x32xf32>
// CHECK-NEXT:      } {"__node0__/I"}
// CHECK-NEXT:      memref.copy %0, %arg2 : memref<4x32xf32> to memref<4x32xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
