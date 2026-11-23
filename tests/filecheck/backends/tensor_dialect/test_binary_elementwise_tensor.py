# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A")
b = O.tensor((J,), dtype, name="B")
c = O.tensor((I, 1), dtype, name="C")

with O.graph(name="add_mul") as gb:
    x = O.add(a, b, name="add")
    O.mul(x, c, name="mul")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="add")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="add_mul_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1) -> (d0, 0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @add_mul(%arg0: tensor<4x32xf32> {llvm.noalias}, %arg1: tensor<32xf32> {llvm.noalias}, %arg2: tensor<4x1xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x32xf32>, tensor<32xf32>) outs(%0 : tensor<4x32xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %4 = arith.addf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %4 : f32
# CHECK-NEXT:      } -> tensor<4x32xf32>
# CHECK-NEXT:      %2 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %3 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg2 : tensor<4x32xf32>, tensor<4x1xf32>) outs(%2 : tensor<4x32xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %4 = arith.mulf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %4 : f32
# CHECK-NEXT:      } -> tensor<4x32xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %3 in restrict writable %arg3 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_add_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_mul_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1) -> (d0, 0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @add_mul(%arg0: tensor<4x32xf32> {llvm.noalias}, %arg1: tensor<32xf32> {llvm.noalias}, %arg2: tensor<4x1xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0) -> (tensor<4x32xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg1[0] [32] [1] : tensor<32xf32> to tensor<32xf32>
# CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        %4 = scf.for %arg6 = %c0_5 to %c32 step %c1_6 iter_args(%arg7 = %extracted_slice_4) -> (tensor<1x32xf32>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[%arg6] [1] [1] : tensor<32xf32> to tensor<1xf32>
# CHECK-NEXT:          %extracted_slice_9 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_7, %extracted_slice_8 : tensor<1x1xf32>, tensor<1xf32>) outs(%extracted_slice_9 : tensor<1x1xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_11: f32, %out: f32):
# CHECK-NEXT:            %6 = arith.addf %in, %in_11 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %6 : f32
# CHECK-NEXT:          } -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_10 = tensor.insert_slice %5 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_10 : tensor<1x32xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %2 = tensor.empty() : tensor<4x32xf32>
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %3 = scf.for %arg4 = %c0_0 to %c4_1 step %c1_2 iter_args(%arg5 = %2) -> (tensor<4x32xf32>) {
# CHECK-NEXT:        %extracted_slice = tensor.extract_slice %1[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0] [1, 1] [1, 1] : tensor<4x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:        %extracted_slice_4 = tensor.extract_slice %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        %4 = scf.for %arg6 = %c0_5 to %c32 step %c1_6 iter_args(%arg7 = %extracted_slice_4) -> (tensor<1x32xf32>) {
# CHECK-NEXT:          %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %extracted_slice_9 = tensor.extract_slice %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_7, %extracted_slice_8 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_9 : tensor<1x1xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_11: f32, %out: f32):
# CHECK-NEXT:            %6 = arith.mulf %in, %in_11 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %6 : f32
# CHECK-NEXT:          } -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_10 = tensor.insert_slice %5 into %arg7[0, %arg6] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_10 : tensor<1x32xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
# CHECK-NEXT:        scf.yield %inserted_slice : tensor<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      bufferization.materialize_in_destination %3 in restrict writable %arg3 : (tensor<4x32xf32>, memref<4x32xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1) -> (d0, 0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @add_mul(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<32xf32> {llvm.noalias}, %arg2: memref<4x1xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      %0 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %alloca) -> (memref<4x32xf32>) {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview_0) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %arg1[%arg6] [1] [1] : memref<32xf32> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_2, %subview_3 : memref<1x1xf32, strided<[32, 1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>) outs(%subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_6: f32, %out: f32):
# CHECK-NEXT:            %3 = arith.addf %in, %in_6 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %3 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_5 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_4, %subview_5 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_1 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_1 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (memref<4x32xf32>) {
# CHECK-NEXT:        %subview = memref.subview %0[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg2[%arg4, 0] [1, 1] [1, 1] : memref<4x1xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %subview_1) -> (memref<1x32xf32, strided<[32, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_3 = memref.subview %subview[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_4 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_3, %subview_0 : memref<1x1xf32, strided<[32, 1], offset: ?>>, memref<1x1xf32, strided<[1, 1], offset: ?>>) outs(%subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_6: f32, %out: f32):
# CHECK-NEXT:            %3 = arith.mulf %in, %in_6 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %3 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_5 = memref.subview %arg7[0, %arg6] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_4, %subview_5 : memref<1x1xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg7 : memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg5[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_2 : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg5 : memref<4x32xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      memref.copy %1, %arg3 : memref<4x32xf32> to memref<4x32xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: add_mul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x32xfloat32
# CHECK-NEXT:    - %1 : 32xfloat32
# CHECK-NEXT:    - %2 : 4x1xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %3: add(%0, %1) {name = 'add'} : [4x32xfloat32, 32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:    - %4: mul(%3, %2) {name = 'mul'} : [4x32xfloat32, 4x1xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
