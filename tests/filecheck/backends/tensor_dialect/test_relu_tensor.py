# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, dtype = 128, "float32"
a = O.tensor((I,), dtype, name="A")

with O.graph(name="relu") as gb:
    O.relu(a, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="relu")
sch.tile("i", {"i1": 16})
sch.interchange([ "i", "i1"])
sch.vectorize(["i1"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="relu_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK: // -----// IR Dump Before transform //----- //
# CHECK-NEXT: #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @relu(%arg0: tensor<128xf32> {llvm.noalias}, %arg1: memref<128xf32> {llvm.noalias}) {
# CHECK-NEXT:     %0 = tensor.empty() : tensor<128xf32>
# CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg0, %cst : tensor<128xf32>, f32) outs(%0 : tensor<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:     ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:       %2 = arith.maximumf %in, %in_0 : f32
# CHECK-NEXT:       linalg.yield %2 : f32
# CHECK-NEXT:     } -> tensor<128xf32>
# CHECK-NEXT:     bufferization.materialize_in_destination %1 in restrict writable %arg1 : (tensor<128xf32>, memref<128xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {sym_name = "relu"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %0 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:     transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:     %1 = transform.get_parent_op %tiled_linalg_op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %1 {
# CHECK-NEXT:       transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     %2 = transform.structured.match interface{LinalgOp} in %1 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.include @_vecto failures(suppress) (%2) : (!transform.any_op) -> ()
# CHECK-NEXT:     %3 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %3 {
# CHECK-NEXT:       transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:       transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After transform //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @relu(%arg0: tensor<128xf32> {llvm.noalias}, %arg1: memref<128xf32> {llvm.noalias}) {
# CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
# CHECK-NEXT:     %0 = ub.poison : f32
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c128 = arith.constant 128 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %1 = tensor.empty() : tensor<128xf32>
# CHECK-NEXT:     %2 = scf.for %arg2 = %c0 to %c128 step %c16 iter_args(%arg3 = %1) -> (tensor<128xf32>) {
# CHECK-NEXT:       %extracted_slice = tensor.extract_slice %arg0[%arg2] [16] [1] : tensor<128xf32> to tensor<16xf32>
# CHECK-NEXT:       %extracted_slice_0 = tensor.extract_slice %arg3[%arg2] [16] [1] : tensor<128xf32> to tensor<16xf32>
# CHECK-NEXT:       %3 = vector.transfer_read %extracted_slice[%c0], %0 {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
# CHECK-NEXT:       %4 = arith.maximumf %3, %cst : vector<16xf32>
# CHECK-NEXT:       %5 = vector.transfer_write %4, %extracted_slice_0[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
# CHECK-NEXT:       %inserted_slice = tensor.insert_slice %5 into %arg3[%arg2] [16] [1] : tensor<16xf32> into tensor<128xf32>
# CHECK-NEXT:       scf.yield %inserted_slice : tensor<128xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     bufferization.materialize_in_destination %2 in restrict writable %arg1 : (tensor<128xf32>, memref<128xf32>) -> ()
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:     transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT:   transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:     %0 = transform.structured.match attributes {sym_name = "relu"} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:     transform.apply_patterns to %0 {
# CHECK-NEXT:       transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:       transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:     } : !transform.any_op
# CHECK-NEXT:     transform.yield 
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT: module attributes {transform.with_named_sequence} {
# CHECK-NEXT:   func.func @relu(%arg0: memref<128xf32> {llvm.noalias}, %arg1: memref<128xf32> {llvm.noalias}) {
# CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
# CHECK-NEXT:     %0 = ub.poison : f32
# CHECK-NEXT:     %c16 = arith.constant 16 : index
# CHECK-NEXT:     %c128 = arith.constant 128 : index
# CHECK-NEXT:     %c0 = arith.constant 0 : index
# CHECK-NEXT:     %1 = scf.for %arg2 = %c0 to %c128 step %c16 iter_args(%arg3 = %arg1) -> (memref<128xf32>) {
# CHECK-NEXT:       %subview = memref.subview %arg0[%arg2] [16] [1] : memref<128xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       %subview_0 = memref.subview %arg3[%arg2] [16] [1] : memref<128xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       %2 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:       %3 = arith.maximumf %2, %cst : vector<16xf32>
# CHECK-NEXT:       vector.transfer_write %3, %subview_0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       %subview_1 = memref.subview %arg3[%arg2] [16] [1] : memref<128xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       memref.copy %subview_0, %subview_1 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:       scf.yield %arg3 : memref<128xf32>
# CHECK-NEXT:     } {"./i"}
# CHECK-NEXT:     memref.copy %1, %arg1 : memref<128xf32> to memref<128xf32>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT:  
# CHECK-NEXT: graph:
# CHECK-NEXT:   name: relu
# CHECK-NEXT:   inputs:
# CHECK-NEXT:   - %0 : 128xfloat32
# CHECK-NEXT:   outputs:
# CHECK-NEXT:   - %1 : 128xfloat32
# CHECK-NEXT:   nodes:
# CHECK-NEXT:   - %1: relu(%0) {name = 'relu'} : [128xfloat32] -> [128xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT: CODE: 0
