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

impl = Backend(graph)

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
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @relu(%arg0: memref<128xf32> {llvm.noalias}, %arg1: memref<128xf32> {llvm.noalias}) {
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %arg0 [[0]] : memref<128xf32> into memref<128xf32>
# CHECK-NEXT:      %collapse_shape_0 = memref.collapse_shape %arg1 [[0]] : memref<128xf32> into memref<128xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapse_shape, %cst : memref<128xf32>, f32) outs(%collapse_shape_0 : memref<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.maximumf %in, %in_1 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%tiled_linalg_op) : (!transform.any_op) -> ()
# CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %1 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %1 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @relu(%arg0: memref<128xf32> {llvm.noalias}, %arg1: memref<128xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
# CHECK-NEXT:      %0 = ub.poison : f32
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      scf.for %arg2 = %c0 to %c128 step %c16 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg2] [16] [1] : memref<128xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[%arg2] [16] [1] : memref<128xf32> to memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<16xf32>
# CHECK-NEXT:        %2 = arith.maximumf %1, %cst : vector<16xf32>
# CHECK-NEXT:        vector.transfer_write %2, %subview_0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1], offset: ?>>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 128xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 128xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'relu'} : [128xfloat32] -> [128xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
