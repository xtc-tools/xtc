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

impl = Backend(graph)

sch = impl.get_scheduler(default_node="add")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="add_mul_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
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
# CHECK-NEXT:    func.func @add_mul(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<32xf32> {llvm.noalias}, %arg2: memref<4x1xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x32xf32>, memref<32xf32>) outs(%alloca : memref<4x32xf32>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.addf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%alloca, %arg2 : memref<4x32xf32>, memref<4x1xf32>) outs(%arg3 : memref<4x32xf32>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.mulf %in, %in_0 fastmath<fast> : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
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
# CHECK-NEXT:    func.func @add_mul(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<32xf32> {llvm.noalias}, %arg2: memref<4x1xf32> {llvm.noalias}, %arg3: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg4 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg1[0] [32] [1] : memref<32xf32> to memref<32xf32, strided<[1]>>
# CHECK-NEXT:        %subview_4 = memref.subview %alloca[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg5 = %c0_5 to %c32 step %c1_6 {
# CHECK-NEXT:          %subview_7 = memref.subview %subview[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_3[%arg5] [1] [1] : memref<32xf32, strided<[1]>> to memref<1xf32, strided<[1], offset: ?>>
# CHECK-NEXT:          %subview_9 = memref.subview %subview_4[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_7, %subview_8 : memref<1x1xf32, strided<[32, 1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>) outs(%subview_9 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_add_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:            %0 = arith.addf %in, %in_10 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %0 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg4 = %c0_0 to %c4_1 step %c1_2 {
# CHECK-NEXT:        %subview = memref.subview %alloca[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[%arg4, 0] [1, 1] [1, 1] : memref<4x1xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
# CHECK-NEXT:        %subview_4 = memref.subview %arg3[%arg4, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg5 = %c0_5 to %c32 step %c1_6 {
# CHECK-NEXT:          %subview_7 = memref.subview %subview[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_3[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[1, 1], offset: ?>> to memref<1x1xf32, strided<[1, 1], offset: ?>>
# CHECK-NEXT:          %subview_9 = memref.subview %subview_4[0, %arg5] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_7, %subview_8 : memref<1x1xf32, strided<[32, 1], offset: ?>>, memref<1x1xf32, strided<[1, 1], offset: ?>>) outs(%subview_9 : memref<1x1xf32, strided<[32, 1], offset: ?>>) attrs =  {__xtc_id_mul_} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:            %0 = arith.mulf %in, %in_10 fastmath<fast> : f32
# CHECK-NEXT:            linalg.yield %0 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
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
