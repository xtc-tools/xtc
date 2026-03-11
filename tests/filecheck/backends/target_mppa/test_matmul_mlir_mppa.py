# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_mppa
# REQUIRES: mlir-target=mppa

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.mppa import MppaDevice

I, J, K, dtype = 4, 8, 16, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.define_memory_mesh(axes={"mx": 1, "my": 1})
sch.define_processor_mesh(axes={"px": 1, "py": 1, "psx": 2, "psy": 8})
sch.tile("i", {"i1": 2})
sch.pack_at("i1", 1)
sched = sch.schedule()

# Create mppa device
mppa = MppaDevice()

comp = impl.get_compiler(
    target=mppa,
    shared_lib=True,
    dump_file="matmul_mlir_mppa",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x16xf32> {llvm.noalias}, %arg1: memref<16x8xf32> {llvm.noalias}, %arg2: memref<4x8xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<4x8xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<4x16xf32>, memref<16x8xf32>) outs(%arg2 : memref<4x8xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.sdist.create_memory_mesh %arg0 "memory_mesh" = <["mx"=1, "my"=1]> : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %1 = transform.sdist.create_processor_mesh %arg0 "processor_mesh" = <["px"=1, "py"=1, "psx"=2, "psy"=8]> from "memory_mesh" : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %3 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k" : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %tiled_linalg_op_6 {
# CHECK-NEXT:        transform.apply_patterns.memref.fold_memref_alias_ops
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %4 = transform.sdist.local_buffer_at %tiled_linalg_op_6 tensor 1 : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i1" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    sdist.processor_mesh @processor_mesh from @memory_mesh = <["px"=1, "py"=1, "psx"=2, "psy"=8]>
# CHECK-NEXT:    sdist.memory_mesh @memory_mesh = <["mx"=1, "my"=1]>
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x16xf32> {llvm.noalias}, %arg1: memref<16x8xf32> {llvm.noalias}, %arg2: memref<4x8xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_3 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c8 step %c1_3 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<1x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_4 : memref<1x1xf32, strided<[8, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c4_1 step %c2 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [2, 16] [1, 1] : memref<4x16xf32> to memref<2x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[0, 0] [16, 8] [1, 1] : memref<16x8xf32> to memref<16x8xf32, strided<[8, 1]>>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[%arg3, 0] [2, 8] [1, 1] : memref<4x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c8 = arith.constant 8 : index
# CHECK-NEXT:        %c1_5 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c8 step %c1_5 {
# CHECK-NEXT:          %subview_6 = memref.subview %subview[0, 0] [2, 16] [1, 1] : memref<2x16xf32, strided<[16, 1], offset: ?>> to memref<2x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_7 = memref.subview %subview_2[0, %arg4] [16, 1] [1, 1] : memref<16x8xf32, strided<[8, 1]>> to memref<16x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %subview_3[0, %arg4] [2, 1] [1, 1] : memref<2x8xf32, strided<[8, 1], offset: ?>> to memref<2x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:          %c0_9 = arith.constant 0 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          %c1_10 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_9 to %c16 step %c1_10 {
# CHECK-NEXT:            %subview_11 = memref.subview %subview_6[0, %arg5] [2, 1] [1, 1] : memref<2x16xf32, strided<[16, 1], offset: ?>> to memref<2x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_7[%arg5, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[8, 1], offset: ?>> to memref<1x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_8[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[8, 1], offset: ?>> to memref<2x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:            %alloc = memref.alloc() : memref<1x1xf32, 2>
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            sdist.read %subview_7[%arg5, %c0_14] to %alloc : memref<16x1xf32, strided<[8, 1], offset: ?>>, memref<1x1xf32, 2>
# CHECK-NEXT:            %c0_15 = arith.constant 0 : index
# CHECK-NEXT:            %c2_16 = arith.constant 2 : index
# CHECK-NEXT:            %c1_17 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_15 to %c2_16 step %c1_17 {
# CHECK-NEXT:              %subview_18 = memref.subview %subview_11[%arg6, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:              %subview_19 = memref.subview %alloc[0, 0] [1, 1] [1, 1] : memref<1x1xf32, 2> to memref<1x1xf32, strided<[1, 1]>, 2>
# CHECK-NEXT:              %subview_20 = memref.subview %subview_13[%arg6, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[8, 1], offset: ?>> to memref<1x1xf32, strided<[8, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_18, %subview_19 : memref<1x1xf32, strided<[16, 1], offset: ?>>, memref<1x1xf32, strided<[1, 1]>, 2>) outs(%subview_20 : memref<1x1xf32, strided<[8, 1], offset: ?>>)
# CHECK-NEXT:            } {"./i1"}
# CHECK-NEXT:          } {"./k"}
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x16xfloat32
# CHECK-NEXT:    - %1 : 16x8xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x8xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x16xfloat32, 16x8xfloat32] -> [4x8xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
