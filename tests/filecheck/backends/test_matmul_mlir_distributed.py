# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_sdist

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
# Create meshes
sch.define_memory_mesh(axes={"mx": 2, "my": 2})
sch.define_processor_mesh(axes={"px": 2, "py": 2, "psx": 2, "psy": 8})
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.unroll({"i1": 2})
# Add distributed buffer
sch.distributed_buffer_at("k", 1, memory_axes=["mx", "*"])
# Bufferize in local memory
sch.pack_at("i", 1)
# Distribute i over px
sch.distribute("i", "px")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_mlir_distributed",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%arg2 : memref<4x32xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.sdist.create_memory_mesh %arg0 "memory_mesh" = <["mx"=2, "my"=2]> : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %1 = transform.sdist.create_processor_mesh %arg0 "processor_mesh" = <["px"=2, "py"=2, "psx"=2, "psy"=8]> from "memory_mesh" : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.sdist.distribute_buffer_at %3 tensor 1 with ["mx", "*"] on "memory_mesh" : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %3 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./k" : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %tiled_linalg_op_2 {
# CHECK-NEXT:        transform.apply_patterns.memref.fold_memref_alias_ops
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %5 = transform.sdist.local_buffer_at %tiled_linalg_op_2 tensor 1 : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./i1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./j1" : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_7 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %6 = transform.sdist.distribute_loop %forall_op {axis = "px", mesh = "processor_mesh"} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    sdist.processor_mesh @processor_mesh from @memory_mesh = <["px"=2, "py"=2, "psx"=2, "psy"=8]>
# CHECK-NEXT:    sdist.memory_mesh @memory_mesh = <["mx"=2, "my"=2]>
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %c0_2 = arith.constant 0 : index
# CHECK-NEXT:        %c32 = arith.constant 32 : index
# CHECK-NEXT:        %c1_3 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_2 to %c32 step %c1_3 {
# CHECK-NEXT:          %subview_4 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_4 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %0 = sdist.distribute %arg1 ["mx", "*"] on @memory_mesh : memref<512x32xf32>, memref<512x32xf32, 1>
# CHECK-NEXT:      %c0_0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0_0 to %c512 step %c1_1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %0[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32, 1> to memref<1x32xf32, strided<[32, 1], offset: ?>, 1>
# CHECK-NEXT:        %subview_3 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        %alloc = memref.alloc() : memref<1x32xf32, 2>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        sdist.read %0[%arg3, %c0_4] to %alloc : memref<512x32xf32, 1>, memref<1x32xf32, 2>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        sdist.for_distributed %arg4 = %c0_5 to %c2 step [%c1_6] on @processor_mesh("px") {
# CHECK-NEXT:          %1 = affine.apply #map(%arg4)
# CHECK-NEXT:          %subview_7 = memref.subview %subview[%1, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %alloc[0, 0] [1, 32] [1, 1] : memref<1x32xf32, 2> to memref<1x32xf32, strided<[32, 1]>, 2>
# CHECK-NEXT:          %subview_9 = memref.subview %subview_3[%1, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %c0_10 = arith.constant 0 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_10 to %c32 step %c16 {
# CHECK-NEXT:            %subview_11 = memref.subview %subview_7[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_12 = memref.subview %subview_8[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1]>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_13 = memref.subview %subview_9[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_14 = arith.constant 0 : index
# CHECK-NEXT:            %c2_15 = arith.constant 2 : index
# CHECK-NEXT:            %c1_16 = arith.constant 1 : index
# CHECK-NEXT:            %c2_17 = arith.constant 2 : index
# CHECK-NEXT:            %subview_18 = memref.subview %subview_11[%c0_14, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_19 = memref.subview %subview_12[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_20 = memref.subview %subview_13[%c0_14, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_21 = arith.constant 0 : index
# CHECK-NEXT:            %c16_22 = arith.constant 16 : index
# CHECK-NEXT:            %c1_23 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_21 to %c16_22 step %c1_23 {
# CHECK-NEXT:              %subview_31 = memref.subview %subview_18[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_32 = memref.subview %subview_19[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x1xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:              %subview_33 = memref.subview %subview_20[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_31, %subview_32 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>, 2>) outs(%subview_33 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:            %c1_24 = arith.constant 1 : index
# CHECK-NEXT:            %2 = arith.muli %c1_16, %c1_24 : index
# CHECK-NEXT:            %3 = arith.addi %c0_14, %2 : index
# CHECK-NEXT:            %subview_25 = memref.subview %subview_11[%3, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_26 = memref.subview %subview_12[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x16xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:            %subview_27 = memref.subview %subview_13[%3, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_28 = arith.constant 0 : index
# CHECK-NEXT:            %c16_29 = arith.constant 16 : index
# CHECK-NEXT:            %c1_30 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg6 = %c0_28 to %c16_29 step %c1_30 {
# CHECK-NEXT:              %subview_31 = memref.subview %subview_25[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:              %subview_32 = memref.subview %subview_26[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>, 2> to memref<1x1xf32, strided<[32, 1], offset: ?>, 2>
# CHECK-NEXT:              %subview_33 = memref.subview %subview_27[0, %arg6] [1, 1] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x1xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_31, %subview_32 : memref<1x1xf32, strided<[512, 1], offset: ?>>, memref<1x1xf32, strided<[32, 1], offset: ?>, 2>) outs(%subview_33 : memref<1x1xf32, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:            } {"./j1"}
# CHECK-NEXT:          } {"./j"}
# CHECK-NEXT:          sdist.terminator
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"./k"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
