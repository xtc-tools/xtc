# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_mppa
# REQUIRES: mlir-target=mppa

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.mppa import MppaDevice

I, J, K, dtype = 16, 16, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.define_memory_mesh(axes={"mx": 1, "my": 1})
sch.define_processor_mesh(axes={"px": 1, "py": 1, "psx": 1, "psy": 1})
sch.tile("i", {"i1": 8})
sch.tile("j", {"j1": 8})
sch.interchange(["i", "j", "i1", "j1", "k"])
sch.vectorize(["i1", "j1", "k"])
#sch.pack_at("i1", 1)
sched = sch.schedule()

# Create mppa device
mppa = MppaDevice()

comp = impl.get_compiler(
    target=mppa,
    shared_lib=True,
    dump_file="matmul_mlir_mppa",
    print_source_ir=True,
    print_transformed_ir=True,
    print_lowered_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<16x64xf32> {llvm.noalias}, %arg1: memref<64x16xf32> {llvm.noalias}, %arg2: memref<16x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<16x16xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%arg2 : memref<16x16xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.sdist.create_memory_mesh %arg0 "memory_mesh" = <["mx"=1, "my"=1]> : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %1 = transform.sdist.create_processor_mesh %arg0 "processor_mesh" = <["px"=1, "py"=1, "psx"=1, "psy"=1]> from "memory_mesh" : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %3 tile_sizes [8, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [0, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./j" : !transform.any_op
# CHECK-NEXT:      transform.annotate %tiled_linalg_op_4 "xtc.request_vectorization" : !transform.any_op
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    sdist.processor_mesh @processor_mesh from @memory_mesh = <["px"=1, "py"=1, "psx"=1, "psy"=1]>
# CHECK-NEXT:    sdist.memory_mesh @memory_mesh = <["mx"=1, "my"=1]>
# CHECK-NEXT:    func.func @matmul(%arg0: memref<16x64xf32> {llvm.noalias}, %arg1: memref<64x16xf32> {llvm.noalias}, %arg2: memref<16x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c16 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 16] [1, 1] : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c16 step %c1 {
# CHECK-NEXT:          %subview_0 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>> to memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_0 : memref<1x1xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c16 step %c8 {
# CHECK-NEXT:        %subview = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<16x64xf32> to memref<8x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[0, 0] [64, 16] [1, 1] : memref<64x16xf32> to memref<64x16xf32, strided<[16, 1]>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[%arg3, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c16 step %c8 {
# CHECK-NEXT:          %subview_2 = memref.subview %subview_0[0, %arg4] [64, 8] [1, 1] : memref<64x16xf32, strided<[16, 1]>> to memref<64x8xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[0, %arg4] [8, 8] [1, 1] : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x8xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:          linalg.matmul {__xtc_id_C_, xtc.request_vectorization} ins(%subview, %subview_2 : memref<8x64xf32, strided<[64, 1], offset: ?>>, memref<64x8xf32, strided<[16, 1], offset: ?>>) outs(%subview_3 : memref<8x8xf32, strided<[16, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After MLIR Opt //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-NEXT:  "builtin.module"() ({
# CHECK-NEXT:    "func.func"() <{arg_attrs = [{llvm.noalias}, {llvm.noalias}, {llvm.noalias}], function_type = (memref<16x64xf32>, memref<64x16xf32>, memref<16x16xf32>) -> (), sym_name = "matmul"}> ({
# CHECK-NEXT:    ^bb0(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>):
# CHECK-NEXT:      "mppa.launch"() ({
# CHECK-NEXT:        "kvxcluster.launch"() ({
# CHECK-NEXT:        ^bb0(%arg3: index):
# CHECK-NEXT:          %0 = "arith.constant"() <{value = 1 : index}> : () -> index
# CHECK-NEXT:          %1 = "arith.constant"() <{value = 16 : index}> : () -> index
# CHECK-NEXT:          %2 = "arith.constant"() <{value = 0 : index}> : () -> index
# CHECK-NEXT:          %3 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
# CHECK-NEXT:          %4 = "arith.constant"() <{value = 8 : index}> : () -> index
# CHECK-NEXT:          "scf.for"(%2, %1, %0) ({
# CHECK-NEXT:          ^bb0(%arg9: index):
# CHECK-NEXT:            %11 = "memref.subview"(%arg2, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>}> : (memref<16x16xf32>, index) -> memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            "scf.for"(%2, %1, %0) ({
# CHECK-NEXT:            ^bb0(%arg10: index):
# CHECK-NEXT:              %12 = "memref.subview"(%11, %arg10) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>}> : (memref<1x16xf32, strided<[16, 1], offset: ?>>, index) -> memref<1x1xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:              "linalg.fill"(%3, %12) <{operandSegmentSizes = array<i32: 1, 1>}> ({
# CHECK-NEXT:              ^bb0(%arg11: f32, %arg12: f32):
# CHECK-NEXT:                "linalg.yield"(%arg11) : (f32) -> ()
# CHECK-NEXT:              }) {__xtc_id_C_0_} : (f32, memref<1x1xf32, strided<[16, 1], offset: ?>>) -> ()
# CHECK-NEXT:              "scf.yield"() : () -> ()
# CHECK-NEXT:            }) {"./j"} : (index, index, index) -> ()
# CHECK-NEXT:            "scf.yield"() : () -> ()
# CHECK-NEXT:          }) {"./i"} : (index, index, index) -> ()
# CHECK-NEXT:          "scf.for"(%2, %1, %4) ({
# CHECK-NEXT:          ^bb0(%arg4: index):
# CHECK-NEXT:            %5 = "memref.subview"(%arg0, %arg4) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 8, 64>, static_strides = array<i64: 1, 1>}> : (memref<16x64xf32>, index) -> memref<8x64xf32, strided<[64, 1], offset: ?>>
# CHECK-NEXT:            %6 = "memref.subview"(%arg2, %arg4) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 8, 16>, static_strides = array<i64: 1, 1>}> : (memref<16x16xf32>, index) -> memref<8x16xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:            "scf.for"(%2, %1, %4) ({
# CHECK-NEXT:            ^bb0(%arg5: index):
# CHECK-NEXT:              %7 = "memref.subview"(%arg1, %arg5) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 64, 8>, static_strides = array<i64: 1, 1>}> : (memref<64x16xf32>, index) -> memref<64x8xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:              %8 = "memref.subview"(%6, %arg5) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: 1, 1>}> : (memref<8x16xf32, strided<[16, 1], offset: ?>>, index) -> memref<8x8xf32, strided<[16, 1], offset: ?>>
# CHECK-NEXT:              "linalg.matmul"(%5, %7, %8) <{indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
# CHECK-NEXT:              ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
# CHECK-NEXT:                %9 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
# CHECK-NEXT:                %10 = "arith.addf"(%arg8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
# CHECK-NEXT:                "linalg.yield"(%10) : (f32) -> ()
# CHECK-NEXT:              }) {__xtc_id_C_, xtc.request_vectorization} : (memref<8x64xf32, strided<[64, 1], offset: ?>>, memref<64x8xf32, strided<[16, 1], offset: ?>>, memref<8x8xf32, strided<[16, 1], offset: ?>>) -> ()
# CHECK-NEXT:              "scf.yield"() : () -> ()
# CHECK-NEXT:            }) {"./j"} : (index, index, index) -> ()
# CHECK-NEXT:            "scf.yield"() : () -> ()
# CHECK-NEXT:          }) {"./i"} : (index, index, index) -> ()
# CHECK-NEXT:          "kvxcluster.launch_terminator"() : () -> ()
# CHECK-NEXT:        }) {mask = 1 : i32, nclusters = 1 : i32} : () -> ()
# CHECK-NEXT:        "kvxcluster.await_all"() : () -> ()
# CHECK-NEXT:        "mppa.yield"() : () -> ()
# CHECK-NEXT:      }) {device = 1 : i32} : () -> ()
# CHECK-NEXT:      "func.return"() : () -> ()
# CHECK-NEXT:    }) : () -> ()
# CHECK-NEXT:  }) {transform.with_named_sequence} : () -> ()
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After MPPA Opt //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @kvxcluster_launch_0_kernel_cc_0(%arg0: memref<16x16xf32, 2>, %arg1: memref<16x64xf32, 2>, %arg2: memref<64x16xf32, 2>) attributes {kernel_for_cluster_id = 0 : index} {
# CHECK-NEXT:      %c64 = arith.constant 64 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c16 step %c1 {
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c16 step %c1 {
# CHECK-NEXT:          %0 = arith.muli %arg3, %c16 overflow<nsw> : index
# CHECK-NEXT:          %1 = arith.addi %0, %arg4 : index
# CHECK-NEXT:          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1, 1], strides: [16, 1] : memref<16x16xf32, 2> to memref<1x1xf32, strided<[16, 1], offset: ?>, 2>
# CHECK-NEXT:          kvxpe.launch %arg5 (npes=1) {
# CHECK-NEXT:            memref.store %cst, %reinterpret_cast[%c0, %c0] : memref<1x1xf32, strided<[16, 1], offset: ?>, 2>
# CHECK-NEXT:            kvxpe.launch_terminator
# CHECK-NEXT:          }
# CHECK-NEXT:          kvxpe.await_all
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c16 step %c8 {
# CHECK-NEXT:        %0 = arith.muli %arg3, %c64 overflow<nsw> : index
# CHECK-NEXT:        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [8, 64], strides: [64, 1] : memref<16x64xf32, 2> to memref<8x64xf32, strided<[64, 1], offset: ?>, 2>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c16 step %c8 {
# CHECK-NEXT:          %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%arg4], sizes: [64, 8], strides: [16, 1] : memref<64x16xf32, 2> to memref<64x8xf32, strided<[16, 1], offset: ?>, 2>
# CHECK-NEXT:          %1 = arith.muli %arg3, %c16 overflow<nsw> : index
# CHECK-NEXT:          %2 = arith.addi %1, %arg4 : index
# CHECK-NEXT:          %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [8, 8], strides: [16, 1] : memref<16x16xf32, 2> to memref<8x8xf32, strided<[16, 1], offset: ?>, 2>
# CHECK-NEXT:          kvxpe.launch %arg5 (npes=1) {
# CHECK-NEXT:            kvxuks.mma_8x8xf32 %reinterpret_cast, %reinterpret_cast_0 -> %reinterpret_cast_1 : memref<8x64xf32, strided<[64, 1], offset: ?>, 2>, memref<64x8xf32, strided<[16, 1], offset: ?>, 2>, memref<8x8xf32, strided<[16, 1], offset: ?>, 2>
# CHECK-NEXT:            kvxpe.launch_terminator
# CHECK-NEXT:          }
# CHECK-NEXT:          kvxpe.await_all
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    func.func @matmul(%arg0: memref<16x64xf32> {llvm.noalias}, %arg1: memref<64x16xf32> {llvm.noalias}, %arg2: memref<16x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      mppa.launch(k300) {
# CHECK-NEXT:        %0 = mppa.alloc : memref<16x16xf32, 2>
# CHECK-NEXT:        mppa.copy %arg2, %0 : memref<16x16xf32> to memref<16x16xf32, 2>
# CHECK-NEXT:        %1 = mppa.alloc : memref<16x64xf32, 2>
# CHECK-NEXT:        mppa.copy %arg0, %1 : memref<16x64xf32> to memref<16x64xf32, 2>
# CHECK-NEXT:        %2 = mppa.alloc : memref<64x16xf32, 2>
# CHECK-NEXT:        mppa.copy %arg1, %2 : memref<64x16xf32> to memref<64x16xf32, 2>
# CHECK-NEXT:        kvxcluster.launch (nclusters=1, mask=1) 
# CHECK-NEXT:          0 -> @kvxcluster_launch_0_kernel_cc_0
# CHECK-NEXT:          with (%0, %1, %2) : memref<16x16xf32, 2>, memref<16x64xf32, 2>, memref<64x16xf32, 2>
# CHECK-NEXT:        kvxcluster.await_all
# CHECK-NEXT:        mppa.dealloc %2 : memref<64x16xf32, 2>
# CHECK-NEXT:        mppa.dealloc %1 : memref<16x64xf32, 2>
# CHECK-NEXT:        mppa.copy %0, %arg2 : memref<16x16xf32, 2> to memref<16x16xf32>
# CHECK-NEXT:        mppa.dealloc %0 : memref<16x16xf32, 2>
# CHECK-NEXT:        kvxcluster.await_all
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 16x64xfloat32
# CHECK-NEXT:    - %1 : 64x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 16x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [16x64xfloat32, 64x16xfloat32] -> [16x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
