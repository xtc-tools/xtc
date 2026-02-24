# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_aie
# REQUIRES: mlir-target=aie

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.aie import AIEDevice
from aie.dialects.aie import AIEDevice as IronAIEDevice

import aie.iron as iron

I, dtype = 64, "float32"
a = O.tensor((I,), dtype, name="A")

with O.graph(name="relu") as gb:
    O.relu(a, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.define_memory_mesh(axes={"mx": 1})
sch.define_processor_mesh(axes={"px": 1, "psx": 4})
sch.vectorize(["i"])
sched = sch.schedule()

# Create AIE device
iron.set_current_device(IronAIEDevice.npu1_1col)
aie = AIEDevice()

comp = impl.get_compiler(
    target=aie,
    shared_lib=True,
    dump_file="relu_mlir_aie",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=False) # validation is not supported yet
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @relu(%arg0: memref<64xf32> {llvm.noalias}, %arg1: memref<64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %arg0 [[0]] : memref<64xf32> into memref<64xf32>
# CHECK-NEXT:      %collapse_shape_0 = memref.collapse_shape %arg1 [[0]] : memref<64xf32> into memref<64xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapse_shape, %cst : memref<64xf32>, f32) outs(%collapse_shape_0 : memref<64xf32>) attrs =  {__xtc_id_relu_} {
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
# CHECK-NEXT:      %0 = transform.sdist.create_memory_mesh %arg0 "memory_mesh" = <["mx"=1]> : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %1 = transform.sdist.create_processor_mesh %arg0 "processor_mesh" = <["px"=1, "psx"=4]> from "memory_mesh" : !transform.any_op -> !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.match attributes {__xtc_id_relu_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.annotate %2 "xtc.request_vectorization" : !transform.any_op
# CHECK-NEXT:      %3 = transform.get_parent_op %2 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %3 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    sdist.processor_mesh @processor_mesh from @memory_mesh = <["px"=1, "psx"=4]>
# CHECK-NEXT:    sdist.memory_mesh @memory_mesh = <["mx"=1]>
# CHECK-NEXT:    func.func @relu(%arg0: memref<64xf32> {llvm.noalias}, %arg1: memref<64xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg0, %cst : memref<64xf32>, f32) outs(%arg1 : memref<64xf32>) attrs =  {__xtc_id_relu_, xtc.request_vectorization} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.maximumf %in, %in_0 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1 : 64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %1: relu(%0) {name = 'relu'} : [64xfloat32] -> [64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    "aie.device"() ({
# CHECK-NEXT:      func.func private @mock(i32)
# CHECK-NEXT:      %0 = "aie.tile"() {col = 0 : i32, row = 0 : i32} : () -> index
# CHECK-NEXT:      %1 = "aie.tile"() {col = 0 : i32, row = 2 : i32} : () -> index
# CHECK-NEXT:      "aie.objectfifo"(%0, %1) {dimensionsFromStreamPerConsumer = #aie<bd_dim_layout_array_array[[]]>, dimensionsToStream = #aie<bd_dim_layout_array[]>, disable_synchronization = false, elemNumber = 2 : i32, elemType = !aie.objectfifo<memref<64xf32>>, plio = false, sym_name = "in1", via_DMA = false} : (index, index) -> ()
# CHECK-NEXT:      "aie.objectfifo"(%1, %0) {dimensionsFromStreamPerConsumer = #aie<bd_dim_layout_array_array[[]]>, dimensionsToStream = #aie<bd_dim_layout_array[]>, disable_synchronization = false, elemNumber = 2 : i32, elemType = !aie.objectfifo<memref<64xf32>>, plio = false, sym_name = "out1", via_DMA = false} : (index, index) -> ()
# CHECK-NEXT:      %2 = "aie.core"(%1) ({
# CHECK-NEXT:        %c1 = arith.constant 1 : index
# CHECK-NEXT:        %c9223372036854775807 = arith.constant 9223372036854775807 : index
# CHECK-NEXT:        %c0 = arith.constant 0 : index
# CHECK-NEXT:        %c64 = arith.constant 64 : index
# CHECK-NEXT:        %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:        scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
# CHECK-NEXT:          %3 = "aie.objectfifo.acquire"() {objFifo_name = @in1, port = 1 : i32, size = 1 : i32} : () -> !aie.objectfifosubview<memref<64xf32>>
# CHECK-NEXT:          %4 = "aie.objectfifo.subview.access"(%3) {index = 0 : i32, objFifo_name = @in1} : (!aie.objectfifosubview<memref<64xf32>>) -> memref<64xf32>
# CHECK-NEXT:          %5 = "aie.objectfifo.acquire"() {objFifo_name = @out1, port = 0 : i32, size = 1 : i32} : () -> !aie.objectfifosubview<memref<64xf32>>
# CHECK-NEXT:          %6 = "aie.objectfifo.subview.access"(%5) {index = 0 : i32, objFifo_name = @out1} : (!aie.objectfifosubview<memref<64xf32>>) -> memref<64xf32>
# CHECK-NEXT:          scf.for %arg1 = %c0 to %c64 step %c1 {
# CHECK-NEXT:            %7 = memref.load %4[%arg1] : memref<64xf32>
# CHECK-NEXT:            %8 = arith.maximumf %7, %cst : f32
# CHECK-NEXT:            memref.store %8, %6[%arg1] : memref<64xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:          "aie.objectfifo.release"() {objFifo_name = @in1, port = 1 : i32, size = 1 : i32} : () -> ()
# CHECK-NEXT:          "aie.objectfifo.release"() {objFifo_name = @out1, port = 0 : i32, size = 1 : i32} : () -> ()
# CHECK-NEXT:        }
# CHECK-NEXT:        "aie.end"() : () -> ()
# CHECK-NEXT:      }) {stack_size = 1024 : i32} : (index) -> index
# CHECK-NEXT:      "aiex.runtime_sequence"() ({
# CHECK-NEXT:      ^bb0(%arg0: memref<64xf32>, %arg1: memref<64xf32>):
# CHECK-NEXT:        %3 = "aiex.dma_configure_task_for"() ({
# CHECK-NEXT:          "aie.dma_bd"(%arg1) {burst_length = 0 : i32, dimensions = #aie<bd_dim_layout_array[<size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>, <size = 64, stride = 1>]>, len = 64 : i32, offset = 0 : i32} : (memref<64xf32>) -> ()
# CHECK-NEXT:          "aie.end"() : () -> ()
# CHECK-NEXT:        }) {alloc = @out1, issue_token = true} : () -> index
# CHECK-NEXT:        %4 = "aiex.dma_configure_task_for"() ({
# CHECK-NEXT:          "aie.dma_bd"(%arg0) {burst_length = 0 : i32, dimensions = #aie<bd_dim_layout_array[<size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>, <size = 64, stride = 1>]>, len = 64 : i32, offset = 0 : i32} : (memref<64xf32>) -> ()
# CHECK-NEXT:          "aie.end"() : () -> ()
# CHECK-NEXT:        }) {alloc = @in1} : () -> index
# CHECK-NEXT:        "aiex.dma_start_task"(%4) : (index) -> ()
# CHECK-NEXT:        "aiex.dma_start_task"(%3) : (index) -> ()
# CHECK-NEXT:        "aiex.dma_await_task"(%3) : (index) -> ()
# CHECK-NEXT:        "aiex.dma_free_task"(%4) : (index) -> ()
# CHECK-NEXT:      }) {sym_name = "sequence"} : () -> ()
# CHECK-NEXT:      "aie.end"() : () -> ()
# CHECK-NEXT:    }) {device = 5 : i32, sym_name = "main"} : () -> ()
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK:       [INFO]   : Bootimage generated successfully
# CHECK:       (XTC: Proper runtime evaluation harness is not supported yet)
# CHECK:       CODE: 0
